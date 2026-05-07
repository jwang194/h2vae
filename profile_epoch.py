"""Profile one training + one validation epoch of HVAE.

Tracks the same flow as train_hvae.py (encode-all → backprop loop → re-encode
→ savetxt → display, then val encode → savetxt → display → second-pass MSE)
and decomposes each phase so the operator can see where time goes — especially
for the vae3d / NIfTI branch where DataLoader decompression and the third
encode pass are likely the dominant levers.

Numerical-precision protection: this script does NOT enable TF32 or autocast
anywhere; heritability matmuls run in fp32 to match what production training
sees today. Heritability speedups should be evaluated in a separate harness.

Usage (NIfTI / vae3d):
    python profile_epoch.py --model vae3d \\
        --images data/images/t1.tsv \\
        --genetics data/genetics/mri_kinship.hdf5 \\
        --covariates data/covariates/PC1_40_Age_Sex_ICV.ukb.hdf5 \\
        --hweights aux/uniform.128.weights \\
        --decode-covariates aux/PC1_40_Age_Sex.covariates \\
        --residualize-covariates aux/ICV.covariates \\
        --kinship --zdim 128 --bs 8

Two epochs are run; the first is discarded as warmup (cuDNN algo selection,
DataLoader worker startup) and the second is reported.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

# Reuse training primitives so the profiler tracks train_hvae.py automatically.
from train_hvae import (
    HVAEConfig, parse_args,
    setup_heritability, encode_all,
    compute_heritability_loss, compute_correlation_loss,
    compute_moment_loss, compute_kl_penalty, _compute_her_estimates,
    read_lines,
)
from h2vae.models import get_model_class
from h2vae.data import (
    ImageDataset, load_data, load_genetics_reindexed, make_streaming_dataset,
)
from h2vae.latent_utils import center_and_scale


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class Timer:
    """Aggregates per-phase wall-clock times across multiple invocations."""

    def __init__(self, sync: bool = True):
        self.records: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.sync = sync and torch.cuda.is_available()

    @contextmanager
    def __call__(self, name: str):
        if self.sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            self.records[name] = self.records.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1

    def reset(self) -> None:
        self.records.clear()
        self.counts.clear()

    def report(self, header: str = "") -> None:
        total = sum(self.records.values())
        if header:
            print(f"\n=== {header} ===")
        print(f"{'Phase':<42} {'Total (s)':>10} {'Calls':>7} {'%':>6}")
        print("-" * 70)
        for name, t in sorted(self.records.items(), key=lambda x: -x[1]):
            n = self.counts.get(name, 0)
            pct = 100 * t / total if total > 0 else 0.0
            print(f"{name:<42} {t:>10.3f} {n:>7d} {pct:>5.1f}%")
        print("-" * 70)
        print(f"{'TOTAL':<42} {total:>10.3f}")


# ---------------------------------------------------------------------------
# Phase-decomposed train / val (mirrors train_hvae.train_epoch / validate_epoch)
# ---------------------------------------------------------------------------

def _next_with_wait(it, timer: Timer, label: str):
    """Advance an iterator, charging blocking time to ``label``."""
    with timer(label):
        return next(it)


def profile_train_epoch(vae, train_loader, optimizer, h_state, cfg, weights,
                        timer: Timer, epoch: int, latent_dir: str) -> None:
    device = cfg.device

    # --- Phase 1: encode-all (start of epoch) ---
    vae.eval()
    n = len(train_loader.dataset)
    Zm = torch.zeros(n, cfg.zdim, device=device)
    Zs = torch.zeros(n, cfg.zdim, device=device)
    with torch.no_grad():
        it = iter(train_loader)
        while True:
            try:
                batch = _next_with_wait(it, timer, "train/encode_all/dataloader_wait")
            except StopIteration:
                break
            with timer("train/encode_all/h2d_copy"):
                y = batch[0].to(device)
            idxs = batch[-1]
            with timer("train/encode_all/encoder_forward"):
                zm, zs = vae.encode(y)
                Zm[idxs] = zm
                Zs[idxs] = zs
    Eps = torch.randn_like(Zs)
    Z = Zm + Eps * Zs

    her_loss_fn = h_state.loss_fn

    # --- Phase 2: backprop loop ---
    vae.train()
    it = iter(train_loader)
    while True:
        try:
            batch = _next_with_wait(it, timer, "train/backprop/dataloader_wait")
        except StopIteration:
            break
        Z = Z.detach()
        with timer("train/backprop/h2d_copy"):
            y = batch[0].to(device)
        idxs = batch[-1]
        eps = Eps[idxs]

        c_batch = None
        if h_state.cov_state.decode_train is not None:
            c_batch = h_state.cov_state.decode_train[idxs]

        with timer("train/backprop/encoder_forward"):
            zm, zs = vae.encode(y)
            z = zm + zs * eps
        with timer("train/backprop/decoder_forward"):
            xr = vae.decode(z, c_batch)
        with timer("train/backprop/mse_kld"):
            mse = vae.mse(y, xr)
            kld = (
                -0.5 * (1 + 2 * torch.log(zs) - zm ** 2 - zs ** 2).sum(1)[:, None]
                / vae.K
            )
            vae_loss = mse + vae.beta * kld

        Z[idxs] = z

        # Reproduce composite_loss assembly with per-component timers.
        with timer("train/backprop/composite_assembly"):
            loss = weights.mse * vae_loss.sum()

        if weights.h > 0:
            with timer("train/backprop/heritability_loss"):
                h_loss = compute_heritability_loss(
                    Z, her_loss_fn, h_state.hweights, idxs=idxs,
                )
            with timer("train/backprop/composite_assembly"):
                loss = loss + weights.h * h_loss

        if weights.corr > 0 or weights.sk > 0:
            with timer("train/backprop/composite_assembly"):
                cs_Z = center_and_scale(Z)
            if weights.corr > 0:
                with timer("train/backprop/corr_loss"):
                    c_loss = compute_correlation_loss(cs_Z)
                with timer("train/backprop/composite_assembly"):
                    loss = loss + weights.corr * c_loss
            if weights.sk > 0:
                with timer("train/backprop/moment_loss"):
                    sk_loss = compute_moment_loss(cs_Z)
                with timer("train/backprop/composite_assembly"):
                    loss = loss + weights.sk * sk_loss

        with timer("train/backprop/kl_penalty"):
            pen = compute_kl_penalty(zs, vae.K)
        with timer("train/backprop/composite_assembly"):
            loss = loss + pen

        with timer("train/backprop/backward"):
            optimizer.zero_grad()
            loss.backward()
        with timer("train/backprop/grad_clip"):
            if cfg.clip > 0:
                torch.nn.utils.clip_grad_norm_(vae.parameters(), cfg.clip)
        with timer("train/backprop/optimizer_step"):
            optimizer.step()

    # --- Phase 3: post-train re-encode + savetxt + display ---
    with timer("train/post/re_encode_all"):
        Zm_post, _ = encode_all(vae, train_loader, cfg.zdim, device)

    with timer("train/post/savetxt_latents"):
        np.savetxt(
            os.path.join(latent_dir, f"Zm_train.{epoch:05d}.txt"),
            Zm_post.detach().cpu().numpy(),
            delimiter="\t",
        )

    with timer("train/post/display_her_even"):
        _ = _compute_her_estimates(Zm_post, h_state.loss_fn)
    if cfg.split_variants:
        with timer("train/post/display_her_odd"):
            _ = _compute_her_estimates(Zm_post, h_state.loss_fn_odd)


def profile_val_epoch(vae, val_loader, h_state, cfg, timer: Timer,
                      epoch: int, latent_dir: str) -> None:
    device = cfg.device

    # --- val encode-all ---
    vae.eval()
    n = len(val_loader.dataset)
    Zm = torch.zeros(n, cfg.zdim, device=device)
    Zs = torch.zeros(n, cfg.zdim, device=device)
    with torch.no_grad():
        it = iter(val_loader)
        while True:
            try:
                batch = _next_with_wait(it, timer, "val/encode_all/dataloader_wait")
            except StopIteration:
                break
            with timer("val/encode_all/h2d_copy"):
                y = batch[0].to(device)
            idxs = batch[-1]
            with timer("val/encode_all/encoder_forward"):
                zm, zs = vae.encode(y)
                Zm[idxs] = zm
                Zs[idxs] = zs

    # --- val savetxt + display ---
    with timer("val/post/savetxt_latents"):
        np.savetxt(
            os.path.join(latent_dir, f"Zm_val.{epoch:05d}.txt"),
            Zm.detach().cpu().numpy(),
            delimiter="\t",
        )

    with timer("val/post/display_her_even"):
        _ = _compute_her_estimates(Zm, h_state.val_fn)
    if cfg.split_variants:
        with timer("val/post/display_her_odd"):
            _ = _compute_her_estimates(Zm, h_state.val_fn_odd)

    # --- val MSE second pass (re-iterates val_loader) ---
    with torch.no_grad():
        it = iter(val_loader)
        while True:
            try:
                batch = _next_with_wait(it, timer, "val/mse_pass/dataloader_wait")
            except StopIteration:
                break
            with timer("val/mse_pass/h2d_copy"):
                y = batch[0].to(device)
            idxs = batch[-1]
            z = Zm[idxs]
            c_batch = None
            if h_state.cov_state.decode_val is not None:
                c_batch = h_state.cov_state.decode_val[idxs]
            with timer("val/mse_pass/decoder_forward"):
                xr = vae.decode(z, c_batch)
            with timer("val/mse_pass/mse_compute"):
                _ = vae.mse(y, xr)


# ---------------------------------------------------------------------------
# Phase → SPEEDUPS.md mapping for the actionable-hints block
# ---------------------------------------------------------------------------

PHASE_HINTS = {
    "train/encode_all/dataloader_wait": "SPEEDUPS #14 (preprocess NIfTI), #19 (workers), #8 (non_blocking)",
    "train/backprop/dataloader_wait":  "SPEEDUPS #14, #19, #8",
    "val/encode_all/dataloader_wait":  "SPEEDUPS #14, #19, #8",
    "val/mse_pass/dataloader_wait":    "SPEEDUPS #5 (fold MSE into val encode); #14, #19, #8",
    "train/post/re_encode_all":         "SPEEDUPS #2 (drop the third encode; reuse next-epoch start)",
    "train/post/savetxt_latents":       "SPEEDUPS #3 (np.save instead of np.savetxt)",
    "val/post/savetxt_latents":         "SPEEDUPS #3",
    "train/post/display_her_even":      "SPEEDUPS #1 (vectorize per-dim heritability — display only)",
    "train/post/display_her_odd":       "SPEEDUPS #1",
    "val/post/display_her_even":        "SPEEDUPS #1",
    "val/post/display_her_odd":         "SPEEDUPS #1",
    "train/backprop/encoder_forward":  "SPEEDUPS #4 (cuDNN bench + TF32), #6 (AMP/bf16), #13 (torch.compile)",
    "train/backprop/decoder_forward":  "SPEEDUPS #4, #6, #13",
    "train/encode_all/encoder_forward": "SPEEDUPS #4, #6, #7 (bigger encode-only batch), #13",
    "val/encode_all/encoder_forward":  "SPEEDUPS #4, #6, #7, #13, #5",
    "val/mse_pass/decoder_forward":    "SPEEDUPS #5 (avoid the second pass entirely), #4, #6, #13",
    "train/backprop/backward":          "SPEEDUPS #4, #6, #13",
    "train/backprop/heritability_loss": "Heritability path — KEEP fp32. SPEEDUPS #9 (skip the Z clone), #10 (apply P implicitly).",
}


def print_actionable_hints(timer: Timer, top_k: int = 10) -> None:
    total = sum(timer.records.values())
    if total <= 0:
        return
    print(f"\n=== Actionable hints (top {top_k} phases by time) ===")
    ranked = sorted(timer.records.items(), key=lambda x: -x[1])[:top_k]
    for name, t in ranked:
        pct = 100 * t / total
        hint = PHASE_HINTS.get(name, "(no direct SPEEDUPS mapping)")
        print(f"{pct:>5.1f}%  {name:<40s}  → {hint}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_loader(dataset, cfg, streaming: bool, data_format: str,
                  shuffle: bool, num_workers: int | None) -> DataLoader:
    loader_kwargs: dict = {}
    if streaming:
        nw = num_workers if num_workers is not None else (
            16 if data_format == "nifti" else 8
        )
        loader_kwargs.update(
            num_workers=nw,
            prefetch_factor=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
        )
    return DataLoader(dataset, batch_size=cfg.bs, shuffle=shuffle, **loader_kwargs)


def main() -> None:
    # Extra CLI flags on top of train_hvae's parse_args.
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument("--profile-epochs", type=int, default=2,
                                help="number of epochs to run; first is warmup")
    profile_parser.add_argument("--nw", type=int, default=None,
                                help="DataLoader num_workers override (streaming only)")
    profile_parser.add_argument("--profile-outdir", type=str, default="/tmp/profile_hvae",
                                help="scratch directory for latent dumps")
    extra_args, remaining = profile_parser.parse_known_args()

    # Restore sys.argv so train_hvae.parse_args sees only its own flags.
    sys.argv = [sys.argv[0]] + remaining
    cfg: HVAEConfig = parse_args()

    # Force fp32 throughout to match production heritability semantics.
    # (We are measuring the baseline; SPEEDUPS #4/#6 should be evaluated separately.)
    torch.backends.cudnn.benchmark = False  # explicit; true baseline

    cfg.outdir = extra_args.profile_outdir
    latent_dir = os.path.join(cfg.outdir, "latents")
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.outdir, "weights"), exist_ok=True)

    torch.manual_seed(cfg.seed)
    device = cfg.device

    # --- Data ---
    genetics_path = cfg.genetics
    if cfg.split_variants:
        genetics_path = f"{cfg.genetics}.even.hdf5"

    required_covs: list[str] = []
    for path in (cfg.decode_covariates, cfg.residualize_covariates):
        if path is not None:
            required_covs.extend(read_lines(path))
    required_covs = list(dict.fromkeys(required_covs))

    setup_timer = Timer()
    with setup_timer("setup/data_load"):
        data = load_data(
            images_path=cfg.images,
            genetics_path=genetics_path,
            covariates_path=cfg.covariates,
            target_phenotype_path=cfg.genetic_correlation,
            train_frac=cfg.train_frac,
            seed=cfg.seed,
            required_covariates=required_covs or None,
        )

    if cfg.split_variants:
        with setup_timer("setup/data_load"):
            odd_path = f"{cfg.genetics}.odd.hdf5"
            all_ids = np.concatenate([data["train_ids_raw"], data["val_ids_raw"]])
            data["genetics_odd"] = load_genetics_reindexed(odd_path, all_ids)

    # Mirror train_hvae.py model + dataset construction.
    ModelClass = get_model_class(cfg.model)
    streaming_kwargs: dict = {}
    if ModelClass.data_format == "nifti":
        model_default = inspect.signature(ModelClass.__init__).parameters["img_size"].default
        streaming_kwargs["target_size"] = (
            cfg.img_size if cfg.img_size is not None else model_default
        )

    if data["streaming"]:
        train_dataset = make_streaming_dataset(
            data["train"]["image_paths"], data["train"]["ids"],
            ModelClass.data_format, **streaming_kwargs,
        )
        val_dataset = make_streaming_dataset(
            data["val"]["image_paths"], data["val"]["ids"],
            ModelClass.data_format, **streaming_kwargs,
        )
    else:
        train_dataset = ImageDataset(data["train"]["images"], data["train"]["ids"])
        val_dataset = ImageDataset(data["val"]["images"], data["val"]["ids"])

    train_loader = _build_loader(
        train_dataset, cfg, data["streaming"], ModelClass.data_format,
        shuffle=True, num_workers=extra_args.nw,
    )
    val_loader = _build_loader(
        val_dataset, cfg, data["streaming"], ModelClass.data_format,
        shuffle=False, num_workers=extra_args.nw,
    )

    # --- Heritability ---
    with setup_timer("setup/heritability_factorize"):
        h_state = setup_heritability(cfg, data)

    # Sanity-check: heritability matrices must be fp32.
    if cfg.kinship and data.get("genetics", {}).get("kinship") is not None:
        # The setup callable closes over the device-side K; we can't reach it
        # directly, but a small forward call reveals the working dtype.
        _probe = torch.randn(data["n_train"], 1, device=device, dtype=torch.float32)
        out = h_state.loss_fn(_probe)
        assert out.dtype == torch.float32, f"heritability loss returned {out.dtype}, expected float32"

    # --- Model ---
    with setup_timer("setup/model_build"):
        vae_cfg = cfg.vae_cfg
        if h_state.cov_state.decode_train is not None:
            vae_cfg["external"] = h_state.cov_state.decode_train.shape[1]
        else:
            vae_cfg["external"] = 0
        constructor_cfg = {k: v for k, v in vae_cfg.items() if not k.startswith("_")}
        vae = ModelClass(**constructor_cfg).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=cfg.vae_lr)
    weights = cfg.loss_weights

    # --- Header ---
    n_train = data["n_train"]
    n_val = len(val_dataset)
    print("\n=== profile_epoch.py — config ===")
    print(f"  model              = {cfg.model}")
    print(f"  data_format        = {ModelClass.data_format}")
    print(f"  streaming          = {data['streaming']}")
    print(f"  n_train            = {n_train}")
    print(f"  n_val              = {n_val}")
    print(f"  zdim               = {cfg.zdim}")
    print(f"  bs                 = {cfg.bs}")
    print(f"  num_workers        = {train_loader.num_workers}")
    print(f"  kinship            = {cfg.kinship}")
    print(f"  split_variants     = {cfg.split_variants}")
    print(f"  decode_cov         = {'yes' if cfg.decode_covariates else 'no'}")
    print(f"  residualize_cov    = {'yes' if cfg.residualize_covariates else 'no'}")
    print(f"  h_weight           = {cfg.h_weight}  corr={cfg.corr_weight}  sk={cfg.sk_weight}")
    print(f"  cuDNN benchmark    = {torch.backends.cudnn.benchmark}")
    print(f"  fp32 matmul prec   = {torch.get_float32_matmul_precision()}")
    print(f"  autocast active    = no (none enabled in this profile)")
    print(f"  profile_epochs     = {extra_args.profile_epochs}  (first = warmup, last = reported)")
    print()
    print("Note: TimerContext brackets every phase with torch.cuda.synchronize(),")
    print("      so totals are upper bounds (no overlap). This is intentional —")
    print("      it gives clean per-phase attribution at the cost of overstating")
    print("      total epoch wall-clock by some IO/compute overlap.")

    setup_timer.report("Setup (one-time costs)")

    # --- Warmup epoch + reported epoch(s) ---
    timer = Timer()
    for ep in range(extra_args.profile_epochs):
        is_warmup = ep < extra_args.profile_epochs - 1
        if is_warmup:
            print(f"\n=== Warmup epoch {ep} (results discarded) ===")
            warm_timer = Timer()
            profile_train_epoch(vae, train_loader, optimizer, h_state, cfg,
                                weights, warm_timer, ep, latent_dir)
            profile_val_epoch(vae, val_loader, h_state, cfg, warm_timer,
                              ep, latent_dir)
        else:
            print(f"\n=== Reported epoch {ep} ===")
            profile_train_epoch(vae, train_loader, optimizer, h_state, cfg,
                                weights, timer, ep, latent_dir)
            profile_val_epoch(vae, val_loader, h_state, cfg, timer,
                              ep, latent_dir)

    timer.report("Profile (reported epoch)")
    print_actionable_hints(timer, top_k=10)


if __name__ == "__main__":
    main()
