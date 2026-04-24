"""Profile one training epoch + one validation epoch of HVAE.

Usage:
    python profile_epoch.py --images <path> --genetics <path> --covariates <path> \
        --hweights <path> --decode-covariates <file> --residualize-covariates <file>

Prints a breakdown of time spent in each phase.
"""

import sys
import time
import os

import torch
import numpy as np

from train_hvae import (
    parse_args, setup_output_dirs, setup_logging, load_split_ids,
    find_latest_checkpoint, save_split_ids, setup_heritability,
    encode_all, composite_loss, compute_heritability_loss,
    compute_correlation_loss, compute_moment_loss, _compute_her_estimates,
    center_and_scale, HVAEConfig, HeritabilityState, read_lines,
)
from h2vae.model import VAE
from h2vae.data import ImageDataset, ImageFileDataset, load_data, load_genetics_reindexed
from torch.utils.data import DataLoader


class Timer:
    def __init__(self):
        self.records = {}

    def __call__(self, name):
        return TimerContext(self, name)

    def summary(self):
        total = sum(v for v in self.records.values())
        print(f"\n{'Phase':<40} {'Time (s)':>10} {'%':>6}")
        print("-" * 58)
        for name, t in sorted(self.records.items(), key=lambda x: -x[1]):
            print(f"{name:<40} {t:>10.3f} {100*t/total:>5.1f}%")
        print("-" * 58)
        print(f"{'TOTAL':<40} {total:>10.3f}")


class TimerContext:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name

    def __enter__(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - self.start
        self.timer.records[self.name] = self.timer.records.get(self.name, 0) + elapsed


def main():
    cfg = parse_args()
    outdir = "/tmp/profile_hvae"
    cfg.outdir = outdir
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "weights"), exist_ok=True)

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

    data = load_data(
        images_path=cfg.images,
        genetics_path=genetics_path,
        covariates_path=cfg.covariates,
        seed=cfg.seed,
        required_covariates=required_covs or None,
    )

    if cfg.split_variants:
        odd_path = f"{cfg.genetics}.odd.hdf5"
        all_ids = np.concatenate([data["train_ids_raw"], data["val_ids_raw"]])
        data["genetics_odd"] = load_genetics_reindexed(odd_path, all_ids)

    if data["streaming"]:
        train_dataset = ImageFileDataset(data["train"]["image_paths"], data["train"]["ids"])
        val_dataset = ImageFileDataset(data["val"]["image_paths"], data["val"]["ids"])
    else:
        train_dataset = ImageDataset(data["train"]["images"], data["train"]["ids"])
        val_dataset = ImageDataset(data["val"]["images"], data["val"]["ids"])

    loader_kwargs = {}
    if data["streaming"]:
        loader_kwargs.update(num_workers=8, prefetch_factor=2, pin_memory=True)

    train_loader = DataLoader(train_dataset, batch_size=cfg.bs, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=cfg.bs, shuffle=False, **loader_kwargs)

    h_state = setup_heritability(cfg, data)

    vae_cfg = cfg.vae_cfg
    if h_state.cov_state.decode_train is not None:
        vae_cfg["external"] = h_state.cov_state.decode_train.shape[1]
    else:
        vae_cfg["external"] = 0

    vae = VAE(**vae_cfg).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.vae_lr)
    weights = cfg.loss_weights

    n_train = data["n_train"]
    print(f"n_train={n_train}, n_val={len(val_dataset)}, zdim={cfg.zdim}, bs={cfg.bs}")
    print(f"kinship={cfg.kinship}, use_taylor={h_state.use_taylor}, split_variants={cfg.split_variants}")
    print(f"h_weight={cfg.h_weight}, corr_weight={cfg.corr_weight}, sk_weight={cfg.sk_weight}")
    print(f"decode_covariates={'yes' if cfg.decode_covariates else 'no'}")
    print(f"residualize_covariates={'yes' if cfg.residualize_covariates else 'no'}")

    timer = Timer()

    # === TRAINING EPOCH ===
    print("\n=== Training Epoch ===")

    with timer("train/1_encode_all"):
        Zm, Zs = encode_all(vae, train_loader, cfg.zdim, device)
        Eps = torch.randn_like(Zs)
        Z = Zm + Eps * Zs

    if h_state.use_taylor:
        with timer("train/1.5_taylor_factory"):
            cs_Z_ref = center_and_scale(Z.detach())
            her_loss_fn = [
                h_state.taylor_train.make_loss(cs_Z_ref[:, i:i + 1])
                for i in range(cfg.zdim)
            ]
    else:
        her_loss_fn = h_state.loss_fn

    vae.train()
    for batch_i, batch_data in enumerate(train_loader):
        Z = Z.detach()
        y = batch_data[0].to(device)
        idxs = batch_data[-1]
        eps = Eps[idxs]

        c_batch = None
        if h_state.cov_state.decode_train is not None:
            c_batch = h_state.cov_state.decode_train[idxs]

        with timer("train/2a_encode_batch"):
            zm, zs = vae.encode(y)
            z = zm + zs * eps

        with timer("train/2b_forward"):
            vae_loss, mse, kld = vae.forward(y, eps, c_batch)

        Z[idxs] = z

        with timer("train/2c_center_and_scale"):
            cs_Z = center_and_scale(Z)

        with timer("train/2d_vae_loss_sum"):
            _vae = weights.mse * vae_loss.sum()

        if weights.h > 0:
            with timer("train/2e_heritability_loss"):
                h_loss = compute_heritability_loss(cs_Z, her_loss_fn, h_state.hweights, idxs=idxs)

        if weights.corr > 0:
            with timer("train/2f_correlation_loss"):
                c_loss = compute_correlation_loss(cs_Z)

        if weights.sk > 0:
            with timer("train/2g_moment_loss"):
                sk_loss = compute_moment_loss(cs_Z)

        with timer("train/2h_kl_penalty"):
            from train_hvae import compute_kl_penalty
            pen = compute_kl_penalty(zs, vae.K)

        # Assemble the actual loss used for backprop
        loss = _vae
        if weights.h > 0:
            loss = loss + weights.h * h_loss
        if weights.corr > 0:
            loss = loss + weights.corr * c_loss
        if weights.sk > 0:
            loss = loss + weights.sk * sk_loss
        loss = loss + pen

        with timer("train/2i_backward"):
            optimizer.zero_grad()
            loss.backward()

        with timer("train/2j_clip_and_step"):
            if cfg.clip > 0:
                torch.nn.utils.clip_grad_norm_(vae.parameters(), cfg.clip)
            optimizer.step()

    with timer("train/3_display_estimates"):
        result = {}
        result["her_estimates"] = _compute_her_estimates(
            Z, h_state.loss_fn, h_state.taylor_train if h_state.use_taylor else None,
        )

    if cfg.split_variants:
        with timer("train/3_display_estimates_odd"):
            result["her_estimates_odd"] = _compute_her_estimates(
                Z, h_state.loss_fn_odd, h_state.taylor_train_odd if h_state.use_taylor else None,
            )

    # === VALIDATION EPOCH ===
    print("\n=== Validation Epoch ===")

    with timer("val/1_encode_all"):
        Zm_v, Zs_v = encode_all(vae, val_loader, cfg.zdim, device)
        Eps_v = torch.randn_like(Zs_v)
        Z_v = Zm_v + Eps_v * Zs_v

    with timer("val/2_display_estimates"):
        val_result = {}
        val_result["her_estimates_val"] = _compute_her_estimates(
            Z_v, h_state.val_fn, h_state.taylor_val if h_state.use_taylor else None,
        )

    if cfg.split_variants:
        with timer("val/2_display_estimates_odd"):
            val_result["her_estimates_val_odd"] = _compute_her_estimates(
                Z_v, h_state.val_fn_odd, h_state.taylor_val_odd if h_state.use_taylor else None,
            )

    with timer("val/3_mse"):
        mse_val = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                y = batch_data[0].to(device)
                idxs_v = batch_data[-1]
                z_v = Z_v[idxs_v]
                c_batch = None
                if h_state.cov_state.decode_val is not None:
                    c_batch = h_state.cov_state.decode_val[idxs_v]
                mse_v = vae.mse(y, vae.decode(z_v, c_batch))
                mse_val += mse_v.sum().item()

    timer.summary()


if __name__ == "__main__":
    main()
