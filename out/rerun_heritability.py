"""Recompute per-epoch heritability for a finished run against a new genetics file.

Reads saved latents (``<outdir>/latents/Zm_{train,val}.{epoch:05d}.txt``) and
the split IDs (``<outdir>/train_ids.npy``, ``val_ids.npy``), builds heritability
estimators against a new ``--genetics`` input, and writes log lines in the
format consumed by ``out/plot_heritability.py``.

No VAE forward pass — the trained latents are taken as given.

Default output path is ``<outdir>/log.<genetics>[.<target>].txt``, where
``<genetics>`` is the basename of ``--genetics`` (with ``.hdf5`` stripped)
and ``<target>`` is present only when ``--genetic-correlation`` is set.
Pass ``--out -`` to write to stdout instead, or ``--out <path>`` for a
specific file.

Usage:
    python3 out/rerun_heritability.py out/my_run \\
        --genetics data/genetics/height_25 \\
        --kinship --split-variants \\
        --covariates data/covariates/PC1_40_Age_Sex_ICV.ukb.hdf5 \\
        --residualize-covariates aux/PC1_40_Age_Sex.covariates
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np
import torch

# Project-local imports: run from project root (parent of out/).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from h2vae.data import load_genetics_reindexed, _reindex
from h2vae.heritability import mom, var_exp, gc


_EPOCH_FROM_NAME = re.compile(r"Zm_train\.(\d+)\.txt$")


def _read_lines(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _load_covariate_subset(
    path: str,
    all_ids: np.ndarray,
    selected_names: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Load a covariates HDF5, select named columns, reindex to all_ids."""
    with h5py.File(path, "r") as f:
        cov_ids = np.array(f["ids"])
        data = np.array(f["data"])
        names = [
            n.decode() if isinstance(n, bytes) else n
            for n in f["covariate_names"][:]
        ]
    name_to_col = {n: i for i, n in enumerate(names)}
    missing = [n for n in selected_names if n not in name_to_col]
    if missing:
        raise ValueError(f"Covariate names not found: {missing}. Available: {names}")
    col_idx = [name_to_col[n] for n in selected_names]

    row_idx = _reindex(cov_ids, all_ids)
    C = data[row_idx][:, col_idx].astype(np.float32)
    return torch.tensor(C, device=device)


def _compute_h2(Z: torch.Tensor, her_fn) -> list[float]:
    """Per-dimension heritability display — mirrors train_hvae._compute_her_estimates.

    Routes through ``her_fn.display`` when present (gc mode) so the logged
    values are bounded correlations rather than the raw covariance loss.
    """
    fn = getattr(her_fn, "display", her_fn)
    Z = Z.detach()
    return [fn(Z[:, i:i + 1]).item() for i in range(Z.shape[1])]


def _fmt(values: list[float]) -> str:
    return ", ".join(f"{v:.3f}" for v in values)


def _basename_tag(path: str) -> str:
    """Filesystem-safe tag derived from a genetics/phenotype path.

    Strips directory, ``.hdf5`` suffix (so ``data/genetics/height_25.hdf5``
    and the split-variants prefix ``data/genetics/height_25`` both yield
    ``height_25``).
    """
    base = os.path.basename(path.rstrip("/"))
    if base.endswith(".hdf5"):
        base = base[: -len(".hdf5")]
    return base


def _default_out_path(outdir: str, genetics: str, target_phenotype: str | None) -> str:
    parts = [_basename_tag(genetics)]
    if target_phenotype is not None:
        parts.append(_basename_tag(target_phenotype))
    return os.path.join(outdir, "log." + ".".join(parts) + ".txt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", type=str, help="training output directory (has latents/, train_ids.npy, val_ids.npy)")
    ap.add_argument("--genetics", type=str, required=True,
                    help="genetics HDF5 file, or prefix when --split-variants is set")
    ap.add_argument("--kinship", action="store_true", default=False,
                    help="use kinship matrix with MoM loss (else use genotypes with var_exp)")
    ap.add_argument("--split-variants", action="store_true", default=False,
                    help="--genetics is a prefix; load {prefix}.even.hdf5 and {prefix}.odd.hdf5")
    ap.add_argument("--covariates", type=str, default=None,
                    help="covariates HDF5 file (needed for --residualize-covariates)")
    ap.add_argument("--residualize-covariates", type=str, default=None,
                    help="text file of covariate names to project out before heritability estimation")
    ap.add_argument("--genetic-correlation", type=str, default=None,
                    help="HDF5 (keys 'data', 'ids') of a target phenotype; switches to per-latent "
                         "SCORE-OVERLAP genetic correlation with this phenotype")
    ap.add_argument("--which-cuda", type=int, default=0)
    ap.add_argument("--out", type=str, default=None,
                    help="output path (default: <outdir>/log.<genetics>[.<target>].txt; "
                         "pass '-' for stdout)")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.which_cuda}" if torch.cuda.is_available() else "cpu")

    # --- Split IDs ---
    train_ids = np.load(os.path.join(args.outdir, "train_ids.npy"))
    val_ids = np.load(os.path.join(args.outdir, "val_ids.npy"))
    all_ids = np.concatenate([train_ids, val_ids])
    n_train = len(train_ids)

    # --- Genetics ---
    if args.split_variants:
        even_path = f"{args.genetics}.even.hdf5"
        odd_path = f"{args.genetics}.odd.hdf5"
        g_main = load_genetics_reindexed(even_path, all_ids)
        g_odd = load_genetics_reindexed(odd_path, all_ids)
    else:
        g_main = load_genetics_reindexed(args.genetics, all_ids)
        g_odd = None

    # --- Covariates (for residualization) ---
    C_train = C_val = None
    if args.residualize_covariates is not None:
        if args.covariates is None:
            raise ValueError("--residualize-covariates requires --covariates")
        selected = _read_lines(args.residualize_covariates)
        C_all = _load_covariate_subset(args.covariates, all_ids, selected, device)
        C_train = C_all[:n_train]
        C_val = C_all[n_train:]

    # --- Target phenotype (for genetic-correlation mode) ---
    use_gc = args.genetic_correlation is not None
    y2_train = y2_val = None
    if use_gc:
        with h5py.File(args.genetic_correlation, "r") as f:
            tp_ids = np.array(f["ids"])
            tp_raw = np.array(f["data"])
        if tp_raw.ndim == 1:
            tp_raw = tp_raw[:, None]
        row_idx = _reindex(tp_ids, all_ids)
        y2_all = torch.tensor(tp_raw[row_idx].astype(np.float32), device=device)
        y2_train = y2_all[:n_train]
        y2_val = y2_all[n_train:]

    # --- Build heritability callables (mirrors train_hvae.setup_heritability) ---
    def _make_pair(genetics_dict):
        if args.kinship:
            K = genetics_dict["kinship"]
            if K is None:
                raise ValueError("--kinship requires kinship in the genetics HDF5")
            K = K.to(device)
            K_train = K[:n_train, :n_train]
            K_val = K[n_train:, n_train:]
            if use_gc:
                tr_fn = gc(K_train, y2_train, kinship=True, C=C_train, device=device)
                va_fn = gc(K_val, y2_val, kinship=True, C=C_val, device=device)
            else:
                tr_fn = mom(K_train, kinship=True, C=C_train, device=device)
                va_fn = mom(K_val, kinship=True, C=C_val, device=device)
        else:
            G = genetics_dict["genotypes"]
            if G is None:
                raise ValueError("genotypes required (or pass --kinship)")
            G = G.to(device)
            G_train = G[:n_train]
            G_val = G[n_train:]
            if use_gc:
                tr_fn = gc(G_train, y2_train, kinship=False, C=C_train, device=device)
                va_fn = gc(G_val, y2_val, kinship=False, C=C_val, device=device)
            else:
                if C_train is None:
                    raise ValueError("--residualize-covariates is required when using genotypes")
                tr_fn = var_exp(G_train, C_train, device=device)
                va_fn = var_exp(G_val, C_val, device=device)
        return tr_fn, va_fn

    tr_fn, va_fn = _make_pair(g_main)
    tr_fn_odd = va_fn_odd = None
    if args.split_variants:
        tr_fn_odd, va_fn_odd = _make_pair(g_odd)

    # --- Iterate saved epochs ---
    latents_dir = os.path.join(args.outdir, "latents")
    train_files = sorted(glob.glob(os.path.join(latents_dir, "Zm_train.*.txt")))
    if not train_files:
        raise FileNotFoundError(f"No Zm_train.*.txt found in {latents_dir}")

    if args.out is None:
        out_path = _default_out_path(args.outdir, args.genetics, args.genetic_correlation)
    else:
        out_path = args.out

    if out_path == "-":
        out_fh = sys.stdout
        close_at_end = False
    else:
        out_fh = open(out_path, "w")
        close_at_end = True
        print(f"writing rerun log to {out_path}", file=sys.stderr)

    try:
        for train_path in train_files:
            m = _EPOCH_FROM_NAME.search(os.path.basename(train_path))
            if m is None:
                continue
            epoch = int(m.group(1))
            val_path = os.path.join(latents_dir, f"Zm_val.{epoch:05d}.txt")
            if not os.path.exists(val_path):
                print(f"# skipping epoch {epoch}: no {val_path}", file=sys.stderr)
                continue

            Zm_train = torch.tensor(
                np.loadtxt(train_path, delimiter="\t"), dtype=torch.float32, device=device,
            )
            Zm_val = torch.tensor(
                np.loadtxt(val_path, delimiter="\t"), dtype=torch.float32, device=device,
            )

            h_train = _compute_h2(Zm_train, tr_fn)
            h_val = _compute_h2(Zm_val, va_fn)

            if args.split_variants:
                h_train_odd = _compute_h2(Zm_train, tr_fn_odd)
                h_val_odd = _compute_h2(Zm_val, va_fn_odd)
                line = (
                    f"epoch {epoch} - mse_train: 0.0000 - mse_val: 0.0000"
                    f" - h_train_even: {_fmt(h_train)}"
                    f" - h_train_odd: {_fmt(h_train_odd)}"
                    f" - h_val_even: {_fmt(h_val)}"
                    f" - h_val_odd: {_fmt(h_val_odd)}"
                )
            else:
                line = (
                    f"epoch {epoch} - mse_train: 0.0000 - mse_val: 0.0000"
                    f" - h_train: {_fmt(h_train)}"
                    f" - h_val: {_fmt(h_val)}"
                )
            out_fh.write(line + "\n")
            out_fh.flush()
    finally:
        if close_at_end:
            out_fh.close()


if __name__ == "__main__":
    main()
