"""Recompute per-epoch heritability for a finished run against a new genetics file.

Reads saved latents (``<outdir>/latents/Zm_{train,val}.{epoch:05d}.txt``) and
the split IDs (``<outdir>/train_ids.npy``, ``val_ids.npy``), builds heritability
estimators against a new ``--genetics`` input, and writes log lines in the
format consumed by ``out/plot_heritability.py``.

Three execution paths mirror ``train_hvae.setup_heritability``:

  * ``--r2``      → genotype HDF5 + OLS R² (``var_exp``).
                    Mutually exclusive with ``--kinship`` and
                    ``--genetic-correlation``.
  * ``--kinship`` → kinship HDF5 + ``mom()`` (or ``gc()`` with
                    ``--genetic-correlation``).
  * neither       → PLINK ``.bed/.bim/.fam`` + rank-B ``mom()``
                    (or ``gc()`` with ``--genetic-correlation``).
                    Each epoch's display is one cache walk through the
                    BED, vectorised across all zdim columns.

No VAE forward pass — the trained latents are taken as given.

Default output path is ``<outdir>/log.<genetics>[.<target>].txt``, where
``<genetics>`` is the basename of ``--genetics`` (with ``.hdf5`` stripped)
and ``<target>`` is present only when ``--genetic-correlation`` is set.
Pass ``--out -`` to write to stdout instead, or ``--out <path>`` for a
specific file.

Usage:
    python3 out/rerun_heritability.py out/my_run \\
        --genetics data/genetics/plinks/impSNPs_unrel_EUR_array \\
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
from h2vae.plink import BedFile, is_plink_prefix
from h2vae.rank_b_heritability import RankBHeritability
from h2vae.rank_b_gencorr_ldsc import RankBGenCorrLDSC


def _parse_chroms_spec(spec: str | None) -> list[int]:
    """Parse a chromosome list/range spec; ``None`` ⇒ ``1..22``."""
    if spec is None:
        return list(range(1, 23))
    out: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if "-" in token:
            lo, hi = token.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif token:
            out.append(int(token))
    return out


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
    """Per-dimension heritability display values for a single Z.

    Three call shapes are routed transparently:
      * ``RankBHeritability``: ``.display(Z)`` is vectorised over zdim and
        runs a single cache walk for all columns at once.
      * ``mom`` / ``gc`` callables (which expose a ``.display`` attribute
        in gc mode): per-column ``(n, 1)`` invocation.
      * Bare ``var_exp`` callable: per-column ``(n, 1)`` invocation.
    """
    Z = Z.detach()
    if isinstance(her_fn, RankBHeritability):
        return her_fn.display(Z).tolist()
    fn = getattr(her_fn, "display", her_fn)
    return [fn(Z[:, i:i + 1]).item() for i in range(Z.shape[1])]


def _batched_display(
    her_fn,
    Z_batch: torch.Tensor,
    max_cols: int = 1024,
) -> torch.Tensor:
    """Display per-column heritability for a wide ``Z_batch = (n, K)`` in one go.

    For ``RankBHeritability`` we exploit the fact that ``.display(Z)`` runs a
    single cache walk over all K columns at once; we just need to cap K per
    call so the intermediate ``u = (m, K)`` fits in GPU memory.  ``max_cols``
    bounds ``u`` to roughly ``m × max_cols × 4`` bytes (≈2.5 GB at m=613k,
    max_cols=1024).

    For non-rank-B callables, we still iterate per-column (mom/gc/var_exp
    handle one phenotype at a time).
    """
    Z_batch = Z_batch.detach()
    K = Z_batch.shape[1]
    if isinstance(her_fn, RankBHeritability):
        out = torch.empty(K, device=Z_batch.device, dtype=torch.float32)
        for c0 in range(0, K, max_cols):
            c1 = min(c0 + max_cols, K)
            out[c0:c1] = her_fn.display(Z_batch[:, c0:c1]).to(torch.float32)
        return out
    fn = getattr(her_fn, "display", her_fn)
    out = torch.empty(K, device=Z_batch.device, dtype=torch.float32)
    for i in range(K):
        out[i] = fn(Z_batch[:, i:i + 1]).to(torch.float32).view(())
    return out


def _fmt(values: list[float]) -> str:
    return ", ".join(f"{v:.3f}" for v in values)


def _basename_tag(path: str) -> str:
    """Filesystem-safe tag derived from a genetics/phenotype path.

    Strips directory, ``.hdf5`` suffix (so ``data/genetics/height_25.hdf5``
    and the split-variants prefix ``data/genetics/height_25`` both yield
    ``height_25``).
    """
    base = os.path.basename(path.rstrip("/"))
    for suffix in (".sumstats.gz", ".hdf5"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


def _default_out_path(outdir: str, genetics: str, target_phenotype: str | None) -> str:
    parts = [_basename_tag(genetics)]
    if target_phenotype is not None:
        parts.append(_basename_tag(target_phenotype))
    return os.path.join(outdir, "log." + ".".join(parts) + ".txt")


def _build_plink_rank_b(
    *,
    genetics: str,
    all_ids: np.ndarray,
    n_train: int,
    y2_train: torch.Tensor | None,
    y2_val: torch.Tensor | None,
    C_train: torch.Tensor | None,
    C_val: torch.Tensor | None,
    use_gc: bool,
    split_variants: bool,
    device: torch.device,
):
    """Construct RankBHeritability instances for the PLINK rank-B path."""
    if split_variants:
        even_prefix = f"{genetics}.even"
        odd_prefix = f"{genetics}.odd"
        if not is_plink_prefix(even_prefix) or not is_plink_prefix(odd_prefix):
            raise ValueError(
                f"--split-variants on PLINK requires {even_prefix}.bed/.bim/.fam "
                f"and {odd_prefix}.bed/.bim/.fam"
            )
        bed_main = BedFile(even_prefix)
        bed_odd = BedFile(odd_prefix)
    else:
        if not is_plink_prefix(genetics):
            raise ValueError(
                f"--genetics={genetics} is not a PLINK prefix; pass --kinship for a "
                f"kinship HDF5 or --r2 for a genotype HDF5"
            )
        bed_main = BedFile(genetics)
        bed_odd = None

    def _instantiate(bed: BedFile):
        row_idx = _reindex(bed.sample_ids, all_ids)
        row_idx_train = row_idx[:n_train]
        row_idx_val = row_idx[n_train:]
        tr = RankBHeritability(
            bed, row_idx_train,
            C=C_train,
            y_target=y2_train if use_gc else None,
            device=device,
        )
        va = RankBHeritability(
            bed, row_idx_val,
            C=C_val,
            y_target=y2_val if use_gc else None,
            device=device,
        )
        return tr, va

    tr_fn, va_fn = _instantiate(bed_main)
    tr_fn_odd = va_fn_odd = None
    if split_variants:
        tr_fn_odd, va_fn_odd = _instantiate(bed_odd)
    return tr_fn, va_fn, tr_fn_odd, va_fn_odd, bed_main, bed_odd


def _wrap_with_ldsc(
    *,
    tr_fn: RankBHeritability,
    va_fn: RankBHeritability,
    tr_fn_odd: RankBHeritability | None,
    va_fn_odd: RankBHeritability | None,
    bed_main: BedFile,
    bed_odd: BedFile | None,
    sumstats_path: str,
    ref_ld_prefix: str,
    w_ld_prefix: str,
    chroms: list[int],
    intercept_hsq: float | None,
    intercept_gencov: float | None,
):
    """Wrap rank-B estimators with RankBGenCorrLDSC (display gives rg per dim)."""
    from h2vae.ldsc_io import build_ldsc_context

    ctx = build_ldsc_context(
        sumstats_path, ref_ld_prefix, w_ld_prefix,
        bed_variant_ids=bed_main.variant_ids,
        bed_a1=bed_main.a1, bed_a2=bed_main.a2,
        chroms=chroms,
    )
    print(
        f"# [rg-ldsc] aligned {ctx.m_use} SNPs (from {ctx.n_total_input} in "
        f"sumstats, {ctx.n_annot} ref-LD annotations)",
        file=sys.stderr,
    )
    tr_ldsc = RankBGenCorrLDSC(tr_fn, ctx,
                               intercept_hsq=intercept_hsq,
                               intercept_gencov=intercept_gencov)
    va_ldsc = RankBGenCorrLDSC(va_fn, ctx,
                               intercept_hsq=intercept_hsq,
                               intercept_gencov=intercept_gencov)
    tr_ldsc_odd = va_ldsc_odd = None
    if bed_odd is not None:
        ctx_odd = build_ldsc_context(
            sumstats_path, ref_ld_prefix, w_ld_prefix,
            bed_variant_ids=bed_odd.variant_ids,
            bed_a1=bed_odd.a1, bed_a2=bed_odd.a2,
            chroms=chroms,
        )
        print(f"# [rg-ldsc] odd: aligned {ctx_odd.m_use} SNPs", file=sys.stderr)
        tr_ldsc_odd = RankBGenCorrLDSC(tr_fn_odd, ctx_odd,
                                        intercept_hsq=intercept_hsq,
                                        intercept_gencov=intercept_gencov)
        va_ldsc_odd = RankBGenCorrLDSC(va_fn_odd, ctx_odd,
                                        intercept_hsq=intercept_hsq,
                                        intercept_gencov=intercept_gencov)
    return tr_ldsc, va_ldsc, tr_ldsc_odd, va_ldsc_odd


def _build_hdf5_estimators(
    *,
    kinship: bool,
    use_gc: bool,
    split_variants: bool,
    genetics: str,
    all_ids: np.ndarray,
    n_train: int,
    y2_train: torch.Tensor | None,
    y2_val: torch.Tensor | None,
    C_train: torch.Tensor | None,
    C_val: torch.Tensor | None,
    device: torch.device,
):
    """Construct fixed-callable estimators for kinship/r2 HDF5 inputs."""
    if split_variants:
        g_main = load_genetics_reindexed(f"{genetics}.even.hdf5", all_ids)
        g_odd = load_genetics_reindexed(f"{genetics}.odd.hdf5", all_ids)
    else:
        g_main = load_genetics_reindexed(genetics, all_ids)
        g_odd = None

    def _make_pair(g):
        if kinship:
            K = g["kinship"]
            if K is None:
                raise ValueError("--kinship requires kinship in the genetics HDF5")
            K = K.to(device)
            K_train = K[:n_train, :n_train]
            K_val = K[n_train:, n_train:]
            if use_gc:
                return (gc(K_train, y2_train, kinship=True, C=C_train, device=device),
                        gc(K_val,   y2_val,   kinship=True, C=C_val,   device=device))
            return (mom(K_train, kinship=True, C=C_train, device=device),
                    mom(K_val,   kinship=True, C=C_val,   device=device))
        G = g["genotypes"]
        if G is None:
            raise ValueError("--r2 requires genotypes in the genetics HDF5")
        G = G.to(device)
        G_train = G[:n_train]
        G_val = G[n_train:]
        if use_gc:
            return (gc(G_train, y2_train, kinship=False, C=C_train, device=device),
                    gc(G_val,   y2_val,   kinship=False, C=C_val,   device=device))
        if C_train is None:
            raise ValueError("--residualize-covariates is required with --r2")
        return (var_exp(G_train, C_train, device=device),
                var_exp(G_val,   C_val,   device=device))

    tr_fn, va_fn = _make_pair(g_main)
    tr_fn_odd = va_fn_odd = None
    if split_variants:
        tr_fn_odd, va_fn_odd = _make_pair(g_odd)
    return tr_fn, va_fn, tr_fn_odd, va_fn_odd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", type=str, nargs="+",
                    help="one or more training output directories (each has latents/, "
                         "train_ids.npy, val_ids.npy). When multiple are given, they must "
                         "share identical train/val IDs so the heritability estimators can "
                         "be built once and reused across all of them.")
    ap.add_argument("--genetics", type=str, required=True,
                    help="genetics file: PLINK prefix (default), kinship HDF5 (--kinship), or "
                         "genotype HDF5 (--r2). Prefix when --split-variants is set.")
    ap.add_argument("--kinship", action="store_true", default=False,
                    help="--genetics is a kinship HDF5; use mom (or gc with --genetic-correlation)")
    ap.add_argument("--r2", action="store_true", default=False,
                    help="--genetics is a genotype HDF5; use OLS R² (var_exp). "
                         "Mutually exclusive with --kinship and --genetic-correlation.")
    ap.add_argument("--split-variants", action="store_true", default=False,
                    help="--genetics is a prefix; load {prefix}.even and {prefix}.odd "
                         "(PLINK if neither --kinship nor --r2 is set, else HDF5)")
    ap.add_argument("--covariates", type=str, default=None,
                    help="covariates HDF5 file (needed for --residualize-covariates)")
    ap.add_argument("--residualize-covariates", type=str, default=None,
                    help="text file of covariate names to project out before heritability estimation")
    ap.add_argument("--genetic-correlation", type=str, default=None,
                    help="HDF5 (keys 'data', 'ids') of a target phenotype; switches to per-latent "
                         "SCORE-OVERLAP genetic correlation with this phenotype")
    ap.add_argument("--rg-ldsc-sumstats", type=str, default=None,
                    help="LDSC-munged sumstats.gz for an external trait; enables per-latent LDSC "
                         "genetic correlation (rg + gencov). Requires the PLINK rank-B path "
                         "(no --r2, no --kinship). Mutually exclusive with --genetic-correlation.")
    ap.add_argument("--rg-ldsc-ref-ld-chr", type=str, default=None,
                    help="LDSC ref-LD per-chrom prefix; required with --rg-ldsc-sumstats")
    ap.add_argument("--rg-ldsc-w-ld-chr", type=str, default=None,
                    help="LDSC regression-weight LD per-chrom prefix; required with --rg-ldsc-sumstats")
    ap.add_argument("--rg-ldsc-intercept-hsq", type=float, default=None,
                    help="fix both hsq intercepts (cohort and external); default leaves them free")
    ap.add_argument("--rg-ldsc-intercept-gencov", type=float, default=None,
                    help="fix the gencov intercept (sample overlap); default leaves it free")
    ap.add_argument("--rg-ldsc-chroms", type=str, default=None,
                    help="chromosome spec (e.g. '1-22' or '1,3,5'); default 1..22")
    ap.add_argument("--which-cuda", type=int, default=0)
    ap.add_argument("--out", type=str, default=None,
                    help="output path (default: <outdir>/log.<genetics>[.<target>].txt; "
                         "pass '-' for stdout)")
    args = ap.parse_args()

    if args.r2 and args.kinship:
        raise ValueError("--r2 and --kinship are mutually exclusive")
    if args.r2 and args.genetic_correlation is not None:
        raise ValueError("--r2 and --genetic-correlation are mutually exclusive")
    use_ldsc = args.rg_ldsc_sumstats is not None
    if use_ldsc:
        if args.r2 or args.kinship:
            raise ValueError("--rg-ldsc-sumstats requires the PLINK rank-B path (no --r2/--kinship)")
        if args.genetic_correlation is not None:
            raise ValueError("--rg-ldsc-sumstats and --genetic-correlation are mutually exclusive")
        if args.rg_ldsc_ref_ld_chr is None or args.rg_ldsc_w_ld_chr is None:
            raise ValueError("--rg-ldsc-sumstats requires --rg-ldsc-ref-ld-chr and --rg-ldsc-w-ld-chr")

    device = torch.device(f"cuda:{args.which_cuda}" if torch.cuda.is_available() else "cpu")

    # --- Split IDs (as written by the original training run) ---
    primary = args.outdir[0]
    train_ids_orig = np.load(os.path.join(primary, "train_ids.npy"))
    val_ids_orig = np.load(os.path.join(primary, "val_ids.npy"))
    for extra in args.outdir[1:]:
        t = np.load(os.path.join(extra, "train_ids.npy"))
        v = np.load(os.path.join(extra, "val_ids.npy"))
        if not (np.array_equal(t, train_ids_orig) and np.array_equal(v, val_ids_orig)):
            raise ValueError(
                f"{extra} has different train/val IDs than {primary}; "
                "group runs by cohort before passing to a single invocation"
            )

    # --- Target phenotype + NaN filter (matches load_data convention) ---
    # When --genetic-correlation is set, drop any sample whose target value is
    # NaN, mirroring h2vae.data.load_data: silently dropping NaN targets is the
    # only sensible behaviour (gc() would otherwise propagate NaN through
    # tr(K̃²), tr(K̃) and the d2 floor → all reported h values become NaN).
    use_gc = args.genetic_correlation is not None
    train_keep = np.ones(len(train_ids_orig), dtype=bool)
    val_keep = np.ones(len(val_ids_orig), dtype=bool)
    y2_train = y2_val = None
    train_y2_raw = val_y2_raw = None
    if use_gc:
        with h5py.File(args.genetic_correlation, "r") as f:
            tp_ids = np.array(f["ids"])
            tp_raw = np.array(f["data"])
        if tp_raw.ndim == 1:
            tp_raw = tp_raw[:, None]
        id_to_tp_row = {v: i for i, v in enumerate(tp_ids)}
        try:
            tr_rows = np.array([id_to_tp_row[i] for i in train_ids_orig])
            va_rows = np.array([id_to_tp_row[i] for i in val_ids_orig])
        except KeyError as e:
            raise ValueError(
                f"target phenotype HDF5 missing id {e.args[0]} from the run's split"
            )
        train_y2_raw = tp_raw[tr_rows]
        val_y2_raw = tp_raw[va_rows]
        train_keep = ~np.isnan(train_y2_raw).any(axis=1)
        val_keep = ~np.isnan(val_y2_raw).any(axis=1)
        n_drop_tr = int((~train_keep).sum()); n_drop_va = int((~val_keep).sum())
        if n_drop_tr or n_drop_va:
            print(
                f"# dropped {n_drop_tr} train + {n_drop_va} val samples with NaN "
                f"target phenotype (kept {int(train_keep.sum())}/{len(train_ids_orig)} "
                f"train, {int(val_keep.sum())}/{len(val_ids_orig)} val)",
                file=sys.stderr,
            )

    train_ids = train_ids_orig[train_keep]
    val_ids = val_ids_orig[val_keep]
    all_ids = np.concatenate([train_ids, val_ids])
    n_train = len(train_ids)

    if use_gc:
        y2_train = torch.tensor(
            train_y2_raw[train_keep].astype(np.float32), device=device,
        )
        y2_val = torch.tensor(
            val_y2_raw[val_keep].astype(np.float32), device=device,
        )

    # --- Covariates (for residualization, post-filter) ---
    C_train = C_val = None
    if args.residualize_covariates is not None:
        if args.covariates is None:
            raise ValueError("--residualize-covariates requires --covariates")
        selected = _read_lines(args.residualize_covariates)
        C_all = _load_covariate_subset(args.covariates, all_ids, selected, device)
        C_train = C_all[:n_train]
        C_val = C_all[n_train:]

    # --- Build heritability estimators (three-way dispatch) ---
    if not args.r2 and not args.kinship:
        (tr_fn, va_fn, tr_fn_odd, va_fn_odd,
         bed_main, bed_odd) = _build_plink_rank_b(
            genetics=args.genetics, all_ids=all_ids, n_train=n_train,
            y2_train=y2_train, y2_val=y2_val,
            C_train=C_train, C_val=C_val,
            use_gc=use_gc, split_variants=args.split_variants, device=device,
        )
        if use_ldsc:
            tr_fn, va_fn, tr_fn_odd, va_fn_odd = _wrap_with_ldsc(
                tr_fn=tr_fn, va_fn=va_fn,
                tr_fn_odd=tr_fn_odd, va_fn_odd=va_fn_odd,
                bed_main=bed_main, bed_odd=bed_odd,
                sumstats_path=args.rg_ldsc_sumstats,
                ref_ld_prefix=args.rg_ldsc_ref_ld_chr,
                w_ld_prefix=args.rg_ldsc_w_ld_chr,
                chroms=_parse_chroms_spec(args.rg_ldsc_chroms),
                intercept_hsq=args.rg_ldsc_intercept_hsq,
                intercept_gencov=args.rg_ldsc_intercept_gencov,
            )
    else:
        tr_fn, va_fn, tr_fn_odd, va_fn_odd = _build_hdf5_estimators(
            kinship=args.kinship, use_gc=use_gc,
            split_variants=args.split_variants,
            genetics=args.genetics, all_ids=all_ids, n_train=n_train,
            y2_train=y2_train, y2_val=y2_val,
            C_train=C_train, C_val=C_val,
            device=device,
        )

    # --- Batched display across all (outdir, epoch) pairs ----------------
    # The dominant cost of .display(Z) is one cache walk over the BED-derived
    # in-memory cache, mostly independent of Z's column count.  By column-
    # stacking all (outdir, epoch) latents into a single (n, K) tensor and
    # chunking only as needed to bound the (m, max_cols) intermediate, we
    # replace n_runs × n_epochs cache walks with ⌈K/max_cols⌉ walks per split.
    #
    # LDSC mode skips the batching: RankBGenCorrLDSC.display() runs a per-dim
    # IRWLS loop that dominates cost, and we need per-epoch last_gencov +
    # last_intercepts diagnostics that the flat-chunk path would discard.
    if args.out is not None and len(args.outdir) > 1:
        raise ValueError("--out with multiple outdirs is ambiguous; omit --out to write "
                         "the default path inside each outdir, or invoke per-outdir.")
    if use_ldsc and args.split_variants:
        raise ValueError("--rg-ldsc-sumstats with --split-variants is not yet supported by "
                         "rerun_heritability.py; rerun the even/odd prefixes separately")

    # Gather per-outdir epoch lists (each outdir's saved epochs are the
    # intersection of available Zm_train.* and Zm_val.* files).
    per_outdir_epochs: list[list[int]] = []
    for outdir in args.outdir:
        latents_dir = os.path.join(outdir, "latents")
        train_files = sorted(glob.glob(os.path.join(latents_dir, "Zm_train.*.txt")))
        if not train_files:
            raise FileNotFoundError(f"No Zm_train.*.txt found in {latents_dir}")
        epochs: list[int] = []
        for train_path in train_files:
            m = _EPOCH_FROM_NAME.search(os.path.basename(train_path))
            if m is None:
                continue
            epoch = int(m.group(1))
            val_path = os.path.join(latents_dir, f"Zm_val.{epoch:05d}.txt")
            if not os.path.exists(val_path):
                print(f"# skipping {outdir} epoch {epoch}: no {val_path}", file=sys.stderr)
                continue
            epochs.append(epoch)
        if not epochs:
            raise FileNotFoundError(f"No paired (Zm_train, Zm_val) files in {outdir}/latents")
        per_outdir_epochs.append(epochs)

    print(
        f"# batched display: {len(args.outdir)} outdir(s), "
        f"{sum(len(e) for e in per_outdir_epochs)} (outdir, epoch) pairs",
        file=sys.stderr,
    )

    # --- LDSC dispatch: per-epoch loop (rg + gencov + intercepts) --------
    if use_ldsc:
        def _intercept_summary(int_vec) -> tuple[float, float]:
            arr = np.asarray(int_vec, dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                return float("nan"), float("nan")
            return float(arr.mean()), float(arr.std())

        for outdir, epochs in zip(args.outdir, per_outdir_epochs):
            if args.out is None:
                out_path = _default_out_path(outdir, args.genetics, args.rg_ldsc_sumstats)
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
                latents_dir = os.path.join(outdir, "latents")
                for epoch in epochs:
                    Zt = np.loadtxt(os.path.join(latents_dir, f"Zm_train.{epoch:05d}.txt"),
                                    delimiter="\t")
                    Zv = np.loadtxt(os.path.join(latents_dir, f"Zm_val.{epoch:05d}.txt"),
                                    delimiter="\t")
                    Zt_t = torch.from_numpy(Zt[train_keep].astype(np.float32)).to(device)
                    Zv_t = torch.from_numpy(Zv[val_keep].astype(np.float32)).to(device)

                    rg_train = tr_fn.display(Zt_t).tolist()
                    gc_train = tr_fn.last_gencov.tolist()
                    rg_val = va_fn.display(Zv_t).tolist()
                    gc_val = va_fn.last_gencov.tolist()
                    skipped = list(va_fn.last_skipped)
                    h1_m, h1_s = _intercept_summary(va_fn.last_intercepts["hsq1"])
                    h2_m, h2_s = _intercept_summary(va_fn.last_intercepts["hsq2"])
                    gc_m, gc_s = _intercept_summary(va_fn.last_intercepts["gencov"])
                    n_dim = len(rg_val)

                    out_fh.write(
                        f"epoch {epoch} - mse_train: 0.0000 - mse_val: 0.0000"
                        f" - h_train: {_fmt(rg_train)}"
                        f" - h_val: {_fmt(rg_val)}\n"
                    )
                    out_fh.write(
                        f"epoch {epoch}"
                        f" - gencov_train: {', '.join(f'{v:+.4e}' for v in gc_train)}"
                        f" - gencov_val: {', '.join(f'{v:+.4e}' for v in gc_val)}\n"
                    )
                    out_fh.write(
                        f"rg_ldsc_intercepts_val (epoch {epoch}): "
                        f"hsq1={h1_m:.3f}±{h1_s:.3f}  hsq2={h2_m:.3f}±{h2_s:.3f}  "
                        f"gencov={gc_m:.3f}±{gc_s:.3f}  skipped={len(skipped)}/{n_dim}\n"
                    )
                    out_fh.flush()
            finally:
                if close_at_end:
                    out_fh.close()
        return

    # Load all Z for all (outdir, epoch) into two big GPU tensors (train, val).
    # Per-outdir entries are placed contiguously along the column axis so we
    # can slice the flat result back to per-(outdir, epoch) chunks.
    def _load_split(split: str, keep: np.ndarray) -> tuple[torch.Tensor, int]:
        cols: list[torch.Tensor] = []
        for outdir, epochs in zip(args.outdir, per_outdir_epochs):
            latents_dir = os.path.join(outdir, "latents")
            for epoch in epochs:
                path = os.path.join(latents_dir, f"Zm_{split}.{epoch:05d}.txt")
                Z_full = np.loadtxt(path, delimiter="\t")
                if Z_full.shape[0] != keep.shape[0]:
                    raise ValueError(
                        f"{path} has {Z_full.shape[0]} rows but split has "
                        f"{keep.shape[0]}; cannot align"
                    )
                cols.append(torch.from_numpy(Z_full[keep].astype(np.float32)))
        Z_big = torch.cat(cols, dim=1).to(device)
        zdim = cols[0].shape[1]
        return Z_big, zdim

    Z_train_big, zdim = _load_split("train", train_keep)
    Z_val_big,   _    = _load_split("val", val_keep)

    print(
        f"# Z_train_big shape={tuple(Z_train_big.shape)}, "
        f"Z_val_big shape={tuple(Z_val_big.shape)}, zdim={zdim}",
        file=sys.stderr,
    )

    # One batched display per split (chunked internally) — and again for the
    # odd-variant estimators when --split-variants.
    h_train_flat = _batched_display(tr_fn, Z_train_big)
    h_val_flat   = _batched_display(va_fn, Z_val_big)
    h_train_odd_flat = h_val_odd_flat = None
    if args.split_variants:
        h_train_odd_flat = _batched_display(tr_fn_odd, Z_train_big)
        h_val_odd_flat   = _batched_display(va_fn_odd, Z_val_big)

    # Walk back through (outdir, epoch) order to slice and write logs.
    cursor = 0
    for outdir, epochs in zip(args.outdir, per_outdir_epochs):
        if args.out is None:
            out_path = _default_out_path(outdir, args.genetics, args.genetic_correlation)
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
            for epoch in epochs:
                lo, hi = cursor, cursor + zdim
                cursor = hi
                h_train = h_train_flat[lo:hi].tolist()
                h_val   = h_val_flat[lo:hi].tolist()

                if args.split_variants:
                    h_train_odd = h_train_odd_flat[lo:hi].tolist()
                    h_val_odd   = h_val_odd_flat[lo:hi].tolist()
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
        finally:
            if close_at_end:
                out_fh.close()


if __name__ == "__main__":
    main()
