"""Train a VAE with genetics-regularized latent space (HVAE).

The training loop uses a two-stage epoch structure:
  1. Encode the full dataset in eval mode to collect latent statistics.
  2. Backprop through mini-batches with a composite loss that combines
     reconstruction (MSE + beta*KLD), heritability, latent decorrelation,
     and moment-matching penalties.

Input data is split across three HDF5 files (images, genetics, covariates),
merged on sample IDs at load time.

Covariate control has two independent pathways:
  --decode-covariates <file>       covariates concatenated to z before decoding
  --residualize-covariates <file>  covariates projected out before heritability estimation
Both are optional and can be active simultaneously with different covariate sets.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import Tensor, optim
from torch.utils.data import DataLoader

from h2vae.models import get_model_class
from h2vae.models.base import BaseVAE
from h2vae.heritability import mom, var_exp, gc
from h2vae.rank_b_heritability import RankBHeritability
from h2vae.rank_b_spectrum import RankBHeritabilitySpectrum
from h2vae.data import (
    ImageDataset, ImageFileDataset, load_data, load_genetics_reindexed,
    make_streaming_dataset,
)
from h2vae.latent_utils import center_and_scale, corrcoef


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LossWeights:
    mse: float
    h: float
    corr: float
    sk: float


@dataclass
class HVAEConfig:
    # Input files
    images: str = ""
    genetics: str = ""
    covariates: str | None = None
    outdir: str = "../out/hvae"
    resume: str | None = None

    # Model
    model: str = "vae2d"
    filts: int = 32
    zdim: int = 64
    beta: float = 1.0
    img_size: int | None = None
    colors: int | None = None
    steps: int | None = None
    gradient_checkpointing: bool = False

    # Training
    train_frac: float = 0.8
    seed: int = 0
    vae_lr: float = 1e-4
    clip: float = 1.0
    bs: int = 64
    epochs: int = 1001
    epoch_cb: int = 10

    # Loss weights
    mse_weight: float = 1.0
    h_weight: float = 1.0
    corr_weight: float = 0.0
    sk_weight: float = 0.0

    # Heritability
    hweights: str | None = None  # text file: one weight per line, one per latent dim
    kinship: bool = False
    r2: bool = False  # --genetics is a genotype HDF5; use OLS R² (var_exp)
    split_variants: bool = False
    genetic_correlation: str | None = None  # HDF5 with target phenotype (switches loss to gc)

    # External genetic-correlation via differentiable LDSC against an out-of-cohort
    # trait (rank-B PLINK path only). Adds a per-step rg loss vs the munged sumstats;
    # --hweights selects which dims are pressured. See h2vae.rank_b_gencorr_ldsc.
    rg_ldsc_sumstats: str | None = None
    rg_ldsc_ref_ld_chr: str | None = None
    rg_ldsc_w_ld_chr: str | None = None
    rg_ldsc_intercept_hsq: float | None = None
    rg_ldsc_intercept_gencov: float | None = None
    rg_ldsc_chroms: str | None = None

    # Heritability-spectrum objective (linearly-accessible heritability; PLINK path only).
    # Maximizes the generalized-eigenvalue spectrum of G v = λ P v instead of per-dim h².
    # With --linear-heritability, --hweights is reinterpreted as per-RANK spectrum weights.
    linear_heritability: bool = False
    spectrum_ridge: float = 1e-4
    spectrum_clamp: bool = False  # relu(λ) before the weighted sum (differentiable nearest-PSD)
    # Relative weights of the two heritability sub-objectives under --linear-heritability:
    # total her loss = spectrum_weight*spectrum_loss + marginal_weight*(−Σ_d h²_d).
    # marginal_weight>0 also grows the per-dim (single-latent) heritabilities.
    spectrum_weight: float = 1.0
    marginal_weight: float = 0.0
    # Restrict the spectrum objective to the first K latent dims (G[:K,:K], P[:K,:K]),
    # leaving dims K..zdim-1 free for reconstruction. 0 = full zdim (no restriction).
    # Anti-overfitting knob for large latent sets, distinct from per-rank --hweights.
    spectrum_dims: int = 0
    # Optional encoder posterior-std floor: zs = softplus(...).clamp_min(zs_floor).
    # 0.0 (default) = legacy (no floor). Set e.g. 1e-8 to stop zs underflowing to 0,
    # which causes log(zs)=-inf in the KL and a NaN that surfaces as the spectrum
    # Cholesky failure at low beta. General-purpose; applies to all model variants.
    zs_floor: float = 0.0

    # Covariates (independent pathways, activated by file presence)
    decode_covariates: str | None = None
    residualize_covariates: str | None = None

    # Device
    which_cuda: int = 0
    debug: bool = False

    @property
    def vae_cfg(self) -> dict:
        cfg = {"nf": self.filts, "zdim": self.zdim, "beta": self.beta}
        if self.img_size is not None:
            cfg["img_size"] = self.img_size
        if self.colors is not None:
            cfg["colors"] = self.colors
        if self.steps is not None:
            cfg["steps"] = self.steps
        if self.gradient_checkpointing:
            cfg["gradient_checkpointing"] = True
        if self.zs_floor > 0:
            cfg["zs_floor"] = self.zs_floor
        return cfg

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.which_cuda}")
        return torch.device("cpu")

    @property
    def loss_weights(self) -> LossWeights:
        return LossWeights(
            mse=self.mse_weight,
            h=self.h_weight,
            corr=self.corr_weight,
            sk=self.sk_weight,
        )


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_args() -> HVAEConfig:
    parser = argparse.ArgumentParser(description="Train HVAE")

    # Input files
    parser.add_argument("--images", type=str, required=True, help="images HDF5 file")
    parser.add_argument("--genetics", type=str, required=True, help="genetics HDF5 file")
    parser.add_argument("--covariates", type=str, default=None, help="covariates HDF5 file")
    parser.add_argument("--outdir", type=str, default=HVAEConfig.outdir, help="output directory")
    parser.add_argument("--resume", type=str, default=None, help="output directory to resume from (loads latest checkpoint)")

    # Model
    parser.add_argument("--model", type=str, default=HVAEConfig.model, help="model architecture (vae2d, vae3d, vae1d)")
    parser.add_argument("--filts", type=int, default=HVAEConfig.filts, help="number of conv filters")
    parser.add_argument("--zdim", type=int, default=HVAEConfig.zdim, help="latent dimension")
    parser.add_argument("--beta", type=float, default=HVAEConfig.beta, help="KL divergence weight (beta-VAE)")
    parser.add_argument("--img-size", type=int, default=None, help="input spatial size (overrides model default)")
    parser.add_argument("--colors", type=int, default=None, help="number of input channels")
    parser.add_argument("--steps", type=int, default=None, help="number of down/up sampling stages")
    parser.add_argument("--gradient-checkpoint", action="store_true", default=False, help="enable gradient checkpointing (trades compute for memory)")

    # Training
    parser.add_argument("--train-frac", type=float, default=HVAEConfig.train_frac, help="fraction of samples for training (rest go to val)")
    parser.add_argument("--seed", type=int, default=HVAEConfig.seed, help="random seed")
    parser.add_argument("--vae-lr", type=float, default=HVAEConfig.vae_lr, help="VAE learning rate")
    parser.add_argument("--clip", type=float, default=HVAEConfig.clip, help="gradient clipping")
    parser.add_argument("--bs", type=int, default=HVAEConfig.bs, help="batch size")
    parser.add_argument("--epoch-cb", type=int, default=HVAEConfig.epoch_cb, help="checkpoint interval")
    parser.add_argument("--epochs", type=int, default=HVAEConfig.epochs, help="total epochs")

    # Loss weights
    parser.add_argument("--h-weight", type=float, default=HVAEConfig.h_weight, help="heritability loss weight")
    parser.add_argument("--mse-weight", type=float, default=HVAEConfig.mse_weight, help="reconstruction loss weight")
    parser.add_argument("--corr-weight", type=float, default=HVAEConfig.corr_weight, help="latent correlation penalty weight")
    parser.add_argument("--sk-weight", type=float, default=HVAEConfig.sk_weight, help="skew/kurtosis penalty weight")

    # Heritability
    parser.add_argument("--hweights", type=str, default=None, help="text file with per-latent heritability weights (one per line)")
    parser.add_argument("--kinship", action="store_true", default=False, help="--genetics is a precomputed-kinship HDF5; use direct mom/gc")
    parser.add_argument("--r2", action="store_true", default=False, help="--genetics is a genotype HDF5; use OLS R² (var_exp). Mutually exclusive with --kinship and --genetic-correlation")
    parser.add_argument("--split-variants", action="store_true", default=False, help="--genetics is a prefix; load {prefix}.even.* and {prefix}.odd.* (HDF5 or PLINK)")
    parser.add_argument("--genetic-correlation", type=str, default=None,
                        help="HDF5 (keys 'data', 'ids') of a target phenotype; switches the loss "
                             "to per-latent SCORE-OVERLAP genetic correlation with this phenotype")
    parser.add_argument("--rg-ldsc-sumstats", type=str, default=None,
                        help="munged .sumstats.gz of an external trait; adds a differentiable "
                             "LDSC genetic-covariance loss against it (displayed as rg). Requires "
                             "the PLINK rank-B path (incompatible with --kinship, --r2, "
                             "--genetic-correlation)")
    parser.add_argument("--rg-ldsc-ref-ld-chr", type=str, default=None,
                        help="per-chrom reference LD-score prefix (LDSC --ref-ld-chr)")
    parser.add_argument("--rg-ldsc-w-ld-chr", type=str, default=None,
                        help="per-chrom regression-weight LD-score prefix (LDSC --w-ld-chr)")
    parser.add_argument("--rg-ldsc-intercept-hsq", type=float, default=None,
                        help="fix both h2 intercepts to this value (default: free, absorbs "
                             "residual stratification)")
    parser.add_argument("--rg-ldsc-intercept-gencov", type=float, default=None,
                        help="fix the gencov intercept to this value (default: free, absorbs "
                             "sample overlap)")
    parser.add_argument("--rg-ldsc-chroms", type=str, default=None,
                        help="restrict rg-ldsc to these chroms: e.g. '1', '1-22', '1,2,3' "
                             "(default: all of 1-22)")
    parser.add_argument("--linear-heritability", action="store_true", default=False,
                        help="maximize the heritability SPECTRUM (tr(P⁻¹G)) instead of per-dim h²; "
                             "PLINK path only. --hweights becomes per-rank spectrum weights")
    parser.add_argument("--spectrum-ridge", type=float, default=HVAEConfig.spectrum_ridge,
                        help="ridge added to the phenotypic correlation P before whitening")
    parser.add_argument("--spectrum-clamp", action="store_true", default=False,
                        help="clamp the spectrum at 0 (relu) before the weighted sum")
    parser.add_argument("--spectrum-weight", type=float, default=HVAEConfig.spectrum_weight,
                        help="coefficient on the spectrum loss (with --linear-heritability)")
    parser.add_argument("--marginal-weight", type=float, default=HVAEConfig.marginal_weight,
                        help="coefficient on the marginal per-dim heritability loss "
                             "(−Σ_d h²_d); >0 also grows single-latent heritabilities")
    parser.add_argument("--spectrum-dims", type=int, default=HVAEConfig.spectrum_dims,
                        help="restrict the spectrum objective to the first K latent dims "
                             "(G[:K,:K], P[:K,:K]); 0=full zdim. With --linear-heritability, "
                             "--hweights must then be at least length K (sliced to [:K])")
    parser.add_argument("--zs-floor", type=float, default=HVAEConfig.zs_floor,
                        help="floor for the encoder posterior std: softplus(...).clamp_min(zs_floor). "
                             "0=off (legacy). e.g. 1e-8 stops zs underflowing to 0 -> log(zs) NaN "
                             "(the spectrum-loss crash at low beta).")

    # Covariates
    parser.add_argument("--decode-covariates", type=str, default=None, help="text file of covariate names for decode conditioning")
    parser.add_argument("--residualize-covariates", type=str, default=None, help="text file of covariate names for heritability residualization")

    # Device
    parser.add_argument("--which-cuda", type=int, default=HVAEConfig.which_cuda, help="CUDA device index")
    parser.add_argument("--debug", action="store_true", default=False, help="drop into debugger at start")

    args = parser.parse_args()
    return HVAEConfig(
        images=args.images,
        genetics=args.genetics,
        covariates=args.covariates,
        outdir=args.outdir,
        resume=args.resume,
        model=args.model,
        filts=args.filts,
        zdim=args.zdim,
        beta=args.beta,
        img_size=args.img_size,
        colors=args.colors,
        steps=args.steps,
        gradient_checkpointing=args.gradient_checkpoint,
        train_frac=args.train_frac,
        seed=args.seed,
        vae_lr=args.vae_lr,
        clip=args.clip,
        bs=args.bs,
        epoch_cb=args.epoch_cb,
        epochs=args.epochs,
        mse_weight=args.mse_weight,
        h_weight=args.h_weight,
        corr_weight=args.corr_weight,
        sk_weight=args.sk_weight,
        hweights=args.hweights,
        kinship=args.kinship,
        r2=args.r2,
        split_variants=args.split_variants,
        genetic_correlation=args.genetic_correlation,
        rg_ldsc_sumstats=args.rg_ldsc_sumstats,
        rg_ldsc_ref_ld_chr=args.rg_ldsc_ref_ld_chr,
        rg_ldsc_w_ld_chr=args.rg_ldsc_w_ld_chr,
        rg_ldsc_intercept_hsq=args.rg_ldsc_intercept_hsq,
        rg_ldsc_intercept_gencov=args.rg_ldsc_intercept_gencov,
        rg_ldsc_chroms=args.rg_ldsc_chroms,
        linear_heritability=args.linear_heritability,
        spectrum_ridge=args.spectrum_ridge,
        spectrum_clamp=args.spectrum_clamp,
        spectrum_weight=args.spectrum_weight,
        marginal_weight=args.marginal_weight,
        spectrum_dims=args.spectrum_dims,
        zs_floor=args.zs_floor,
        decode_covariates=args.decode_covariates,
        residualize_covariates=args.residualize_covariates,
        which_cuda=args.which_cuda,
        debug=args.debug,
    )


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def setup_output_dirs(outdir: str) -> tuple[str, str, str]:
    """Create output, weights, plots, and latents directories. Returns (wdir, fdir, ldir)."""
    wdir = os.path.join(outdir, "weights")
    fdir = os.path.join(outdir, "plots")
    ldir = os.path.join(outdir, "latents")
    for d in [outdir, wdir, fdir, ldir]:
        os.makedirs(d, exist_ok=True)
    return wdir, fdir, ldir


def setup_logging(outdir: str) -> None:
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )
    fh = logging.FileHandler(os.path.join(outdir, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def find_latest_checkpoint(outdir: str) -> tuple[str | None, int]:
    """Find the latest weights file in outdir/weights/.

    Returns:
        path: Path to the latest checkpoint, or None if none found.
        epoch: Epoch number of the latest checkpoint, or 0.
    """
    wdir = os.path.join(outdir, "weights")
    if not os.path.isdir(wdir):
        return None, 0
    weight_files = [f for f in os.listdir(wdir) if f.startswith("weights.") and f.endswith(".pt")]
    if not weight_files:
        return None, 0
    epochs = [int(f.split(".")[1]) for f in weight_files]
    latest = max(epochs)
    return os.path.join(wdir, f"weights.{latest:05d}.pt"), latest


# ---------------------------------------------------------------------------
# Text file readers
# ---------------------------------------------------------------------------

def read_lines(path: str) -> list[str]:
    """Read non-empty stripped lines from a text file."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def read_weights(path: str) -> Tensor:
    """Read per-latent heritability weights from a text file (one float per line)."""
    values = [float(x) for x in read_lines(path)]
    return torch.tensor(values, dtype=torch.float32)


def select_covariate_columns(
    covariates: Tensor,
    all_names: list[str],
    selected_names: list[str],
) -> Tensor:
    """Select columns from a covariates matrix by name."""
    name_to_idx = {name: i for i, name in enumerate(all_names)}
    indices = []
    for name in selected_names:
        if name not in name_to_idx:
            raise ValueError(f"Covariate '{name}' not found. Available: {all_names}")
        indices.append(name_to_idx[name])
    return covariates[:, indices]


# ---------------------------------------------------------------------------
# Heritability and covariate setup
# ---------------------------------------------------------------------------

@dataclass
class CovariateState:
    """Holds per-split covariate tensors for the decode pathway."""
    decode_train: Tensor | None = None
    decode_val: Tensor | None = None


@dataclass
class HeritabilityState:
    """Holds heritability loss callables / rank-B modules and related state.

    Three execution paths share this struct:

    * ``--kinship``  → ``loss_fn`` / ``val_fn`` are kinship-mode mom/gc.
    * ``--r2``       → ``loss_fn`` / ``val_fn`` are genotype-mode
      ``var_exp()`` (or ``gc()`` if combined with --genetic-correlation,
      though that combo is currently rejected at parse time).
    * neither flag (PLINK)  → ``rank_b`` / ``rank_b_val`` are the
      rank-B modules, and the callable slots stay ``None``.
    """
    # Backprop loss (even-chromosome when split, single otherwise)
    loss_fn: Callable | None = None
    val_fn: Callable | None = None

    # Display-only odd-chromosome estimators (None when not splitting)
    loss_fn_odd: Callable | None = None
    val_fn_odd: Callable | None = None

    # Rank-B modules (PLINK path). RankBHeritabilitySpectrum is a subclass, so
    # the runtime type is compatible; the annotation is widened for clarity.
    rank_b: RankBHeritability | RankBHeritabilitySpectrum | None = None
    rank_b_val: RankBHeritability | RankBHeritabilitySpectrum | None = None
    rank_b_odd: RankBHeritability | RankBHeritabilitySpectrum | None = None
    rank_b_val_odd: RankBHeritability | RankBHeritabilitySpectrum | None = None

    cov_state: CovariateState = None  # type: ignore[assignment]
    hweights: Tensor | None = None


def _parse_chroms_spec(spec: str | None) -> list[int]:
    """Parse a ``--rg-ldsc-chroms`` spec into a list of chromosome ints.

    Accepts ``None`` (-> 1..22), a single chrom (``"7"``), an inclusive
    range (``"1-22"``), a comma-list (``"1,2,3"``), or combinations
    (``"1-4,8"``).
    """
    if spec is None:
        return list(range(1, 23))
    chroms: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            chroms.extend(range(int(lo), int(hi) + 1))
        else:
            chroms.append(int(part))
    return chroms


def setup_heritability(
    cfg: HVAEConfig,
    data: dict,
) -> HeritabilityState:
    """Set up heritability loss functions / rank-B modules.

    Three orthogonal flags determine the path:

    * ``--r2``    → genotype HDF5 + ``var_exp()``. Mutually exclusive
      with ``--kinship`` and ``--genetic-correlation``.
    * ``--kinship`` → kinship HDF5 + direct ``mom()`` (or ``gc()`` with
      ``--genetic-correlation``).
    * neither      → PLINK ``.bed/.bim/.fam`` + rank-B heritability,
      with mode ``mom`` (default) or ``gc`` (with ``--genetic-correlation``).
    """
    if cfg.r2 and cfg.kinship:
        raise ValueError("--r2 and --kinship are mutually exclusive")
    if cfg.r2 and cfg.genetic_correlation is not None:
        raise ValueError("--r2 and --genetic-correlation are mutually exclusive")
    if cfg.linear_heritability and (cfg.r2 or cfg.kinship or cfg.genetic_correlation is not None):
        raise ValueError(
            "--linear-heritability is the PLINK rank-B path only; it is "
            "incompatible with --r2, --kinship, and --genetic-correlation"
        )
    if cfg.rg_ldsc_sumstats is not None:
        if cfg.r2 or cfg.kinship:
            raise ValueError("--rg-ldsc-sumstats requires the PLINK rank-B path "
                             "(incompatible with --r2 / --kinship)")
        if cfg.genetic_correlation is not None:
            raise ValueError("--rg-ldsc-sumstats and --genetic-correlation are "
                             "mutually exclusive (external vs. in-cohort gc)")
        if cfg.linear_heritability:
            raise ValueError("--rg-ldsc-sumstats and --linear-heritability are "
                             "mutually exclusive (both wrap the rank-B objective)")
        if cfg.rg_ldsc_ref_ld_chr is None or cfg.rg_ldsc_w_ld_chr is None:
            raise ValueError("--rg-ldsc-sumstats requires --rg-ldsc-ref-ld-chr "
                             "and --rg-ldsc-w-ld-chr")

    device = cfg.device
    cov_state = CovariateState()
    n_train = data["n_train"]
    h_state = HeritabilityState(cov_state=cov_state)

    # --- Covariate names ---
    cov_names = data.get("covariate_names")

    # --- Decode covariates pathway ---
    if cfg.decode_covariates is not None:
        selected = read_lines(cfg.decode_covariates)
        if cov_names is None:
            raise ValueError("--decode-covariates requires covariate_names in covariates HDF5")
        C_train = data["train"]["covariates"]
        C_val = data["val"]["covariates"]
        if C_train is None:
            raise ValueError("--decode-covariates requires --covariates HDF5")
        cov_state.decode_train = select_covariate_columns(C_train, cov_names, selected).to(device)
        cov_state.decode_val = select_covariate_columns(C_val, cov_names, selected).to(device)

    # --- Residualize covariates pathway ---
    C_resid_train = None
    C_resid_val = None
    if cfg.residualize_covariates is not None:
        selected = read_lines(cfg.residualize_covariates)
        if cov_names is None:
            raise ValueError("--residualize-covariates requires covariate_names in covariates HDF5")
        C_train = data["train"]["covariates"]
        C_val = data["val"]["covariates"]
        if C_train is None:
            raise ValueError("--residualize-covariates requires --covariates HDF5")
        C_resid_train = select_covariate_columns(C_train, cov_names, selected).to(device)
        C_resid_val = select_covariate_columns(C_val, cov_names, selected).to(device)

    # --- Target phenotype (for genetic-correlation mode) ---
    use_gc = cfg.genetic_correlation is not None
    y2_train = y2_val = None
    if use_gc:
        y2_train = data["train"]["target_phenotype"]
        y2_val = data["val"]["target_phenotype"]
        if y2_train is None:
            raise ValueError("--genetic-correlation specified but target_phenotype absent from data dict")
        y2_train = y2_train.to(device)
        y2_val = y2_val.to(device)

    # --- Per-latent heritability weights (used by all paths) ---
    hweights_tensor = None
    if cfg.hweights is not None:
        hweights_tensor = read_weights(cfg.hweights).to(device)
        h_state.hweights = hweights_tensor

    # --- PLINK rank-B path (default when neither --r2 nor --kinship) ---
    plink_prefix = data["genetics"].get("plink_prefix")
    if not cfg.r2 and not cfg.kinship:
        if plink_prefix is None:
            raise ValueError(
                "expected --genetics to point to PLINK .bed/.bim/.fam files; "
                "use --kinship for a precomputed-kinship HDF5 or --r2 for a "
                "genotype HDF5"
            )
        from h2vae.plink import BedFile
        row_idx_full = data["genetics"]["plink_row_idx"]
        row_idx_train = row_idx_full[:n_train]
        row_idx_val = row_idx_full[n_train:]
        bed = BedFile(plink_prefix)

        # Pick the rank-B estimator class.  --linear-heritability swaps the
        # per-dim objective for the heritability-SPECTRUM objective
        # (RankBHeritabilitySpectrum); --hweights then acts as per-rank weights.
        if cfg.linear_heritability:
            if use_gc:
                raise ValueError(
                    "--linear-heritability is incompatible with --genetic-correlation"
                )

            def _make_rank_b(_bed, _row_idx, _C, _y_target):
                return RankBHeritabilitySpectrum(
                    _bed, _row_idx,
                    C=_C,
                    ridge=cfg.spectrum_ridge,
                    spectrum_clamp=cfg.spectrum_clamp,
                    rank_weights=hweights_tensor,
                    spectrum_dims=cfg.spectrum_dims,
                    spectrum_weight=cfg.spectrum_weight,
                    marginal_weight=cfg.marginal_weight,
                    device=device,
                )
        else:
            def _make_rank_b(_bed, _row_idx, _C, _y_target):
                return RankBHeritability(
                    _bed, _row_idx,
                    C=_C,
                    y_target=_y_target if use_gc else None,
                    hweights=hweights_tensor,
                    device=device,
                )

        h_state.rank_b = _make_rank_b(bed, row_idx_train, C_resid_train, y2_train)
        h_state.rank_b_val = _make_rank_b(bed, row_idx_val, C_resid_val, y2_val)

        if cfg.split_variants:
            odd_genetics = data.get("genetics_odd", {})
            odd_prefix = odd_genetics.get("plink_prefix") if odd_genetics else None
            if odd_prefix is None:
                raise ValueError(
                    "--split-variants on PLINK requires "
                    f"{cfg.genetics}.odd.bed/.bim/.fam to exist"
                )
            odd_row_idx_full = odd_genetics["plink_row_idx"]
            bed_odd = BedFile(odd_prefix)
            h_state.rank_b_odd = _make_rank_b(
                bed_odd, odd_row_idx_full[:n_train], C_resid_train, y2_train)
            h_state.rank_b_val_odd = _make_rank_b(
                bed_odd, odd_row_idx_full[n_train:], C_resid_val, y2_val)

        # --- External LDSC genetic-correlation: wrap the rank-B modules ------
        # RankBGenCorrLDSC is a duck-typed drop-in (rebuild / update_and_loss /
        # display); it optimizes the genetic COVARIANCE vs an external trait and
        # exposes per-dim rg (last_rg), gencov (last_gencov), and the IRWLS
        # intercepts (last_intercepts) for the per-epoch diagnostics.
        if cfg.rg_ldsc_sumstats is not None:
            from h2vae.ldsc_io import build_ldsc_context
            from h2vae.rank_b_gencorr_ldsc import RankBGenCorrLDSC

            chroms = _parse_chroms_spec(cfg.rg_ldsc_chroms)
            ctx = build_ldsc_context(
                cfg.rg_ldsc_sumstats,
                cfg.rg_ldsc_ref_ld_chr, cfg.rg_ldsc_w_ld_chr,
                bed_variant_ids=bed.variant_ids,
                bed_a1=bed.a1, bed_a2=bed.a2,
                chroms=chroms,
            )
            logging.info(
                f"[rg-ldsc] aligned {ctx.m_use} SNPs (from {ctx.n_total_input} "
                f"in sumstats, {ctx.n_annot} ref-LD annotations) for rg loss"
            )
            h_state.rank_b = RankBGenCorrLDSC(
                h_state.rank_b, ctx,
                intercept_hsq=cfg.rg_ldsc_intercept_hsq,
                intercept_gencov=cfg.rg_ldsc_intercept_gencov,
                hweights=hweights_tensor,
            )
            h_state.rank_b_val = RankBGenCorrLDSC(
                h_state.rank_b_val, ctx,
                intercept_hsq=cfg.rg_ldsc_intercept_hsq,
                intercept_gencov=cfg.rg_ldsc_intercept_gencov,
                hweights=hweights_tensor,
            )
            if cfg.split_variants:
                ctx_odd = build_ldsc_context(
                    cfg.rg_ldsc_sumstats,
                    cfg.rg_ldsc_ref_ld_chr, cfg.rg_ldsc_w_ld_chr,
                    bed_variant_ids=bed_odd.variant_ids,
                    bed_a1=bed_odd.a1, bed_a2=bed_odd.a2,
                    chroms=chroms,
                )
                logging.info(f"[rg-ldsc] odd: aligned {ctx_odd.m_use} SNPs")
                h_state.rank_b_odd = RankBGenCorrLDSC(
                    h_state.rank_b_odd, ctx_odd,
                    intercept_hsq=cfg.rg_ldsc_intercept_hsq,
                    intercept_gencov=cfg.rg_ldsc_intercept_gencov,
                    hweights=hweights_tensor,
                )
                h_state.rank_b_val_odd = RankBGenCorrLDSC(
                    h_state.rank_b_val_odd, ctx_odd,
                    intercept_hsq=cfg.rg_ldsc_intercept_hsq,
                    intercept_gencov=cfg.rg_ldsc_intercept_gencov,
                    hweights=hweights_tensor,
                )
        return h_state

    # --- Build heritability estimators (kinship or --r2 genotype-HDF5 path) ---
    assert cfg.r2 or cfg.kinship  # PLINK path returned above
    if cfg.kinship:
        # Fixed callables from kinship matrices.
        K = data["genetics"]["kinship"]
        if K is None:
            raise ValueError("--kinship requires kinship in genetics HDF5")
        K_train = K[:n_train, :n_train].to(device)
        K_val = K[n_train:, n_train:].to(device)
        if use_gc:
            h_state.loss_fn = gc(K_train, y2_train, kinship=True, C=C_resid_train, device=device)
            h_state.val_fn = gc(K_val, y2_val, kinship=True, C=C_resid_val, device=device)
        else:
            h_state.loss_fn = mom(K_train, kinship=True, C=C_resid_train, device=device)
            h_state.val_fn = mom(K_val, kinship=True, C=C_resid_val, device=device)

        if cfg.split_variants:
            K_odd = data["genetics_odd"]["kinship"]
            if K_odd is None:
                raise ValueError("--split-variants with --kinship requires kinship in odd genetics HDF5")
            K_odd_train = K_odd[:n_train, :n_train].to(device)
            K_odd_val = K_odd[n_train:, n_train:].to(device)
            if use_gc:
                h_state.loss_fn_odd = gc(K_odd_train, y2_train, kinship=True, C=C_resid_train, device=device)
                h_state.val_fn_odd = gc(K_odd_val, y2_val, kinship=True, C=C_resid_val, device=device)
            else:
                h_state.loss_fn_odd = mom(K_odd_train, kinship=True, C=C_resid_train, device=device)
                h_state.val_fn_odd = mom(K_odd_val, kinship=True, C=C_resid_val, device=device)
    else:
        # Genotype mode. Without --genetic-correlation use full var_exp (exact OLS R²);
        # with --genetic-correlation use gc (which internally builds a GRM from genotypes).
        if not use_gc and C_resid_train is None:
            raise ValueError("--residualize-covariates is required when using genotypes (no --kinship)")

        G = data["genetics"]["genotypes"]
        if G is None:
            raise ValueError("Genotypes required in genetics HDF5 (or use --kinship)")
        G_train = G[:n_train, :].to(device)
        G_val = G[n_train:, :].to(device)
        if use_gc:
            h_state.loss_fn = gc(G_train, y2_train, kinship=False, C=C_resid_train, device=device)
            h_state.val_fn = gc(G_val, y2_val, kinship=False, C=C_resid_val, device=device)
        else:
            h_state.loss_fn = var_exp(G_train, C_resid_train, device)
            h_state.val_fn = var_exp(G_val, C_resid_val, device)

        if cfg.split_variants:
            G_odd = data["genetics_odd"]["genotypes"]
            if G_odd is None:
                raise ValueError("--split-variants requires genotypes in odd genetics HDF5")
            G_odd_train = G_odd[:n_train, :].to(device)
            G_odd_val = G_odd[n_train:, :].to(device)
            if use_gc:
                h_state.loss_fn_odd = gc(G_odd_train, y2_train, kinship=False, C=C_resid_train, device=device)
                h_state.val_fn_odd = gc(G_odd_val, y2_val, kinship=False, C=C_resid_val, device=device)
            else:
                h_state.loss_fn_odd = var_exp(G_odd_train, C_resid_train, device)
                h_state.val_fn_odd = var_exp(G_odd_val, C_resid_val, device)

    return h_state


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def compute_heritability_loss(
    Z: Tensor,
    her_loss_fn: Callable,
    hweights: Tensor | None,
    idxs: Tensor | None = None,
) -> Tensor:
    """Heritability loss across all latent dimensions.

    The loss callable standardizes y internally, so Z is passed raw.
    Only rows at ``idxs`` are kept in the computation graph; all other
    rows are detached so backprop only flows through the current batch.

    Args:
        Z: Full latent matrix, shape (n, zdim).
        her_loss_fn: Batched callable accepting ``(n, zdim)`` and returning ``(zdim,)``.
        hweights: Optional per-latent weights.
        idxs: Indices of the current batch's rows in Z.
    """
    if idxs is not None:
        # Detach non-batch rows; keep batch rows live in the graph.
        Z_detached = Z.detach().clone()
        Z_detached[idxs] = Z[idxs]
        Z = Z_detached

    her_per_dim = -her_loss_fn(Z)  # (zdim,)
    if hweights is not None:
        return (her_per_dim * hweights).sum()
    return her_per_dim.sum()


def compute_correlation_loss(cs_Z: Tensor) -> Tensor:
    """Sum of absolute upper-triangular correlation coefficients.

    Args:
        cs_Z: Pre-centered/scaled latent matrix, shape (n, zdim).
    """
    corr_matrix = corrcoef(cs_Z.T)
    return torch.abs(torch.triu(corr_matrix, diagonal=1)).sum()


def compute_moment_loss(cs_Z: Tensor) -> Tensor:
    """Skewness + excess kurtosis penalty toward standard normal.

    Args:
        cs_Z: Pre-centered/scaled latent matrix, shape (n, zdim).
    """
    n = cs_Z.shape[0]
    skewness = torch.abs((cs_Z ** 3).sum()) / n
    excess_kurtosis = torch.abs((cs_Z ** 4).sum() / n - 3.0)
    return skewness + excess_kurtosis


def compute_kl_penalty(zs: Tensor, K: int) -> Tensor:
    """KL-like penalty from encoder log-variance. K = total pixel count."""
    return (-0.5 * zs.sum(1)[:, None] / K).sum()


def composite_loss(
    vae_loss: Tensor,
    zs: Tensor,
    Z: Tensor,
    her_loss_fn: Callable | list[Callable] | None,
    hweights: Tensor | None,
    K: int,
    weights: LossWeights,
    idxs: Tensor | None = None,
    rank_b: RankBHeritability | None = None,
    zm_batch: Tensor | None = None,
) -> tuple[Tensor, dict]:
    """Compute the weighted composite loss.

    Args:
        idxs: Batch indices, required when ``her_loss_fn`` is a list
            (Taylor mode) **or** when ``rank_b`` is provided.
        rank_b: Optional rank-B heritability module. When set, the
            heritability term comes from ``rank_b.update_and_loss``;
            ``her_loss_fn`` is ignored.
        zm_batch: Minibatch rows of ``Zm`` (live in the graph). Only
            consumed when ``rank_b`` is given.

    Returns:
        loss: Scalar loss tensor.
        metrics: Dict of per-component loss values (detached).
    """
    loss = weights.mse * vae_loss.sum()
    metrics: dict[str, float] = {"vae_loss": vae_loss.sum().item()}

    if weights.h > 0:
        if rank_b is not None:
            if zm_batch is None or idxs is None:
                raise ValueError(
                    "rank_b requires both zm_batch and idxs"
                )
            h_loss = rank_b.update_and_loss(zm_batch, idxs)
        else:
            h_loss = compute_heritability_loss(Z, her_loss_fn, hweights, idxs=idxs)
        loss = loss + weights.h * h_loss
        metrics["her_loss"] = h_loss.item()

    if weights.corr > 0 or weights.sk > 0:
        cs_Z = center_and_scale(Z)
        if weights.corr > 0:
            c_loss = compute_correlation_loss(cs_Z)
            loss = loss + weights.corr * c_loss
            metrics["corr_loss"] = c_loss.item()
        if weights.sk > 0:
            sk_loss = compute_moment_loss(cs_Z)
            loss = loss + weights.sk * sk_loss
            metrics["sk_loss"] = sk_loss.item()

    pen = compute_kl_penalty(zs, K)
    loss = loss + pen
    metrics["pen_term"] = pen.item()

    return loss, metrics


# ---------------------------------------------------------------------------
# Display heritability estimates
# ---------------------------------------------------------------------------

def _compute_her_estimates(
    Z: Tensor,
    her_fn: Callable,
) -> list[float]:
    """Compute per-dimension heritability estimates for display.

    If ``her_fn`` exposes a ``.display`` attribute (as ``gc()`` does), the
    display callable is used instead of the bare loss callable — for
    genetic-correlation training the loss is the genetic covariance, but
    we want to log the bounded ``ρ̂`` form.

    Args:
        Z: Latent matrix, shape (n, zdim).
        her_fn: Loss callable; uses ``her_fn.display`` if present.
    """
    fn = getattr(her_fn, "display", her_fn)
    Z = Z.detach()
    return [fn(Z[:, i:i + 1]).item() for i in range(Z.shape[1])]


# ---------------------------------------------------------------------------
# Encode phase
# ---------------------------------------------------------------------------

def encode_all(
    vae: BaseVAE,
    loader: DataLoader,
    zdim: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Encode the full dataset in eval mode.

    Returns:
        Zm: Latent means, shape (n, zdim).
        Zs: Latent stds, shape (n, zdim).
    """
    vae.eval()
    n = len(loader.dataset)
    Zm = torch.zeros(n, zdim, device=device)
    Zs = torch.zeros(n, zdim, device=device)

    with torch.no_grad():
        for data in loader:
            y = data[0].to(device, non_blocking=True)
            idxs = data[-1]
            # SPEEDUPS #6: bf16 autocast on encode only; cast back to fp32 so
            # downstream consumers (heritability, KL, latent dump) stay fp32.
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                zm, zs = vae.encode(y)
            Zm[idxs] = zm.float()
            Zs[idxs] = zs.float()

    return Zm, Zs


# ---------------------------------------------------------------------------
# Training and validation epochs
# ---------------------------------------------------------------------------

def train_epoch(
    vae: BaseVAE,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    h_state: HeritabilityState,
    cfg: HVAEConfig,
    epoch: int,
) -> dict:
    """Run one training epoch (encode-then-backprop).

    Returns dict with keys: mse, loss, Zm_start. The display heritability
    estimates and the per-epoch ``Zm_train`` savetxt are deferred to
    ``main()``: the *next* epoch's start-of-epoch encode (which sees
    post-update weights) is reused as this epoch's display state, dropping
    one full encode_all pass per epoch.
    """
    device = cfg.device
    weights = cfg.loss_weights

    # Phase 1: encode full dataset
    Zm, Zs = encode_all(vae, train_loader, cfg.zdim, device)
    Eps = torch.randn_like(Zs)

    # Heritability/corr/moment losses run on the posterior MEAN buffer (Zm),
    # not on a sample (Z = Zm + Eps*Zs). Routing the gradient through zm only
    # means dense_zs receives no gradient from these terms, eliminating the
    # zs->0 collapse pathway that drove softplus underflow under bf16.
    # The display path (`_compute_her_estimates`) was already on Zm, so this
    # also aligns the optimization target with the displayed metric.
    Zm_buf = Zm.clone()

    # Rank-B mode: refresh u_raw and w_raw with the full-cohort Zm
    # (one BED stream). Subsequent minibatch loss calls update both
    # rank-B-style and require no further BED stream.
    if h_state.rank_b is not None:
        h_state.rank_b.rebuild(Zm)
        if h_state.rank_b_odd is not None:
            h_state.rank_b_odd.rebuild(Zm)

    her_loss_fn = h_state.loss_fn

    # Phase 2: mini-batch backprop
    vae.train()
    epoch_mse = 0.0
    epoch_loss = 0.0

    for data in train_loader:
        Zm_buf = Zm_buf.detach()
        y = data[0].to(device, non_blocking=True)
        idxs = data[-1]
        eps = Eps[idxs]

        c_batch = None
        if h_state.cov_state.decode_train is not None:
            c_batch = h_state.cov_state.decode_train[idxs]

        # SPEEDUPS #6: bf16 autocast on encode + decode only. zm/zs/xr are
        # cast back to fp32 immediately so the heritability path, KL term
        # (sensitive to log(zs)), and MSE stay in fp32.
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            zm, zs = vae.encode(y)
        zm, zs = zm.float(), zs.float()
        z = zm + zs * eps
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            xr = vae.decode(z, c_batch)
        xr = xr.float()
        mse = vae.mse(y, xr)
        kld = (
            -0.5 * (1 + 2 * torch.log(zs) - zm ** 2 - zs ** 2).sum(1)[:, None]
            / vae.K
        )
        vae_loss = mse + vae.beta * kld

        Zm_buf[idxs] = zm

        loss, metrics = composite_loss(
            vae_loss, zs, Zm_buf, her_loss_fn, h_state.hweights, vae.K, weights,
            idxs=idxs, rank_b=h_state.rank_b, zm_batch=zm,
        )

        optimizer.zero_grad()
        loss.backward()
        if cfg.clip > 0:
            torch.nn.utils.clip_grad_norm_(vae.parameters(), cfg.clip)
        optimizer.step()

        epoch_mse += mse.sum().item()
        epoch_loss += loss.item()

    # Zm here is the start-of-epoch encode (post-(prev epoch) weights). main()
    # uses the next epoch's Zm to display + savetxt this epoch's results.
    return {"mse": epoch_mse, "loss": epoch_loss, "Zm_start": Zm}


def validate_epoch(
    vae: BaseVAE,
    val_loader: DataLoader,
    h_state: HeritabilityState,
    cfg: HVAEConfig,
    epoch: int,
    Zm_train: Tensor | None = None,
) -> dict:
    """Run validation: encode, compute MSE and heritability.

    Single pass over ``val_loader``: encode, decode (using zm directly,
    matching the previous "z = Zm[idxs]" identification), and accumulate
    MSE — eliminates the second val_loader iteration that re-decompressed
    every NIfTI volume (SPEEDUPS #5).

    Returns dict with keys: mse_val, her_estimates_val, and optionally
    her_estimates_val_odd when split-variants is active.
    """
    device = cfg.device

    vae.eval()
    n = len(val_loader.dataset)
    Zm = torch.zeros(n, cfg.zdim, device=device)

    mse_val = 0.0
    with torch.no_grad():
        for data in val_loader:
            y = data[0].to(device, non_blocking=True)
            idxs = data[-1]
            # SPEEDUPS #6: bf16 autocast on encode + decode only.
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                zm, _ = vae.encode(y)
            zm = zm.float()
            Zm[idxs] = zm

            c_batch = None
            if h_state.cov_state.decode_val is not None:
                c_batch = h_state.cov_state.decode_val[idxs]
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                xr = vae.decode(zm, c_batch)
            xr = xr.float()
            mse_val += vae.mse(y, xr).sum().item()

    np.savetxt(
        os.path.join(cfg.outdir, "latents", f"Zm_val.{epoch:05d}.txt"),
        Zm.detach().cpu().numpy(),
        delimiter="\t",
    )

    # Display estimates on posterior means for stability
    result: dict = {"mse_val": mse_val}
    if h_state.rank_b_val is not None:
        result["her_estimates_val"] = h_state.rank_b_val.display(Zm).tolist()
        if cfg.split_variants:
            result["her_estimates_val_odd"] = h_state.rank_b_val_odd.display(Zm).tolist()
    else:
        result["her_estimates_val"] = _compute_her_estimates(Zm, h_state.val_fn)
        if cfg.split_variants:
            result["her_estimates_val_odd"] = _compute_her_estimates(Zm, h_state.val_fn_odd)

    # rg-ldsc: the display() above populated last_gencov / last_intercepts /
    # last_skipped on the wrapper. Stash the val gencov for the flush log line
    # and emit the per-epoch IRWLS-intercept diagnostic (mean±std across dims).
    if cfg.rg_ldsc_sumstats is not None and h_state.rank_b_val is not None:
        rb = h_state.rank_b_val
        result["gencov_val"] = rb.last_gencov.tolist()
        ints = rb.last_intercepts

        def _ms(t: torch.Tensor) -> tuple[float, float]:
            v = t[~torch.isnan(t)]
            if v.numel() == 0:
                return float("nan"), float("nan")
            return float(v.mean()), float(v.std())

        h1m, h1s = _ms(ints["hsq1"])
        h2m, h2s = _ms(ints["hsq2"])
        gcm, gcs = _ms(ints["gencov"])
        logging.info(
            "rg_ldsc_intercepts_val (epoch %d): hsq1=%.3f±%.3f  "
            "hsq2=%.3f±%.3f  gencov=%.3f±%.3f  skipped=%d/%d",
            epoch, h1m, h1s, h2m, h2s, gcm, gcs,
            len(rb.last_skipped), len(result["her_estimates_val"]),
        )

    # Heritability-spectrum objective: additionally report the val spectrum and
    # persist it (the per-dim h² above stays as-is). spectrum_display() exists
    # only on RankBHeritabilitySpectrum.
    if cfg.linear_heritability and h_state.rank_b_val is not None:
        spec, total = h_state.rank_b_val.spectrum_display(Zm)
        result["spectrum_val"] = spec.detach().cpu().tolist()
        result["spectrum_total_val"] = float(total)
        np.savetxt(
            os.path.join(cfg.outdir, "latents", f"spectrum_val.{epoch:05d}.txt"),
            spec.detach().cpu().numpy(),
            delimiter="\t",
        )
        # Held-out spectrum: score the TRAIN-optimal directions on val data.
        # W_train is fit on train latents; evaluated on val's own (G,P) it gives
        # the unbiased generalization of the spectrum (overfit directions deflate).
        if Zm_train is not None and h_state.rank_b is not None:
            _, W_train = h_state.rank_b.eig_decompose(Zm_train)
            held = h_state.rank_b_val.heritability_of_directions(Zm, W_train)
            held = held.detach().cpu()
            result["spectrum_val_heldout"] = held.tolist()
            result["spectrum_total_val_heldout"] = float(held.sum())
            np.savetxt(
                os.path.join(cfg.outdir, "latents",
                             f"spectrum_val_heldout.{epoch:05d}.txt"),
                held.numpy(), delimiter="\t",
            )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_split_ids(outdir: str, train_ids: np.ndarray, val_ids: np.ndarray) -> None:
    """Save train/val sample IDs to the output directory."""
    np.save(os.path.join(outdir, "train_ids.npy"), train_ids)
    np.save(os.path.join(outdir, "val_ids.npy"), val_ids)


def load_split_ids(outdir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load train/val sample IDs from a previous run's output directory."""
    train_path = os.path.join(outdir, "train_ids.npy")
    val_path = os.path.join(outdir, "val_ids.npy")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Split ID files not found in {outdir}")
    return np.load(train_path), np.load(val_path)


def main() -> None:
    cfg = parse_args()
    wdir, fdir, ldir = setup_output_dirs(cfg.outdir)
    setup_logging(cfg.outdir)
    logging.info("config = %s", cfg)

    torch.manual_seed(cfg.seed)
    device = cfg.device

    # SPEEDUPS #4: NIfTI volumes pad to a fixed cubic target_size, so input
    # shapes are stable; cuDNN benchmark picks the best Conv3d algorithm once
    # and reuses it. (Global TF32 is intentionally NOT enabled here — it would
    # silently downgrade heritability matmuls below the fp32 floor.)
    torch.backends.cudnn.benchmark = True

    if cfg.debug:
        breakpoint()

    # --- Resume: load split IDs from previous run ---
    resume_train_ids = None
    resume_val_ids = None
    start_epoch = 0

    if cfg.resume is not None:
        ckpt_path, ckpt_epoch = find_latest_checkpoint(cfg.resume)
        if ckpt_path is not None:
            logging.info("resuming from %s (epoch %d)", ckpt_path, ckpt_epoch)
            start_epoch = ckpt_epoch + 1
            resume_train_ids, resume_val_ids = load_split_ids(cfg.resume)
        else:
            logging.info("--resume specified but no checkpoints found in %s", cfg.resume)

    # --- Data ---
    from h2vae.plink import is_plink_prefix
    genetics_path = cfg.genetics
    if cfg.split_variants:
        # Prefer a PLINK prefix at `{cfg.genetics}.even`; fall back to
        # `{cfg.genetics}.even.hdf5`.
        even_plink = f"{cfg.genetics}.even"
        if is_plink_prefix(even_plink):
            genetics_path = even_plink
        else:
            genetics_path = f"{cfg.genetics}.even.hdf5"

    # Union of covariate names whose completeness gates sample inclusion.
    # Lets the covariates HDF5 carry NaNs for unmeasured fields; the cohort
    # is then filtered per-run to only those columns actually in use.
    required_covs: list[str] = []
    for path in (cfg.decode_covariates, cfg.residualize_covariates):
        if path is not None:
            required_covs.extend(read_lines(path))
    required_covs = list(dict.fromkeys(required_covs))

    data = load_data(
        images_path=cfg.images,
        genetics_path=genetics_path,
        covariates_path=cfg.covariates,
        target_phenotype_path=cfg.genetic_correlation,
        train_frac=cfg.train_frac,
        seed=cfg.seed,
        train_ids=resume_train_ids,
        val_ids=resume_val_ids,
        required_covariates=required_covs or None,
    )

    if cfg.split_variants:
        odd_plink = f"{cfg.genetics}.odd"
        if is_plink_prefix(odd_plink):
            odd_path = odd_plink
        else:
            odd_path = f"{cfg.genetics}.odd.hdf5"
        all_ids = np.concatenate([data["train_ids_raw"], data["val_ids_raw"]])
        data["genetics_odd"] = load_genetics_reindexed(odd_path, all_ids)
        logging.info("split-variants: loaded even from %s, odd from %s", genetics_path, odd_path)

    logging.info(
        "dataset sizes: n_train=%d, n_val=%d",
        len(data["train_ids_raw"]), len(data["val_ids_raw"]),
    )

    # Save split IDs for downstream analysis and future resumption
    save_split_ids(cfg.outdir, data["train_ids_raw"], data["val_ids_raw"])

    ModelClass = get_model_class(cfg.model)
    streaming_kwargs = {}
    if ModelClass.data_format == "nifti":
        # NIFTI volumes need padding to a uniform cubic size for batching.
        # Use the explicit --img-size if provided, otherwise the model default.
        import inspect
        model_default = inspect.signature(ModelClass.__init__).parameters["img_size"].default
        streaming_kwargs["target_size"] = cfg.img_size if cfg.img_size is not None else model_default
    if data["streaming"]:
        train_dataset = make_streaming_dataset(
            data["train"]["image_paths"], data["train"]["ids"], ModelClass.data_format,
            **streaming_kwargs,
        )
        val_dataset = make_streaming_dataset(
            data["val"]["image_paths"], data["val"]["ids"], ModelClass.data_format,
            **streaming_kwargs,
        )
    else:
        train_dataset = ImageDataset(data["train"]["images"], data["train"]["ids"])
        val_dataset = ImageDataset(data["val"]["images"], data["val"]["ids"])

    # NIFTI volumes require gzip decompression per file — more workers needed
    # to keep the GPU fed vs lightweight PNG decoding.
    nw = 16 if ModelClass.data_format == "nifti" else 8
    loader_kwargs = {
        "num_workers": nw,
        "prefetch_factor": 2,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": True,
    }

    train_loader = DataLoader(train_dataset, batch_size=cfg.bs, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=cfg.bs, shuffle=False, **loader_kwargs)

    # --- Heritability + covariates ---
    h_state = setup_heritability(cfg, data)

    # --- Model ---
    ModelClass = get_model_class(cfg.model)
    vae_cfg = cfg.vae_cfg
    if h_state.cov_state.decode_train is not None:
        vae_cfg["external"] = h_state.cov_state.decode_train.shape[1]
    else:
        vae_cfg["external"] = 0

    vae_cfg["_model_class"] = cfg.model
    pickle.dump(vae_cfg, open(os.path.join(cfg.outdir, "vae.cfg.p"), "wb"))
    constructor_cfg = {k: v for k, v in vae_cfg.items() if not k.startswith("_")}
    vae = ModelClass(**constructor_cfg).to(device)

    # Load checkpoint weights
    if cfg.resume is not None and start_epoch > 0:
        ckpt_path, _ = find_latest_checkpoint(cfg.resume)
        vae.load_state_dict(torch.load(ckpt_path, map_location=device))

    # --- Optimizer ---
    optimizer = optim.Adam(vae.parameters(), lr=cfg.vae_lr)

    # --- Training loop ---
    # Per-epoch train her_estimates + Zm_train savetxt are computed using the
    # *next* epoch's start-of-epoch encode (SPEEDUPS #2): that encode sees the
    # post-update weights for free, removing one full encode_all per epoch.
    # We therefore buffer one epoch's mse/loss/val metrics and flush them on
    # the next iteration once Zm_start is available.
    def _flush_pending(pending: dict | None, Zm_for_train: Tensor) -> None:
        if pending is None:
            return
        prev_epoch = pending["epoch"]
        np.savetxt(
            os.path.join(ldir, f"Zm_train.{prev_epoch:05d}.txt"),
            Zm_for_train.detach().cpu().numpy(),
            delimiter="\t",
        )
        if h_state.rank_b is not None:
            her_train = h_state.rank_b.display(Zm_for_train).tolist()
            her_train_odd = (
                h_state.rank_b_odd.display(Zm_for_train).tolist()
                if cfg.split_variants else None
            )
        else:
            her_train = _compute_her_estimates(Zm_for_train, h_state.loss_fn)
            her_train_odd = (
                _compute_her_estimates(Zm_for_train, h_state.loss_fn_odd)
                if cfg.split_variants else None
            )
        val = pending["val_metrics"]
        if cfg.split_variants:
            logging.info(
                "epoch %d - mse_train: %.4f - mse_val: %.4f"
                " - h_train_even: %s - h_train_odd: %s"
                " - h_val_even: %s - h_val_odd: %s",
                prev_epoch,
                pending["mse"],
                val["mse_val"],
                ", ".join(f"{h:.3f}" for h in her_train),
                ", ".join(f"{h:.3f}" for h in her_train_odd),
                ", ".join(f"{h:.3f}" for h in val["her_estimates_val"]),
                ", ".join(f"{h:.3f}" for h in val["her_estimates_val_odd"]),
            )
        else:
            logging.info(
                "epoch %d - mse_train: %.4f - mse_val: %.4f - h_train: %s - h_val: %s",
                prev_epoch,
                pending["mse"],
                val["mse_val"],
                ", ".join(f"{h:.3f}" for h in her_train),
                ", ".join(f"{h:.3f}" for h in val["her_estimates_val"]),
            )

        # rg-ldsc: log the optimized per-dim genetic COVARIANCE (train + val).
        # rank_b.display(Zm_for_train) above set last_gencov for train; val
        # gencov was stashed by validate_epoch.
        if cfg.rg_ldsc_sumstats is not None and h_state.rank_b is not None:
            gencov_tr = h_state.rank_b.last_gencov.tolist()
            gencov_va = val.get("gencov_val", [])
            logging.info(
                "epoch %d - gencov_train: %s - gencov_val: %s",
                prev_epoch,
                ", ".join(f"{g:+.4e}" for g in gencov_tr),
                ", ".join(f"{g:+.4e}" for g in gencov_va),
            )

        # Heritability-spectrum objective: additionally log the spectrum total +
        # top-5 and persist the train spectrum (additive; no log line removed).
        if cfg.linear_heritability and h_state.rank_b is not None:
            spec_tr, tot_tr = h_state.rank_b.spectrum_display(Zm_for_train)
            spec_tr = spec_tr.detach().cpu()
            np.savetxt(
                os.path.join(ldir, f"spectrum_train.{prev_epoch:05d}.txt"),
                spec_tr.numpy(),
                delimiter="\t",
            )
            spec_val = val.get("spectrum_val", [])
            spec_held = val.get("spectrum_val_heldout", [])
            logging.info(
                "epoch %d - h2_total_train: %.4f - top5_train: %s"
                " - h2_total_val: %.4f - top5_val: %s"
                " - h2_total_val_heldout: %.4f - top5_val_heldout: %s",
                prev_epoch,
                float(tot_tr),
                ", ".join(f"{x:.3f}" for x in spec_tr[:5].tolist()),
                val.get("spectrum_total_val", float("nan")),
                ", ".join(f"{x:.3f}" for x in spec_val[:5]),
                val.get("spectrum_total_val_heldout", float("nan")),
                ", ".join(f"{x:.3f}" for x in spec_held[:5]),
            )

    pending: dict | None = None
    for epoch in range(start_epoch, cfg.epochs):
        train_metrics = train_epoch(vae, train_loader, optimizer, h_state, cfg, epoch)
        Zm_start = train_metrics.pop("Zm_start")

        # Use this epoch's start-of-epoch Zm to flush previous epoch's display.
        _flush_pending(pending, Zm_start)

        val_metrics = validate_epoch(vae, val_loader, h_state, cfg, epoch,
                                     Zm_train=Zm_start)

        pending = {
            "epoch": epoch,
            "mse": train_metrics["mse"],
            "loss": train_metrics["loss"],
            "val_metrics": val_metrics,
        }

        if epoch % cfg.epoch_cb == 0:
            logging.info("epoch %d - saving checkpoint", epoch)
            torch.save(vae.state_dict(), os.path.join(wdir, f"weights.{epoch:05d}.pt"))

    # Trailing flush: one final encode_all to finalize the last epoch's display.
    if pending is not None:
        Zm_final, _ = encode_all(vae, train_loader, cfg.zdim, device)
        _flush_pending(pending, Zm_final)


if __name__ == "__main__":
    main()
