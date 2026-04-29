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
    split_variants: bool = False
    genetic_correlation: str | None = None  # HDF5 with target phenotype (switches loss to gc)

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
    parser.add_argument("--kinship", action="store_true", default=False, help="use kinship matrix instead of genotypes")
    parser.add_argument("--split-variants", action="store_true", default=False, help="--genetics is a prefix; load {prefix}.even.hdf5 and {prefix}.odd.hdf5")
    parser.add_argument("--genetic-correlation", type=str, default=None,
                        help="HDF5 (keys 'data', 'ids') of a target phenotype; switches the loss "
                             "to per-latent SCORE-OVERLAP genetic correlation with this phenotype")

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
        split_variants=args.split_variants,
        genetic_correlation=args.genetic_correlation,
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
    """Holds heritability loss functions and related state.

    Kinship mode (``--kinship``): ``loss_fn`` / ``val_fn`` are batched MoM callables.
    Genotype mode (default):       ``loss_fn`` / ``val_fn`` are batched ``var_exp`` callables.
    In both modes, the same callable is used for both backprop loss and display.
    """
    # Backprop loss (even-chromosome when split, single otherwise)
    loss_fn: Callable | None = None
    val_fn: Callable | None = None

    # Display-only odd-chromosome estimators (None when not splitting)
    loss_fn_odd: Callable | None = None
    val_fn_odd: Callable | None = None

    cov_state: CovariateState = None  # type: ignore[assignment]
    hweights: Tensor | None = None


def setup_heritability(
    cfg: HVAEConfig,
    data: dict,
) -> HeritabilityState:
    """Set up heritability loss functions and covariate state.

    Three modes, chosen by flags:
      * ``--kinship`` + no ``--genetic-correlation``: batched ``mom()`` (h² via MoM).
      * no ``--kinship`` + no ``--genetic-correlation``: batched ``var_exp()`` (OLS R²).
      * ``--genetic-correlation <file>``: batched ``gc()`` — per-latent SCORE-OVERLAP
        genetic correlation with the supplied target phenotype.  Composes with
        ``--kinship`` (controls whether the genetics matrix is treated as a
        kinship or a genotype matrix to form the GRM) and ``--split-variants``
        (even chromosomes for backprop, odd for display).
    """
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

    # --- Build heritability estimators ---
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

    # --- Per-latent heritability weights ---
    if cfg.hweights is not None:
        h_state.hweights = read_weights(cfg.hweights).to(device)

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
    her_loss_fn: Callable | list[Callable],
    hweights: Tensor | None,
    K: int,
    weights: LossWeights,
    idxs: Tensor | None = None,
) -> tuple[Tensor, dict]:
    """Compute the weighted composite loss.

    Args:
        idxs: Batch indices, required when ``her_loss_fn`` is a list
            (Taylor mode).

    Returns:
        loss: Scalar loss tensor.
        metrics: Dict of per-component loss values (detached).
    """
    loss = weights.mse * vae_loss.sum()
    metrics: dict[str, float] = {"vae_loss": vae_loss.sum().item()}

    if weights.h > 0:
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

    Args:
        Z: Latent matrix, shape (n, zdim).
        her_fn: A callable that takes (n, 1) and returns a scalar estimate.
            In MoM mode this is the batched ``mom()`` callable.
            In Taylor mode this is the exact ``var_exp()`` callable.
    """
    Z = Z.detach()
    return [her_fn(Z[:, i:i + 1]).item() for i in range(Z.shape[1])]


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
            y = data[0].to(device)
            idxs = data[-1]
            zm, zs = vae.encode(y)
            Zm[idxs] = zm
            Zs[idxs] = zs

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

    Returns dict with keys: mse, loss, her_estimates, and optionally
    her_estimates_odd when split-variants is active.
    """
    device = cfg.device
    weights = cfg.loss_weights

    # Phase 1: encode full dataset
    Zm, Zs = encode_all(vae, train_loader, cfg.zdim, device)
    Eps = torch.randn_like(Zs)
    Z = Zm + Eps * Zs

    her_loss_fn = h_state.loss_fn

    # Phase 2: mini-batch backprop
    vae.train()
    epoch_mse = 0.0
    epoch_loss = 0.0

    for data in train_loader:
        Z = Z.detach()
        y = data[0].to(device)
        idxs = data[-1]
        eps = Eps[idxs]

        c_batch = None
        if h_state.cov_state.decode_train is not None:
            c_batch = h_state.cov_state.decode_train[idxs]

        zm, zs = vae.encode(y)
        z = zm + zs * eps
        xr = vae.decode(z, c_batch)
        mse = vae.mse(y, xr)
        kld = (
            -0.5 * (1 + 2 * torch.log(zs) - zm ** 2 - zs ** 2).sum(1)[:, None]
            / vae.K
        )
        vae_loss = mse + vae.beta * kld

        Z[idxs] = z

        loss, metrics = composite_loss(
            vae_loss, zs, Z, her_loss_fn, h_state.hweights, vae.K, weights,
            idxs=idxs,
        )

        optimizer.zero_grad()
        loss.backward()
        if cfg.clip > 0:
            torch.nn.utils.clip_grad_norm_(vae.parameters(), cfg.clip)
        optimizer.step()

        epoch_mse += mse.sum().item()
        epoch_loss += loss.item()

    # Re-encode with current weights; use posterior means (not samples) for
    # stable h² display — avoids reparameterization noise when zs is large.
    Zm_post, _ = encode_all(vae, train_loader, cfg.zdim, device)

    np.savetxt(
        os.path.join(cfg.outdir, "latents", f"Zm_train.{epoch:05d}.txt"),
        Zm_post.detach().cpu().numpy(),
        delimiter="\t",
    )

    # Display estimates (even / primary)
    result: dict = {"mse": epoch_mse, "loss": epoch_loss}
    result["her_estimates"] = _compute_her_estimates(Zm_post, h_state.loss_fn)

    # Display estimates (odd, if split-variants)
    if cfg.split_variants:
        result["her_estimates_odd"] = _compute_her_estimates(Zm_post, h_state.loss_fn_odd)

    return result


def validate_epoch(
    vae: BaseVAE,
    val_loader: DataLoader,
    h_state: HeritabilityState,
    cfg: HVAEConfig,
    epoch: int,
) -> dict:
    """Run validation: encode, compute MSE and heritability.

    Returns dict with keys: mse_val, her_estimates_val, and optionally
    her_estimates_val_odd when split-variants is active.
    """
    device = cfg.device

    Zm, Zs = encode_all(vae, val_loader, cfg.zdim, device)

    np.savetxt(
        os.path.join(cfg.outdir, "latents", f"Zm_val.{epoch:05d}.txt"),
        Zm.detach().cpu().numpy(),
        delimiter="\t",
    )

    # Display estimates on posterior means for stability
    result: dict = {}
    result["her_estimates_val"] = _compute_her_estimates(Zm, h_state.val_fn)

    # Display estimates (odd, if split-variants)
    if cfg.split_variants:
        result["her_estimates_val_odd"] = _compute_her_estimates(Zm, h_state.val_fn_odd)

    mse_val = 0.0
    with torch.no_grad():
        for data in val_loader:
            y = data[0].to(device)
            idxs = data[-1]
            z = Zm[idxs]
            c_batch = None
            if h_state.cov_state.decode_val is not None:
                c_batch = h_state.cov_state.decode_val[idxs]
            mse = vae.mse(y, vae.decode(z, c_batch))
            mse_val += mse.sum().item()

    result["mse_val"] = mse_val
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
    genetics_path = cfg.genetics
    if cfg.split_variants:
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
    for epoch in range(start_epoch, cfg.epochs):
        train_metrics = train_epoch(vae, train_loader, optimizer, h_state, cfg, epoch)
        val_metrics = validate_epoch(vae, val_loader, h_state, cfg, epoch)

        if cfg.split_variants:
            logging.info(
                "epoch %d - mse_train: %.4f - mse_val: %.4f"
                " - h_train_even: %s - h_train_odd: %s"
                " - h_val_even: %s - h_val_odd: %s",
                epoch,
                train_metrics["mse"],
                val_metrics["mse_val"],
                ", ".join(f"{h:.3f}" for h in train_metrics["her_estimates"]),
                ", ".join(f"{h:.3f}" for h in train_metrics["her_estimates_odd"]),
                ", ".join(f"{h:.3f}" for h in val_metrics["her_estimates_val"]),
                ", ".join(f"{h:.3f}" for h in val_metrics["her_estimates_val_odd"]),
            )
        else:
            logging.info(
                "epoch %d - mse_train: %.4f - mse_val: %.4f - h_train: %s - h_val: %s",
                epoch,
                train_metrics["mse"],
                val_metrics["mse_val"],
                ", ".join(f"{h:.3f}" for h in train_metrics["her_estimates"]),
                ", ".join(f"{h:.3f}" for h in val_metrics["her_estimates_val"]),
            )

        if epoch % cfg.epoch_cb == 0:
            logging.info("epoch %d - saving checkpoint", epoch)
            torch.save(vae.state_dict(), os.path.join(wdir, f"weights.{epoch:05d}.pt"))


if __name__ == "__main__":
    main()
