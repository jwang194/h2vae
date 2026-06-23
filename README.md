# h2vae

Maximizing the genetic signal in a VAE latent space.

h2vae trains a variational autoencoder whose latent dimensions are regularized to be genetically heritable. The training loss combines image reconstruction (MSE + KL) with a differentiable heritability estimator, encouraging the model to learn latent representations that capture genetically driven variation in imaging data.

Heritability can be estimated from a precomputed kinship matrix (Haseman-Elston method-of-moments) or streamed directly from PLINK genotypes via a memory-efficient **rank-B** estimator (the default). On top of the per-dimension objective, two alternatives are available: a rotation-invariant **heritability spectrum** objective (`--linear-heritability`), and an **external-trait genetic-correlation** objective that pushes latents to share genetics with an out-of-cohort GWAS via a differentiable LDSC genetic-covariance loss (`--rg-ldsc-sumstats`).

## Installation

Requires Python 3.10+ with PyTorch, h5py, numpy, scipy, and matplotlib.

```bash
pip install torch h5py numpy scipy matplotlib seaborn pandas
```

## Quick start

### 1. Prepare data

Three HDF5 files, each with an `ids` array for sample alignment:

| File | Required keys | Description |
|------|--------------|-------------|
| Images | `data` (n, h, w, c), `ids` (n,) | Image data (any numeric dtype) |
| Genetics | `kinship` + `kinship_ids` or `genotypes` + `genotype_ids` | Kinship matrix or genotype matrix |
| Covariates | `data` (n, p), `ids` (n,), `covariate_names` (p,) | Covariate matrix with named columns |

Images can also be provided as a TSV manifest (columns: ID, file path) for streaming mode, supporting PNG and NIfTI files.

Genetics can instead be **PLINK `.bed/.bim/.fam`** files — pass the path *prefix* to `--genetics` (the default when neither `--kinship` nor `--r2` is set). This is the rank-B path: genotypes are streamed from disk in bit-packed chunks rather than materializing a full kinship or genotype matrix, so it scales to UK Biobank-sized cohorts.

### 2. Train

```bash
# Default: rank-B heritability streamed from PLINK genotypes (.bed/.bim/.fam prefix)
python3 train_hvae.py \
    --images data/images/T1_x_0.5.hdf5 \
    --genetics data/genetics/plinks/impSNPs_unrel_EUR_array \
    --covariates data/covariates/PC1_40_Age_Sex_ICV.ukb.hdf5 \
    --outdir out/my_run \
    --zdim 64 --h-weight 1.0 \
    --hweights aux/uniform.64.weights \
    --residualize-covariates aux/PC1_40_Age_Sex.covariates

# Kinship mode (precomputed GRM, Haseman-Elston method-of-moments)
python3 train_hvae.py \
    --images data/images/T1_x_0.5.hdf5 \
    --genetics data/genetics/kinship.hdf5 --kinship \
    --covariates data/covariates/PC1_40_Age_Sex_ICV.ukb.hdf5 \
    --outdir out/my_run_kinship \
    --zdim 64 --h-weight 1.0 \
    --hweights aux/uniform.64.weights \
    --residualize-covariates aux/ICV.covariates

# Heritability-spectrum objective (rotation-invariant; PLINK path only)
python3 train_hvae.py \
    --images data/images/T1_x_0.5.hdf5 \
    --genetics data/genetics/plinks/impSNPs_unrel_EUR_array \
    --covariates data/covariates/PC1_40_Age_Sex_ICV.ukb.hdf5 \
    --outdir out/my_run_spectrum \
    --zdim 64 --linear-heritability --spectrum-dims 8 \
    --h-weight 1.0 --hweights aux/uniform.8.weights --zs-floor 1e-8 \
    --residualize-covariates aux/PC1_40_Age_Sex.covariates

# External-trait genetic correlation via differentiable LDSC (PLINK path only)
python3 train_hvae.py \
    --images data/images/spirogram_flow_volume.hdf5 \
    --genetics data/genetics/plinks/impSNPs_unrel_EUR_array \
    --covariates data/covariates/PC1_40_Age_Sex_ICV.ukb.hdf5 \
    --outdir out/my_run_rg --model vae1d --zdim 16 \
    --h-weight 10.0 --hweights aux/first2.16.weights \
    --rg-ldsc-sumstats   data/ldsc/asthma.sumstats.gz \
    --rg-ldsc-ref-ld-chr data/ldsc/1000G_EUR_Phase3_array/ \
    --rg-ldsc-w-ld-chr   data/ldsc/1000G_EUR_Phase3_array/ \
    --residualize-covariates aux/PC1_40_Age_Sex.covariates
```

### 3. Plot heritability curves

```bash
# Single experiment
python3 out/plot_heritability.py single out/my_run

# Compare control (h-weight=0) vs experiment
python3 out/plot_heritability.py compare out/control out/my_run --epoch 200
```

### 4. Extract latents

```bash
python3 eval_latents.py --outdir out/my_run --epoch 100
```

## Key concepts

### Heritability estimation modes

| Mode | Flag | Genetics input | Loss | Display metric |
|------|------|---------------|------|---------------|
| Rank-B method of moments | (default) | PLINK `.bed/.bim/.fam` prefix | Streamed Haseman-Elston h² | h² |
| Method of moments | `--kinship` | Kinship matrix (n, n) HDF5 | Haseman-Elston h² | h² |
| Variance explained | `--r2` | Genotype matrix (n, m) HDF5 | Taylor-expanded R² | Exact OLS R² |
| In-cohort genetic correlation | `--genetic-correlation <file>` | any of the above + target phenotype HDF5 | Per-latent SCORE-overlap genetic covariance with target | Genetic correlation |

### Heritability-spectrum objective (`--linear-heritability`)

Instead of maximizing each latent's marginal h² independently, this maximizes the rotation-invariant **heritability spectrum** — the generalized eigenvalues of `G v = λ P v`, where `G` is the genetic-covariance matrix and `P` the phenotypic covariance of the latents. PLINK rank-B path only.

- `--spectrum-weight` / `--marginal-weight` — blend the spectrum objective with the per-dimension h² objective.
- `--spectrum-dims K` — restrict the spectrum to the first `K` latent dims (leaving the rest free for reconstruction).
- `--spectrum-ridge` — ridge added to `P` before whitening; `--spectrum-clamp` — relu the eigenvalues before the weighted sum (differentiable nearest-PSD).
- `--hweights` is reinterpreted as per-**rank** spectrum weights.
- `--zs-floor` floors the encoder posterior std so `log(zs)` stays finite (avoids a Cholesky NaN at low `--beta`).

### External-trait genetic correlation (`--rg-ldsc-sumstats`)

Adds a differentiable LDSC loss that maximizes the **genetic covariance** between each latent and an external trait's munged GWAS sumstats (displayed as genetic correlation ρ̂). PLINK rank-B path only; mutually exclusive with `--kinship`, `--r2`, `--genetic-correlation`, and `--linear-heritability`.

- `--rg-ldsc-sumstats <file>` — munged `.sumstats.gz` of the external trait.
- `--rg-ldsc-ref-ld-chr` / `--rg-ldsc-w-ld-chr` — per-chromosome LDSC reference and regression-weight LD-score prefixes.
- `--rg-ldsc-intercept-hsq` / `--rg-ldsc-intercept-gencov` — fix the LDSC intercepts (default: free, absorbing residual stratification / sample overlap).
- `--rg-ldsc-chroms` — restrict to a subset of chromosomes (e.g. `1-22`, `1,2,3`).
- `--hweights` selects which latent dims are pressured toward the external trait.

### Covariate pathways

Two independent pathways, controlled by text files listing covariate column names (one per line):

- **`--decode-covariates <file>`** — Covariates concatenated to z before decoding. Helps the decoder explain non-genetic image variation (e.g., age, sex, population structure).
- **`--residualize-covariates <file>`** — Covariates projected out before heritability estimation. Prevents confounders from inflating the heritability signal. Required in genotype mode.

Both can be active simultaneously with different covariate sets.

### Split-variants validation

```bash
python3 train_hvae.py \
    --genetics data/genetics/exome \
    --split-variants \
    ...
```

When `--split-variants` is set, `--genetics` is treated as a prefix. The training script loads `{prefix}.even.hdf5` and `{prefix}.odd.hdf5`. Even-chromosome variants drive the heritability loss during backpropagation; odd-chromosome variants are used only for display, providing an independent validation axis for whether the learned heritability generalizes across the genome.

### Model architectures

Select with `--model <name>`:

| Name | Input shape | Use case |
|------|------------|----------|
| `vae2d` (default) | (n, c, h, w) | 2D images (e.g., brain MRI slices) |
| `vae3d` | (n, c, d, h, w) | 3D volumes (e.g., full brain MRI) |
| `vae1d` | (n, c, l) | 1D time series |

### Training loop

Each epoch has two phases:

1. **Encode all** — Run the full dataset through the encoder in eval mode to collect latent statistics (means and stds).
2. **Mini-batch backprop** — For each batch, re-encode, sample from the posterior, update the full latent matrix Z, and backprop through a composite loss that combines reconstruction (MSE + KL), heritability, and optional regularization terms (latent correlation penalty, skew/kurtosis penalty).

## Output structure

Each run writes to `--outdir`:

```
weights/            Checkpoints (weights.NNNNN.pt)
plots/              Diagnostic plots
log.txt             Training log (parseable by out/plot_heritability.py)
vae.cfg.p           Pickled VAE constructor kwargs
train_ids.npy       Training split sample IDs
val_ids.npy         Validation split sample IDs
```

## CLI reference

Run `python3 train_hvae.py --help` for the full list of flags. Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--images` | (required) | Image HDF5 or TSV manifest |
| `--genetics` | (required) | Genetics HDF5 (or prefix with `--split-variants`) |
| `--covariates` | None | Covariates HDF5 |
| `--model` | `vae2d` | Model architecture |
| `--zdim` | 64 | Latent dimensionality |
| `--h-weight` | 1.0 | Heritability loss weight |
| `--beta` | 1.0 | KL divergence weight |
| `--kinship` | off | Use kinship matrix instead of genotypes |
| `--split-variants` | off | Split even/odd chromosome validation |
| `--genetic-correlation` | None | HDF5 of target phenotype; switches loss to in-cohort per-latent genetic correlation |
| `--r2` | off | Treat `--genetics` as a genotype-matrix HDF5; Taylor-expanded R² |
| `--linear-heritability` | off | Maximize the heritability spectrum instead of per-dim h² (PLINK path) |
| `--spectrum-dims` | 0 | Restrict the spectrum to the first K latent dims (0 = all) |
| `--spectrum-weight` | 1.0 | Weight on the spectrum objective |
| `--marginal-weight` | 0.0 | Weight on the per-dim h² objective (blended with the spectrum) |
| `--spectrum-ridge` | 1e-4 | Ridge added to P before whitening |
| `--spectrum-clamp` | off | relu the spectrum eigenvalues before the weighted sum |
| `--zs-floor` | 0.0 | Floor on the encoder posterior std (e.g. `1e-8`) |
| `--rg-ldsc-sumstats` | None | External trait `.sumstats.gz`; adds a differentiable LDSC genetic-covariance loss (also `--rg-ldsc-ref-ld-chr`, `--rg-ldsc-w-ld-chr`, `--rg-ldsc-intercept-hsq`, `--rg-ldsc-intercept-gencov`, `--rg-ldsc-chroms`) |
| `--train-frac` | 0.8 | Train/val split ratio |
| `--hweights` | None | Per-latent heritability weight file |
| `--mse-weight` | 1.0 | Reconstruction loss weight |
| `--corr-weight` | 0.0 | Latent correlation penalty weight |
| `--sk-weight` | 0.0 | Skew/kurtosis penalty weight |
| `--decode-covariates` | None | Covariate names for decode conditioning |
| `--residualize-covariates` | None | Covariate names for heritability residualization |
| `--resume` | None | Output directory to resume from |
| `--gradient-checkpoint` | off | Trade compute for memory |
| `--filts` | (model default) | Number of conv filters |
| `--vae-lr` | 1e-4 | Learning rate |
| `--bs` | 64 | Batch size |
| `--epochs` | 1001 | Total epochs |
| `--epoch-cb` | — | Checkpoint interval |
