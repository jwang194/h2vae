# h2vae

Maximizing the genetic signal in a VAE latent space.

h2vae trains a variational autoencoder whose latent dimensions are regularized to be genetically heritable. The training loss combines image reconstruction (MSE + KL) with a differentiable heritability estimator, encouraging the model to learn latent representations that capture genetically driven variation in imaging data.

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

### 2. Train

```bash
# Kinship mode (method-of-moments heritability)
python3 train_hvae.py \
    --images data/images/T1_x_0.5.hdf5 \
    --genetics data/genetics/kinship.hdf5 \
    --covariates data/covariates/PC1_40_Age_Sex_ICV.ukb.hdf5 \
    --outdir out/my_run \
    --zdim 64 --h-weight 1.0 --kinship \
    --hweights aux/uniform.64.weights \
    --residualize-covariates aux/ICV.covariates

# Genotype mode (Taylor-expanded variance explained)
python3 train_hvae.py \
    --images data/images/T1_x_0.5.hdf5 \
    --genetics data/genetics/genotypes.hdf5 \
    --covariates data/covariates/PC1_40_Age_Sex_ICV.ukb.hdf5 \
    --outdir out/my_run_taylor \
    --zdim 128 --h-weight 0.05 \
    --hweights aux/uniform.128.weights \
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
| Method of moments | `--kinship` | Kinship matrix (n, n) | Haseman-Elston h² | h² |
| Variance explained | (default) | Genotype matrix (n, m) | Taylor-expanded R² | Exact OLS R² |

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
2. **Mini-batch backprop** — For each batch, re-encode, sample from the posterior, update the full latent matrix Z, and backprop through a composite loss that combines reconstruction, heritability, and optional regularization terms (correlation, moment matching).

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
| `--train-frac` | 0.8 | Train/val split ratio |
| `--hweights` | None | Per-latent heritability weight file |
| `--decode-covariates` | None | Covariate names for decode conditioning |
| `--residualize-covariates` | None | Covariate names for heritability residualization |
| `--resume` | None | Output directory to resume from |
| `--gradient-checkpoint` | off | Trade compute for memory |
| `--vae-lr` | 1e-4 | Learning rate |
| `--bs` | 64 | Batch size |
| `--epochs` | 1001 | Total epochs |
