"""Dataset and HDF5 data loading with ID-based sample merging across files.

Each data type lives in its own HDF5 file with its own ``ids`` array.
The loader performs an inner join across all provided files so that
downstream code sees only samples present in every source.

Expected file structures:

Images HDF5 (required)::

    data    (n, h, w, c) numeric
    ids     (n,)

Covariates HDF5 (optional)::

    data              (n, p) float
    ids               (n,)
    covariate_names   (p,) string

Genetics HDF5 (required)::

    kinship       (n, n) float   [optional — use with --kinship]
    kinship_ids   (n,)           [optional]
    genotypes     (n, m) float   [optional — default]
    genotype_ids  (n,)           [optional]
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import h5py


class ImageDataset(Dataset):
    """Torch dataset wrapping image tensors with sample indices.

    Args:
        images: Image tensor of shape (n, c, h, w).
        ids: Sample ID tensor of shape (n,).
    """

    def __init__(self, images: Tensor, ids: Tensor):
        self.images = images
        self.ids = ids

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        return self.images[index], self.ids[index], index


class ImageFileDataset(Dataset):
    """Torch dataset that streams PNG images from disk.

    Each call to ``__getitem__`` reads a single image file, converts it to a
    float32 tensor in (c, h, w) layout with values in [0, 1].

    Args:
        paths: List of file paths to PNG images.
        ids: Sample ID tensor of shape (n,).
    """

    def __init__(self, paths: list[str], ids: Tensor):
        self.paths = paths
        self.ids = ids

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        from PIL import Image

        img = Image.open(self.paths[index])
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[:, :, None]
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        return tensor, self.ids[index], index


class NiftiFileDataset(Dataset):
    """Torch dataset that streams NIFTI volumes from disk.

    Each call to ``__getitem__`` loads a ``.nii``, ``.nii.gz``, or
    ``.nii.zst`` file via nibabel, z-score normalises non-zero voxels, and
    zero-pads to a cubic ``target_size``.  Returns a ``(1, D, H, W)``
    float32 tensor.

    Args:
        paths: List of file paths to NIFTI files.
        ids: Sample ID tensor of shape ``(n,)``.
        target_size: Cubic side length to pad to.  If *None*, returns the
            volume at its native resolution (all volumes must then share
            the same shape).
    """

    def __init__(self, paths: list[str], ids: Tensor,
                 target_size: int | None = None):
        self.paths = paths
        self.ids = ids
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        import nibabel as nib

        vol = nib.load(self.paths[index]).get_fdata().astype(np.float32)

        # Z-score normalise non-zero voxels
        mask = vol != 0
        if mask.any():
            vol[mask] = (vol[mask] - vol[mask].mean()) / (vol[mask].std() + 1e-8)

        # Pad to cubic target_size (centre-pad with zeros)
        if self.target_size is not None:
            t = self.target_size
            padded = np.zeros((t, t, t), dtype=np.float32)
            d, h, w = vol.shape
            d0 = (t - d) // 2
            h0 = (t - h) // 2
            w0 = (t - w) // 2
            padded[d0:d0 + d, h0:h0 + h, w0:w0 + w] = vol
            vol = padded

        tensor = torch.from_numpy(vol).unsqueeze(0)  # (1, D, H, W)
        return tensor, self.ids[index], index


class TimeSeriesFileDataset(Dataset):
    """Torch dataset that streams time-series data from disk.

    Supports ``.npy`` and ``.txt`` (whitespace-delimited) files.  Each file
    should contain a 1-D array (single channel) or a 2-D array of shape
    ``(channels, seq_length)``.

    Args:
        paths: List of file paths.
        ids: Sample ID tensor of shape ``(n,)``.
    """

    def __init__(self, paths: list[str], ids: Tensor):
        self.paths = paths
        self.ids = ids

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        p = self.paths[index]
        if p.endswith(".npy"):
            arr = np.load(p).astype(np.float32)
        else:
            arr = np.loadtxt(p, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]  # (seq_length,) -> (1, seq_length)
        tensor = torch.from_numpy(arr)
        return tensor, self.ids[index], index


def make_streaming_dataset(
    paths: list[str],
    ids: Tensor,
    data_format: str,
    **kwargs,
) -> Dataset:
    """Create the appropriate streaming dataset for *data_format*.

    Args:
        paths: Per-sample file paths.
        ids: Sample ID tensor.
        data_format: One of ``"image"``, ``"nifti"``, or ``"timeseries"``.
        **kwargs: Forwarded to the dataset constructor (e.g.
            ``target_size`` for :class:`NiftiFileDataset`).
    """
    if data_format == "image":
        return ImageFileDataset(paths, ids)
    elif data_format == "nifti":
        return NiftiFileDataset(paths, ids, **kwargs)
    elif data_format == "timeseries":
        return TimeSeriesFileDataset(paths, ids)
    else:
        raise ValueError(f"Unknown data_format: {data_format!r}")


def _intersect_ids(*id_arrays: np.ndarray) -> np.ndarray:
    """Return sorted array of IDs present in all input arrays."""
    common = set(id_arrays[0])
    for ids in id_arrays[1:]:
        common &= set(ids)
    return np.sort(np.array(list(common)))


def _reindex(ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    """Return indices into ``ids`` that select ``target_ids`` in order.

    Assumes ``target_ids`` is a subset of ``ids``.
    """
    id_to_idx = {v: i for i, v in enumerate(ids)}
    return np.array([id_to_idx[t] for t in target_ids])


def _split_ids(
    common_ids: np.ndarray,
    train_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle and split IDs into train/val."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(common_ids))
    n_train = int(len(common_ids) * train_frac)
    return common_ids[perm[:n_train]], common_ids[perm[n_train:]]


def _read_image_tsv(path: str) -> tuple[np.ndarray, list[str]]:
    """Read a TSV file with IDs in column 0 and image paths in column 1.

    Returns:
        ids: Array of integer sample IDs.
        paths: List of image file paths.
    """
    import pandas as pd

    df = pd.read_csv(path, sep="\t", header=None, dtype={0: np.int64, 1: str})
    return df[0].values, df[1].tolist()


def load_data(
    images_path: str,
    genetics_path: str,
    covariates_path: str | None = None,
    target_phenotype_path: str | None = None,
    train_frac: float = 0.8,
    seed: int = 0,
    train_ids: np.ndarray | None = None,
    val_ids: np.ndarray | None = None,
    required_covariates: list[str] | None = None,
) -> dict:
    """Load data from separate HDF5 files, inner-join on IDs, and split.

    If ``train_ids`` and ``val_ids`` are provided (e.g. from a resumed run),
    they are used directly instead of re-splitting. The inner join is still
    applied — any IDs in the provided split that are missing from the data
    files are dropped.

    Args:
        images_path: Path to images HDF5 (must contain data + ids) **or** a
            TSV file with IDs in the first column and PNG file paths in the
            second column.  When a TSV is provided, images are not loaded
            into memory; the returned dict contains ``image_paths`` lists
            instead of image tensors, suitable for use with
            :class:`ImageFileDataset`.
        genetics_path: Path to genetics HDF5 (kinship and/or genotypes).
        covariates_path: Optional path to covariates HDF5.
        target_phenotype_path: Optional path to a target-phenotype HDF5
            (keys ``data`` shape ``(n,)`` or ``(n, 1)`` and ``ids`` shape ``(n,)``).
            When provided, the phenotype is included in the inner join and
            returned under ``train`` / ``val`` as a ``(n_split, 1)`` float32 tensor.
        train_frac: Fraction of samples for training (rest go to val).
        seed: Random seed for the train/val shuffle.
        train_ids: Pre-specified training IDs (overrides train_frac/seed).
        val_ids: Pre-specified validation IDs (overrides train_frac/seed).
        required_covariates: Optional list of covariate names that must be
            non-NaN for a sample to be retained. After the ID inner-join,
            any sample with NaN in any of these columns is dropped. Lets
            the caller supply a master covariates HDF5 with NaNs for
            missing values and filter the cohort per-experiment based on
            which columns that experiment actually uses.

    Returns:
        Dict with structure::

            {
                "train": {"images": Tensor | None, "image_paths": list[str] | None,
                           "ids": Tensor, "covariates": Tensor | None,
                           "target_phenotype": Tensor | None},
                "val":   {"images": Tensor | None, "image_paths": list[str] | None,
                           "ids": Tensor, "covariates": Tensor | None,
                           "target_phenotype": Tensor | None},
                "genetics": {"kinship": Tensor | None, "genotypes": Tensor | None},
                "covariate_names": list[str] | None,
                "streaming": bool,
                "n_train": int,
                "train_ids_raw": np.ndarray,
                "val_ids_raw": np.ndarray,
            }

        Genetics tensors are reindexed to the merged ID order (train first,
        then val). Kinship is (n, n), genotypes is (n, m).
    """
    # --- Read images ---
    streaming = images_path.endswith(".tsv")
    if streaming:
        image_ids, image_paths_all = _read_image_tsv(images_path)
        id_to_path = dict(zip(image_ids, image_paths_all))
        images_raw = None
    else:
        with h5py.File(images_path, "r") as f:
            image_ids = np.array(f["ids"])
            images_raw = np.array(f["data"])
        id_to_path = None

    # --- Read genetics ---
    # Three accepted forms for ``genetics_path``:
    #   1. an HDF5 file containing ``kinship`` (and ``kinship_ids``);
    #   2. an HDF5 file containing ``genotypes`` (and ``genotype_ids``);
    #   3. a PLINK prefix — ``{path}.bed/.bim/.fam`` all present. In
    #      this case we only use the .fam for cohort IDs here; the BED
    #      is opened lazily by the rank-B heritability path in
    #      ``setup_heritability``.
    from h2vae.plink import is_plink_prefix, read_fam_ids
    has_kinship = False
    has_genotypes = False
    has_plink = False
    plink_prefix: str | None = None
    if is_plink_prefix(genetics_path):
        has_plink = True
        plink_prefix = genetics_path
        gen_ids = read_fam_ids(genetics_path)
    else:
        with h5py.File(genetics_path, "r") as f:
            if "kinship" in f:
                has_kinship = True
                kin_ids = np.array(f["kinship_ids"])
                kinship_raw = np.array(f["kinship"])
            if "genotypes" in f:
                has_genotypes = True
                gen_ids = np.array(f["genotype_ids"])
                genotypes_raw = np.array(f["genotypes"])

    if not (has_kinship or has_genotypes or has_plink):
        raise ValueError(
            f"No kinship, genotypes, or PLINK files found at {genetics_path}"
        )

    # --- Read covariates (optional) ---
    has_covariates = covariates_path is not None
    cov_names = None
    if has_covariates:
        with h5py.File(covariates_path, "r") as f:
            cov_ids = np.array(f["ids"])
            covariates_raw = np.array(f["data"])
            if "covariate_names" in f:
                cov_names = [
                    name.decode() if isinstance(name, bytes) else name
                    for name in f["covariate_names"][:]
                ]

    # --- Read target phenotype (optional) ---
    has_target_phenotype = target_phenotype_path is not None
    if has_target_phenotype:
        with h5py.File(target_phenotype_path, "r") as f:
            tp_ids = np.array(f["ids"])
            tp_raw = np.array(f["data"])
        if tp_raw.ndim == 1:
            tp_raw = tp_raw[:, None]

    # --- Collect ID arrays for inner join ---
    all_id_arrays = [image_ids]
    if has_kinship:
        all_id_arrays.append(kin_ids)
    if has_genotypes or has_plink:
        all_id_arrays.append(gen_ids)
    if has_covariates:
        all_id_arrays.append(cov_ids)
    if has_target_phenotype:
        all_id_arrays.append(tp_ids)

    common_ids = _intersect_ids(*all_id_arrays)
    n = len(common_ids)
    if n == 0:
        raise ValueError("Inner join produced zero samples — no IDs shared across all files")

    # --- Drop IDs with NaN in the target phenotype ---
    # The target phenotype drives the gc loss, so any NaN there is fatal;
    # silently dropping those samples is the only sensible behaviour.
    if has_target_phenotype:
        id_to_tp_row = {v: i for i, v in enumerate(tp_ids)}
        tp_rows = np.array([id_to_tp_row[i] for i in common_ids])
        tp_subset = tp_raw[tp_rows]
        keep_mask = ~np.isnan(tp_subset).any(axis=1)
        dropped = int((~keep_mask).sum())
        if dropped:
            print(
                f"load_data: dropped {dropped} samples with NaN in target phenotype "
                f"(kept {int(keep_mask.sum())} of {len(common_ids)})"
            )
            common_ids = common_ids[keep_mask]
            n = len(common_ids)
        if n == 0:
            raise ValueError(
                "All samples dropped — every joined sample has NaN in the target phenotype"
            )

    # --- Drop IDs with NaN in required covariate columns ---
    if required_covariates:
        if not has_covariates:
            raise ValueError("required_covariates specified but no covariates HDF5 provided")
        if cov_names is None:
            raise ValueError("required_covariates specified but covariates HDF5 has no covariate_names")
        missing = [name for name in required_covariates if name not in cov_names]
        if missing:
            raise ValueError(f"required_covariates not found in covariate_names: {missing}")
        col_idx = [cov_names.index(name) for name in required_covariates]
        id_to_cov_row = {v: i for i, v in enumerate(cov_ids)}
        cov_rows = np.array([id_to_cov_row[i] for i in common_ids])
        cov_subset = covariates_raw[cov_rows][:, col_idx]
        keep_mask = ~np.isnan(cov_subset).any(axis=1)
        dropped = int((~keep_mask).sum())
        if dropped:
            common_ids = common_ids[keep_mask]
            n = len(common_ids)
        if n == 0:
            raise ValueError(
                f"All samples dropped after NaN filter on required_covariates={required_covariates}"
            )

    # --- Split into train/val ---
    if train_ids is not None and val_ids is not None:
        # Use provided split, but intersect with available data
        common_set = set(common_ids)
        train_common = np.array([x for x in train_ids if x in common_set])
        val_common = np.array([x for x in val_ids if x in common_set])
    else:
        train_common, val_common = _split_ids(common_ids, train_frac, seed)

    n_train = len(train_common)
    all_common = np.concatenate([train_common, val_common])

    # --- Reindex images ---
    train_ids_tensor = torch.tensor(train_common.astype(np.float32))
    val_ids_tensor = torch.tensor(val_common.astype(np.float32))

    train_images = None
    val_images = None
    train_image_paths = None
    val_image_paths = None

    if streaming:
        train_image_paths = [id_to_path[i] for i in train_common]
        val_image_paths = [id_to_path[i] for i in val_common]
    else:
        img_idx_train = _reindex(image_ids, train_common)
        img_idx_val = _reindex(image_ids, val_common)
        if images_raw.ndim == 4:
            # (n, h, w, c) -> (n, c, h, w) float32
            arr_train = images_raw[img_idx_train].astype(np.float32).transpose((0, 3, 1, 2))
            arr_val = images_raw[img_idx_val].astype(np.float32).transpose((0, 3, 1, 2))
        elif images_raw.ndim == 3:
            # (n, c, L) — 1D, channels already leading
            arr_train = images_raw[img_idx_train].astype(np.float32)
            arr_val = images_raw[img_idx_val].astype(np.float32)
        elif images_raw.ndim == 2:
            # (n, L) -> (n, 1, L)
            arr_train = images_raw[img_idx_train].astype(np.float32)[:, None, :]
            arr_val = images_raw[img_idx_val].astype(np.float32)[:, None, :]
        else:
            raise ValueError(f"Unsupported images_raw.ndim={images_raw.ndim}")
        train_images = torch.tensor(arr_train)
        val_images = torch.tensor(arr_val)

    # --- Reindex covariates ---
    train_covariates = None
    val_covariates = None
    if has_covariates:
        cov_idx_train = _reindex(cov_ids, train_common)
        cov_idx_val = _reindex(cov_ids, val_common)
        train_covariates = torch.tensor(covariates_raw[cov_idx_train].astype(np.float32))
        val_covariates = torch.tensor(covariates_raw[cov_idx_val].astype(np.float32))

    # --- Reindex target phenotype ---
    train_target = None
    val_target = None
    if has_target_phenotype:
        tp_idx_train = _reindex(tp_ids, train_common)
        tp_idx_val = _reindex(tp_ids, val_common)
        train_target = torch.tensor(tp_raw[tp_idx_train].astype(np.float32))
        val_target = torch.tensor(tp_raw[tp_idx_val].astype(np.float32))

    # --- Reindex genetics (full [train; val] order) ---
    # ``plink_row_idx`` indexes into the BED file's full sample list and
    # is consumed by the rank-B heritability path (which mmaps the BED
    # and decodes on demand).
    genetics: dict[str, Tensor | str | np.ndarray | None] = {
        "kinship": None, "genotypes": None,
        "plink_prefix": plink_prefix, "plink_row_idx": None,
    }
    if has_kinship:
        kin_idx = _reindex(kin_ids, all_common)
        genetics["kinship"] = torch.tensor(
            kinship_raw[np.ix_(kin_idx, kin_idx)].astype(np.float32)
        )
    if has_genotypes:
        gen_idx = _reindex(gen_ids, all_common)
        genetics["genotypes"] = torch.tensor(
            genotypes_raw[gen_idx].astype(np.float32)
        )
    if has_plink:
        genetics["plink_row_idx"] = _reindex(gen_ids, all_common)

    return {
        "train": {
            "images": train_images,
            "image_paths": train_image_paths,
            "ids": train_ids_tensor,
            "covariates": train_covariates,
            "target_phenotype": train_target,
        },
        "val": {
            "images": val_images,
            "image_paths": val_image_paths,
            "ids": val_ids_tensor,
            "covariates": val_covariates,
            "target_phenotype": val_target,
        },
        "genetics": genetics,
        "covariate_names": cov_names,
        "streaming": streaming,
        "n_train": n_train,
        "train_ids_raw": train_common,
        "val_ids_raw": val_common,
    }


def load_genetics_reindexed(
    path: str,
    id_order: np.ndarray,
) -> dict[str, Tensor | str | np.ndarray | None]:
    """Load a genetics source and reindex to match a given ID order.

    Accepts an HDF5 file (kinship and/or genotypes) **or** a PLINK
    prefix (``<path>.bed`` etc. exist).  Used to load a second genetics
    source — e.g. odd-chromosome variants — after ``load_data()`` has
    established the sample order via inner join.

    Args:
        path: HDF5 file path or PLINK prefix.
        id_order: Array of sample IDs in the desired order (train first,
            then val). Must be a subset of the IDs in the file.

    Returns:
        Dict with keys ``"kinship"``, ``"genotypes"``, ``"plink_prefix"``,
        ``"plink_row_idx"`` — each populated only as appropriate to the
        source format.
    """
    from h2vae.plink import is_plink_prefix, read_fam_ids
    genetics: dict[str, Tensor | str | np.ndarray | None] = {
        "kinship": None, "genotypes": None,
        "plink_prefix": None, "plink_row_idx": None,
    }
    if is_plink_prefix(path):
        gen_ids = read_fam_ids(path)
        genetics["plink_prefix"] = path
        genetics["plink_row_idx"] = _reindex(gen_ids, id_order)
        return genetics

    with h5py.File(path, "r") as f:
        if "kinship" in f:
            kin_ids = np.array(f["kinship_ids"])
            kinship_raw = np.array(f["kinship"])
            kin_idx = _reindex(kin_ids, id_order)
            genetics["kinship"] = torch.tensor(
                kinship_raw[np.ix_(kin_idx, kin_idx)].astype(np.float32)
            )
        if "genotypes" in f:
            gen_ids = np.array(f["genotype_ids"])
            genotypes_raw = np.array(f["genotypes"])
            gen_idx = _reindex(gen_ids, id_order)
            genetics["genotypes"] = torch.tensor(
                genotypes_raw[gen_idx].astype(np.float32)
            )
    return genetics
