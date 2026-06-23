"""Extract posterior means from a trained VAE checkpoint.

Reads the VAE config, reconstructs the model, loads a checkpoint, then
encodes all training and validation samples. Outputs two .npy files
(one per split) containing the posterior mean for each sample.

Usage:
    python3 eval_latents.py --outdir out/my_run --images data/images/T1_x_0.5.hdf5
    python3 eval_latents.py --outdir out/my_run --images data/images/T1_x_0.5.hdf5 --epoch 500
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from h2vae.models import get_model_class
from h2vae.data import ImageDataset, make_streaming_dataset


def find_latest_checkpoint(outdir: str) -> tuple[str, int]:
    """Find the latest weights file in outdir/weights/."""
    wdir = os.path.join(outdir, "weights")
    weight_files = [f for f in os.listdir(wdir) if f.startswith("weights.") and f.endswith(".pt")]
    if not weight_files:
        raise FileNotFoundError(f"No checkpoints found in {wdir}")
    epochs = [int(f.split(".")[1]) for f in weight_files]
    latest = max(epochs)
    return os.path.join(wdir, f"weights.{latest:05d}.pt"), latest


def encode_split(
    vae: torch.nn.Module,
    loader: DataLoader,
    zdim: int,
    device: torch.device,
) -> np.ndarray:
    """Encode a full split and return posterior means as a numpy array."""
    vae.eval()
    n = len(loader.dataset)
    Zm = torch.zeros(n, zdim)

    with torch.no_grad():
        for data in loader:
            y = data[0].to(device)
            idxs = data[-1]
            zm, _ = vae.encode(y)
            Zm[idxs] = zm.cpu()

    return Zm.numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract VAE posterior means")
    parser.add_argument("--outdir", type=str, required=True, help="training output directory")
    parser.add_argument("--images", type=str, required=True, help="images HDF5 or TSV (same as training)")
    parser.add_argument("--epoch", type=int, default=None, help="checkpoint epoch (default: latest)")
    parser.add_argument("--bs", type=int, default=64, help="batch size")
    parser.add_argument("--which-cuda", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.which_cuda}" if torch.cuda.is_available() else "cpu")

    # --- Load VAE config and build model ---
    cfg_path = os.path.join(args.outdir, "vae.cfg.p")
    with open(cfg_path, "rb") as f:
        vae_cfg = pickle.load(f)
    print(f"VAE config: {vae_cfg}")

    model_name = vae_cfg.pop("_model_class", "vae2d")
    ModelClass = get_model_class(model_name)
    constructor_cfg = {k: v for k, v in vae_cfg.items() if not k.startswith("_")}
    vae = ModelClass(**constructor_cfg).to(device)

    # --- Load checkpoint ---
    if args.epoch is not None:
        ckpt_path = os.path.join(args.outdir, "weights", f"weights.{args.epoch:05d}.pt")
        epoch = args.epoch
    else:
        ckpt_path, epoch = find_latest_checkpoint(args.outdir)
    print(f"Loading checkpoint: {ckpt_path}")
    vae.load_state_dict(torch.load(ckpt_path, map_location=device))

    # --- Load split IDs ---
    train_ids = np.load(os.path.join(args.outdir, "train_ids.npy"))
    val_ids = np.load(os.path.join(args.outdir, "val_ids.npy"))

    # --- Load images and build datasets ---
    streaming = args.images.endswith(".tsv")
    if streaming:
        import pandas as pd
        df = pd.read_csv(args.images, sep="\t", header=None, dtype={0: np.int64, 1: str})
        id_to_path = dict(zip(df[0].values, df[1].tolist()))

        train_paths = [id_to_path[i] for i in train_ids]
        val_paths = [id_to_path[i] for i in val_ids]
        train_id_tensor = torch.tensor(train_ids.astype(np.float32))
        val_id_tensor = torch.tensor(val_ids.astype(np.float32))

        streaming_kwargs = {}
        if ModelClass.data_format == "nifti":
            import inspect
            model_default = inspect.signature(ModelClass.__init__).parameters["img_size"].default
            streaming_kwargs["target_size"] = model_default

        train_dataset = make_streaming_dataset(
            train_paths, train_id_tensor, ModelClass.data_format, **streaming_kwargs,
        )
        val_dataset = make_streaming_dataset(
            val_paths, val_id_tensor, ModelClass.data_format, **streaming_kwargs,
        )

        nw = 16 if ModelClass.data_format == "nifti" else 8
        loader_kwargs = dict(num_workers=nw, prefetch_factor=2,
                             pin_memory=True, persistent_workers=True)
    else:
        import h5py
        with h5py.File(args.images, "r") as f:
            image_ids = np.array(f["ids"])
            images_raw = np.array(f["data"])

        id_to_idx = {v: i for i, v in enumerate(image_ids)}
        train_idx = np.array([id_to_idx[i] for i in train_ids])
        val_idx = np.array([id_to_idx[i] for i in val_ids])

        def _to_tensor(a):  # match h2vae/data.py: handle 2D/3D(1D model)/4D(2D model)
            a = a.astype(np.float32)
            if a.ndim == 4:        # (n,h,w,c) -> (n,c,h,w)  [vae2d]
                a = a.transpose((0, 3, 1, 2))
            elif a.ndim == 2:      # (n,L) -> (n,1,L)         [vae1d]
                a = a[:, None, :]
            # ndim==3 (n,c,L): already channels-leading [vae1d]
            return torch.tensor(a)
        train_images = _to_tensor(images_raw[train_idx])
        val_images = _to_tensor(images_raw[val_idx])

        train_dataset = ImageDataset(train_images, torch.tensor(train_ids.astype(np.float32)))
        val_dataset = ImageDataset(val_images, torch.tensor(val_ids.astype(np.float32)))
        loader_kwargs = {}

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, **loader_kwargs)

    # --- Encode ---
    zdim = vae_cfg["zdim"]
    print(f"Encoding train split ({len(train_dataset)} samples)...")
    Zm_train = encode_split(vae, train_loader, zdim, device)
    print(f"Encoding val split ({len(val_dataset)} samples)...")
    Zm_val = encode_split(vae, val_loader, zdim, device)

    # --- Save ---
    train_out = os.path.join(args.outdir, f"Zm_train.{epoch:05d}.txt")
    val_out = os.path.join(args.outdir, f"Zm_val.{epoch:05d}.txt")
    np.savetxt(train_out, Zm_train, delimiter="\t")
    np.savetxt(val_out, Zm_val, delimiter="\t")
    print(f"Saved: {train_out} {Zm_train.shape}")
    print(f"Saved: {val_out} {Zm_val.shape}")


if __name__ == "__main__":
    main()
