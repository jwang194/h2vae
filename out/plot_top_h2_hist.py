"""Plot the distribution of the highest-heritability latent for an HVAE run.

Picks the latent dimension with maximum h_val at the epoch matching the saved
``Zm_{train,val}.NNNNN.txt`` files in the experiment directory and renders a
histogram of its values across train + val samples.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "figure.figsize": (16, 9),
    "figure.dpi": 300,
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

HUE_PALETTE = ["goldenrod", "darkviolet", "darkturquoise"]

_EPOCH_RE = re.compile(
    r"epoch (?P<epoch>\d+) - mse_train: [\d.]+ - mse_val: [\d.]+ "
    r"- h_train: (?P<train>[\d.,\s-]+) - h_val: (?P<val>[\d.,\s-]+)"
)


def parse_h_val(log_path: str) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    with open(log_path) as f:
        for line in f:
            m = _EPOCH_RE.search(line)
            if m is None:
                continue
            out[int(m.group("epoch"))] = np.fromstring(m.group("val"), sep=",")
    return out


def find_zm_epoch(exp_dir: str) -> int:
    epochs = []
    for name in os.listdir(exp_dir):
        m = re.match(r"Zm_val\.(\d+)\.txt$", name)
        if m:
            epochs.append(int(m.group(1)))
    if not epochs:
        sys.exit(f"no Zm_val.NNNNN.txt found in {exp_dir}")
    return max(epochs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    epoch = find_zm_epoch(args.exp_dir)
    log_path = os.path.join(args.exp_dir, "log.txt")
    h_val_by_epoch = parse_h_val(log_path)
    if epoch not in h_val_by_epoch:
        sys.exit(f"epoch {epoch} not present in {log_path}")
    h = h_val_by_epoch[epoch]
    top = int(np.argmax(h))
    print(f"epoch {epoch}: top latent = z{top} with h_val = {h[top]:.3f}")

    z_train = np.loadtxt(os.path.join(args.exp_dir, f"Zm_train.{epoch:05d}.txt"))
    z_val = np.loadtxt(os.path.join(args.exp_dir, f"Zm_val.{epoch:05d}.txt"))
    x_train = z_train[:, top]
    x_val = z_val[:, top]
    x_all = np.concatenate([x_train, x_val])

    plots_dir = os.path.join(args.exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = args.out or os.path.join(
        plots_dir, f"top_h2_hist.epoch{epoch:05d}.z{top}.png"
    )

    bins = np.linspace(x_all.min(), x_all.max(), 60)
    fig, ax = plt.subplots()
    ax.hist(x_train, bins=bins, color=HUE_PALETTE[0], alpha=0.6,
            label=f"Train (n={len(x_train)})", edgecolor="white", linewidth=0.4)
    ax.hist(x_val, bins=bins, color=HUE_PALETTE[1], alpha=0.6,
            label=f"Val (n={len(x_val)})", edgecolor="white", linewidth=0.4)
    ax.set_xlabel(f"z{top}")
    ax.set_ylabel("Count")
    ax.set_title(
        f"{os.path.basename(os.path.normpath(args.exp_dir))} — "
        f"epoch {epoch}, z{top} (h²_val = {h[top]:.3f})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
