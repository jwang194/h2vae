"""Plot heritability curves over training epochs from an HVAE log file.

Subcommands:

    python plot_heritability.py single <log_dir> [--out <output.png>]
        Single-experiment plot with mean/max h² curves.

    python plot_heritability.py compare <control_dir> <experiment_dir> [--out <output.png>]
        Side-by-side comparison: val h² curves (left) and violin plot at
        the peak-heritability epoch of the experiment (right).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ranksums

# ---------------------------------------------------------------------------
# Nature-style defaults
# ---------------------------------------------------------------------------
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

HUE_PALETTE = ["goldenrod", "darkviolet", "darkturquoise", "#E05263"]

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

_EPOCH_RE_SPLIT = re.compile(
    r"epoch (\d+) - mse_train: [\d.]+ - mse_val: [\d.]+"
    r" - h_train_even: [\d.,\s-]+ - h_train_odd: ([\d.,\s-]+)"
    r" - h_val_even: [\d.,\s-]+ - h_val_odd: ([\d.,\s-]+)"
)

_EPOCH_RE = re.compile(
    r"epoch (\d+) - mse_train: [\d.]+ - mse_val: [\d.]+ "
    r"- h_train: ([\d.,\s-]+) - h_val: ([\d.,\s-]+)"
)


def parse_log(path: str) -> dict:
    """Parse a log.txt file and return per-epoch heritability arrays.

    For runs trained with ``--split-variants``, the log carries four h² streams
    (``h_train_even``, ``h_train_odd``, ``h_val_even``, ``h_val_odd``); this
    parser keeps only the ``_odd`` (held-out-chromosome) streams so downstream
    plotting can treat split and non-split runs uniformly.

    Returns:
        dict with keys: epochs, h_train, h_val
        where h_train and h_val are arrays of shape (n_epochs, zdim).
    """
    # Use a dict keyed by epoch to deduplicate (last occurrence wins,
    # handling resumed training runs that re-log earlier epochs).
    by_epoch: dict[int, tuple[list[float], list[float]]] = {}

    with open(path) as f:
        for line in f:
            m = _EPOCH_RE_SPLIT.search(line) or _EPOCH_RE.search(line)
            if m is None:
                continue
            epoch = int(m.group(1))
            h_train = [float(x) for x in m.group(2).split(",")]
            h_val = [float(x) for x in m.group(3).split(",")]
            by_epoch[epoch] = (h_train, h_val)

    sorted_epochs = sorted(by_epoch)
    return {
        "epochs": np.array(sorted_epochs),
        "h_train": np.array([by_epoch[e][0] for e in sorted_epochs]),
        "h_val": np.array([by_epoch[e][1] for e in sorted_epochs]),
    }


# ---------------------------------------------------------------------------
# Line plot helpers
# ---------------------------------------------------------------------------

def _plot_lines(ax, epochs, h_matrix, color, label):
    """Plot mean (solid, with SD shading) and max (dashed) for one condition."""
    mean = h_matrix.mean(axis=1)
    std = h_matrix.std(axis=1)
    mx = h_matrix.max(axis=1)

    ax.plot(epochs, mean, color=color, linestyle="-", label=f"Mean $h^2$ ({label})")
    ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)
    ax.plot(epochs, mx, color=color, linestyle="--", label=f"Max $h^2$ ({label})")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heritability(data: dict, out_path: str) -> None:
    epochs = data["epochs"]
    fig, ax = plt.subplots()

    _plot_lines(ax, epochs, data["h_train"], HUE_PALETTE[0], "train")
    _plot_lines(ax, epochs, data["h_val"], HUE_PALETTE[1], "val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("$h^2$")
    ax.set_title("Latent heritability over training")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved to {out_path}")


def _add_significance_bracket(ax, x0, x1, y0, y1, pval):
    """Draw a bracket descending from a bar to y0 (left) and y1 (right)."""
    bar_y = max(y0, y1) + 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x0, x0, x1, x1], [y0, bar_y, bar_y, y1],
            color="black", linewidth=1.2)
    ax.text((x0 + x1) / 2, bar_y, f"$p$ = {pval:.2e}",
            ha="center", va="bottom", fontsize=14)


def plot_compare(ctrl_data: dict, exp_data: dict, out_path: str, epoch: int | None = None) -> None:
    """Side-by-side comparison of control (h_weight=0) vs experiment.

    Left panel: val-set mean h² (with SD shading) and max h² for both runs.
    Right panel: seaborn violin + boxplot of per-latent h² at the experiment's
    peak epoch, with Wilcoxon rank-sum p-value bracket.
    """
    fig, (ax_ts, ax_vln) = plt.subplots(1, 2, figsize=(20, 9))

    c_ctrl, c_exp = HUE_PALETTE[0], HUE_PALETTE[1]

    # --- Left panel: time-series ---
    _plot_lines(ax_ts, ctrl_data["epochs"], ctrl_data["h_val"], c_ctrl, "Control")
    _plot_lines(ax_ts, exp_data["epochs"], exp_data["h_val"], c_exp, "Experiment")

    ax_ts.set_xlabel("Epoch")
    ax_ts.set_ylabel("$h^2$ (val)")
    ax_ts.set_title("Validation heritability over training")
    ax_ts.set_ylim(0, 1.0)
    ax_ts.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax_ts.legend(fontsize=14)

    # --- Right panel: violin at target epoch ---
    if epoch is not None:
        peak_idx = int(np.argmin(np.abs(exp_data["epochs"] - epoch)))
        peak_epoch = exp_data["epochs"][peak_idx]
    else:
        exp_mean_val = exp_data["h_val"].mean(axis=1)
        peak_idx = int(np.argmax(exp_mean_val))
        peak_epoch = exp_data["epochs"][peak_idx]

    ctrl_idx = int(np.argmin(np.abs(ctrl_data["epochs"] - peak_epoch)))

    h_ctrl = ctrl_data["h_val"][ctrl_idx]
    h_exp = exp_data["h_val"][peak_idx]

    df = pd.DataFrame({
        "$h^2$ (val)": np.concatenate([h_ctrl, h_exp]),
        "Condition": ["Control"] * len(h_ctrl) + ["Experiment"] * len(h_exp),
    })

    sns.violinplot(
        data=df, x="Condition", y="$h^2$ (val)", hue="Condition",
        palette=[c_ctrl, c_exp], inner="box", cut=0, legend=False, ax=ax_vln,
    )

    stat, pval = ranksums(h_exp, h_ctrl, alternative="greater")

    y_range = ax_vln.get_ylim()[1] - ax_vln.get_ylim()[0]
    gap = 0.03 * y_range
    _add_significance_bracket(ax_vln, 0, 1, h_ctrl.max() + gap, h_exp.max() + gap, pval)

    ax_vln.set_title(f"Epoch {peak_epoch}")
    ax_vln.axhline(0, color="grey", linewidth=0.5, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved to {out_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_log(log_dir: str) -> dict:
    log_path = os.path.join(log_dir, "log.txt")
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found", file=sys.stderr)
        sys.exit(1)
    data = parse_log(log_path)
    if len(data["epochs"]) == 0:
        print(f"Error: no epoch lines found in {log_path}", file=sys.stderr)
        sys.exit(1)
    return data


# ---------------------------------------------------------------------------
# Linear heritability maximizer comparison
# ---------------------------------------------------------------------------

def plot_linear_max(experiment_dir: str, out_path: str) -> None:
    """Plot raw per-dim h² vs theoretical and empirical linear-maximized h²."""
    lm_dir = os.path.join(experiment_dir, "linear_max")
    comparison_files = [
        f for f in os.listdir(lm_dir)
        if f.startswith("comparison.") and f.endswith(".txt")
    ]
    if not comparison_files:
        print(f"Error: no comparison.*.txt in {lm_dir}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(os.path.join(lm_dir, sorted(comparison_files)[-1]), sep="\t")

    fig, ax = plt.subplots(figsize=(12, 7))
    x = df["rank"].to_numpy()
    ax.plot(x, df["raw_h2"], label="Raw per-dim (sorted)", color=HUE_PALETTE[0], lw=2.5)
    ax.plot(x, df["theoretical_h2"], label="Linear maximizer (theoretical)", color=HUE_PALETTE[1], lw=2.5)
    ax.errorbar(
        x, df["empirical_h2"], yerr=df["empirical_h2_se"],
        label="Linear maximizer (empirical)", color=HUE_PALETTE[2],
        fmt="o", markersize=5, capsize=3, lw=1.5,
    )
    ax.set_xlabel("Component rank")
    ax.set_ylabel("h²")
    ax.set_title(os.path.basename(os.path.normpath(experiment_dir)))
    ax.legend()
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot heritability curves from HVAE logs")
    sub = parser.add_subparsers(dest="command", required=True)

    # single
    p_single = sub.add_parser("single", help="Plot curves for one experiment")
    p_single.add_argument("log_dir", type=str, help="Output directory containing log.txt")
    p_single.add_argument("--out", type=str, default=None)

    # compare
    p_cmp = sub.add_parser("compare", help="Compare control (h_weight=0) vs experiment")
    p_cmp.add_argument("control_dir", type=str, help="Control output directory (h_weight=0)")
    p_cmp.add_argument("experiment_dir", type=str, help="Experiment output directory")
    p_cmp.add_argument("--epoch", type=int, default=None, help="Epoch for violin plot (default: peak mean h² epoch)")
    p_cmp.add_argument("--out", type=str, default=None)

    # linear_max
    p_lm = sub.add_parser("linear_max", help="Compare raw per-dim h² vs linear heritability maximizer (theoretical + empirical)")
    p_lm.add_argument("experiment_dir", type=str, help="Experiment output directory")
    p_lm.add_argument("--out", type=str, default=None)

    args = parser.parse_args()

    if args.command == "single":
        data = _load_log(args.log_dir)
        if args.out is not None:
            out_path = args.out
        else:
            plots_dir = os.path.join(args.log_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            out_path = os.path.join(plots_dir, "heritability.png")
        plot_heritability(data, out_path)

    elif args.command == "compare":
        ctrl_data = _load_log(args.control_dir)
        exp_data = _load_log(args.experiment_dir)
        if args.out is not None:
            out_path = args.out
        else:
            plots_dir = os.path.join(args.experiment_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            out_path = os.path.join(plots_dir, "heritability_compare.png")
        plot_compare(ctrl_data, exp_data, out_path, epoch=args.epoch)

    elif args.command == "linear_max":
        if args.out is not None:
            out_path = args.out
        else:
            plots_dir = os.path.join(args.experiment_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            out_path = os.path.join(plots_dir, "linear_max.png")
        plot_linear_max(args.experiment_dir, out_path)


if __name__ == "__main__":
    main()
