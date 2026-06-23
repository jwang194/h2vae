"""Plot heritability curves over training epochs from an HVAE log file.

Each subcommand takes the log file itself (not its parent directory), so the
same plotter handles the main ``log.txt`` as well as rerun logs such as
``log.height_25.txt`` produced by ``rerun_heritability.py``.

Two flags select which heritability stream to plot:

    --split {train,val}     (default: val)
    --chrom {even,odd}      (default: odd; only meaningful for --split-variants logs)

Both choices are baked into the default output filename:
``<logfile_dir>/plots/<logfile_stem>.<split>[_<chrom>].png``.

Subcommands:

    python plot_heritability.py single <log_path> [--split S] [--chrom C] [--out PATH]
        Single-experiment plot with mean/max h² curves for the chosen stream.

    python plot_heritability.py compare <control_log> <experiment_log>
                               [--split S] [--chrom C] [--epoch N] [--out PATH]
        Side-by-side comparison on the chosen stream: h² curves (left) and a
        violin plot at the peak-heritability epoch of the experiment (right).
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
    r"epoch (?P<epoch>\d+) - mse_train: [\d.]+ - mse_val: [\d.]+"
    r" - h_train_even: (?P<train_even>[\d.,\s-]+)"
    r" - h_train_odd: (?P<train_odd>[\d.,\s-]+)"
    r" - h_val_even: (?P<val_even>[\d.,\s-]+)"
    r" - h_val_odd: (?P<val_odd>[\d.,\s-]+)"
)

_EPOCH_RE = re.compile(
    r"epoch (?P<epoch>\d+) - mse_train: [\d.]+ - mse_val: [\d.]+ "
    r"- h_train: (?P<train>[\d.,\s-]+) - h_val: (?P<val>[\d.,\s-]+)"
)


def parse_log(path: str) -> dict:
    """Parse a log.txt file and return all available heritability streams.

    Returns:
        dict with keys ``epochs`` and ``streams``.
        ``streams`` maps a stream name to an array of shape ``(n_epochs, zdim)``.
        Names are ``{train, val}`` for non-split logs, or
        ``{train_even, train_odd, val_even, val_odd}`` for split-variants logs.
    """
    by_epoch: dict[int, dict[str, list[float]]] = {}
    keys: tuple[str, ...] | None = None

    with open(path) as f:
        for line in f:
            m = _EPOCH_RE_SPLIT.search(line)
            if m is not None:
                if keys is None:
                    keys = ("train_even", "train_odd", "val_even", "val_odd")
                epoch = int(m.group("epoch"))
                by_epoch[epoch] = {
                    k: [float(x) for x in m.group(k).split(",")] for k in keys
                }
                continue
            m = _EPOCH_RE.search(line)
            if m is not None:
                if keys is None:
                    keys = ("train", "val")
                epoch = int(m.group("epoch"))
                by_epoch[epoch] = {
                    k: [float(x) for x in m.group(k).split(",")] for k in keys
                }

    sorted_epochs = sorted(by_epoch)
    if keys is None:
        return {"epochs": np.array([], dtype=int), "streams": {}}
    streams = {
        k: np.array([by_epoch[e][k] for e in sorted_epochs])
        for k in keys
    }
    return {
        "epochs": np.array(sorted_epochs),
        "streams": streams,
    }


def _resolve_stream_key(streams: dict, split: str, chrom: str | None) -> tuple[str, bool]:
    """Pick the stream name corresponding to ``--split`` / ``--chrom``.

    Returns ``(key, is_split_variants)``.  Raises ValueError if the requested
    stream is not present.
    """
    is_split = any("_" in k for k in streams)
    if is_split:
        eff_chrom = chrom if chrom is not None else "odd"
        key = f"{split}_{eff_chrom}"
    else:
        if chrom is not None:
            print(f"Warning: --chrom {chrom!r} ignored (non-split-variants log)",
                  file=sys.stderr)
        key = split
    if key not in streams:
        raise ValueError(
            f"stream {key!r} not present in log; available: {sorted(streams)}"
        )
    return key, is_split


# ---------------------------------------------------------------------------
# Line plot helpers
# ---------------------------------------------------------------------------

def _plot_lines(ax, epochs, h_matrix, color, label, mask=None, value_name="h^2"):
    """Plot mean (solid, with SD shading) and max (dashed) for one condition.

    If ``mask`` (same shape as ``h_matrix``) is provided, mean/std/max at each
    epoch are computed only over latent dims where the mask is True.  Epochs
    with no passing dim contribute NaN (and so leave a line gap).
    """
    if mask is None:
        mean = h_matrix.mean(axis=1)
        std = h_matrix.std(axis=1)
        mx = h_matrix.max(axis=1)
    else:
        masked = np.where(mask, h_matrix, np.nan)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(masked, axis=1)
            std = np.nanstd(masked, axis=1)
            mx = np.nanmax(masked, axis=1)

    ax.plot(epochs, mean, color=color, linestyle="-", label=f"Mean ${value_name}$ ({label})")
    ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)
    ax.plot(epochs, mx, color=color, linestyle="--", label=f"Max ${value_name}$ ({label})")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heritability(data: dict, out_path: str, stream_key: str) -> None:
    epochs = data["epochs"]
    h_matrix = data["streams"][stream_key]

    fig, ax = plt.subplots()
    _plot_lines(ax, epochs, h_matrix, HUE_PALETTE[0], stream_key.replace("_", " "))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("$h^2$")
    ax.set_title(f"Latent heritability over training — {stream_key.replace('_', ' ')}")
    ax.legend()
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


def _align_mask(value_epochs, value_matrix, mask_epochs, mask_matrix, threshold):
    """Return a boolean mask aligned to ``value_epochs`` of shape ``value_matrix``.

    For each value epoch, picks the nearest mask epoch and tests
    ``mask_matrix[mask_idx] >= threshold``.  Shapes must match on the latent
    axis.
    """
    if mask_matrix.shape[1] != value_matrix.shape[1]:
        raise ValueError(
            f"mask latent dim ({mask_matrix.shape[1]}) != value latent dim "
            f"({value_matrix.shape[1]})"
        )
    idxs = np.array([
        int(np.argmin(np.abs(mask_epochs - e))) for e in value_epochs
    ])
    return mask_matrix[idxs] >= threshold


def plot_compare(
    ctrl_data: dict,
    exp_data: dict,
    out_path: str,
    stream_key: str,
    epoch: int | None = None,
    ctrl_mask_data: dict | None = None,
    exp_mask_data: dict | None = None,
    mask_threshold: float | None = None,
    value_name: str = "h^2",
    absolute: bool = False,
) -> None:
    """Side-by-side comparison of control (h_weight=0) vs experiment.

    Left panel: mean value (with SD shading) and max value for both runs on
    the selected stream.  Right panel: seaborn violin + boxplot of per-latent
    values at the experiment's peak epoch, with Wilcoxon rank-sum p-value
    bracket.

    If ``ctrl_mask_data`` / ``exp_mask_data`` (each parsed by ``parse_log``)
    and ``mask_threshold`` are provided, per-dim values are included in the
    line and violin panels only where the mask stream (e.g. MoM h²) on the
    same split meets the threshold at the matching epoch.
    """
    fig, (ax_ts, ax_vln) = plt.subplots(1, 2, figsize=(20, 9))

    c_ctrl, c_exp = HUE_PALETTE[0], HUE_PALETTE[1]

    ctrl_h = ctrl_data["streams"][stream_key]
    exp_h = exp_data["streams"][stream_key]
    if absolute:
        ctrl_h = np.abs(ctrl_h)
        exp_h = np.abs(exp_h)

    nice = stream_key.replace("_", " ")
    display_name = f"|{value_name}|" if absolute else value_name
    y_label = f"${display_name}$ ({nice})"

    ctrl_mask = None
    exp_mask = None
    if mask_threshold is not None:
        if ctrl_mask_data is None or exp_mask_data is None:
            raise ValueError(
                "mask_threshold requires both ctrl_mask_data and exp_mask_data"
            )
        ctrl_mask = _align_mask(
            ctrl_data["epochs"], ctrl_h,
            ctrl_mask_data["epochs"], ctrl_mask_data["streams"][stream_key],
            mask_threshold,
        )
        exp_mask = _align_mask(
            exp_data["epochs"], exp_h,
            exp_mask_data["epochs"], exp_mask_data["streams"][stream_key],
            mask_threshold,
        )

    # --- Left panel: time-series ---
    _plot_lines(ax_ts, ctrl_data["epochs"], ctrl_h, c_ctrl, "Control",
                mask=ctrl_mask, value_name=display_name)
    _plot_lines(ax_ts, exp_data["epochs"], exp_h, c_exp, "Experiment",
                mask=exp_mask, value_name=display_name)

    ax_ts.set_xlabel("Epoch")
    ax_ts.set_ylabel(y_label)
    title = f"${display_name}$ over training — {nice}"
    if mask_threshold is not None:
        title += f" (MoM $h^2 \\geq$ {mask_threshold})"
    ax_ts.set_title(title)
    ax_ts.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax_ts.legend(fontsize=14)

    # --- Right panel: violin at target epoch ---
    if epoch is not None:
        peak_idx = int(np.argmin(np.abs(exp_data["epochs"] - epoch)))
        peak_epoch = exp_data["epochs"][peak_idx]
    else:
        if exp_mask is None:
            peak_series = exp_h.mean(axis=1)
        else:
            masked = np.where(exp_mask, exp_h, np.nan)
            with np.errstate(invalid="ignore"):
                peak_series = np.nanmean(masked, axis=1)
        peak_idx = int(np.nanargmax(peak_series))
        peak_epoch = exp_data["epochs"][peak_idx]

    ctrl_idx = int(np.argmin(np.abs(ctrl_data["epochs"] - peak_epoch)))

    h_ctrl = ctrl_h[ctrl_idx]
    h_exp = exp_h[peak_idx]
    if ctrl_mask is not None:
        h_ctrl = h_ctrl[ctrl_mask[ctrl_idx]]
    if exp_mask is not None:
        h_exp = h_exp[exp_mask[peak_idx]]

    df = pd.DataFrame({
        y_label: np.concatenate([h_ctrl, h_exp]),
        "Condition": ["Control"] * len(h_ctrl) + ["Experiment"] * len(h_exp),
    })

    sns.violinplot(
        data=df, x="Condition", y=y_label, hue="Condition",
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

def _load_log(log_path: str) -> dict:
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found", file=sys.stderr)
        sys.exit(1)
    data = parse_log(log_path)
    if len(data["epochs"]) == 0:
        print(f"Error: no epoch lines found in {log_path}", file=sys.stderr)
        sys.exit(1)
    return data


def _default_plot_path(log_path: str, suffix: str = "") -> str:
    """Build ``<log_dir>/plots/<log_stem><suffix>.png`` from a log path."""
    log_dir = os.path.dirname(os.path.abspath(log_path))
    stem = os.path.basename(log_path)
    if stem.endswith(".txt"):
        stem = stem[: -len(".txt")]
    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return os.path.join(plots_dir, stem + suffix + ".png")


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

def _add_stream_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--split", choices=["train", "val"], default="val",
                   help="Which split's heritability to plot (default: val)")
    p.add_argument("--chrom", choices=["even", "odd"], default=None,
                   help="For split-variants logs: which chromosome subset to plot "
                        "(default: odd; ignored for non-split logs)")


def _stream_suffix(split: str, chrom: str | None, is_split: bool) -> str:
    if is_split:
        return f".{split}_{chrom or 'odd'}"
    return f".{split}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot heritability curves from HVAE logs")
    sub = parser.add_subparsers(dest="command", required=True)

    # single
    p_single = sub.add_parser("single", help="Plot curves for one experiment")
    p_single.add_argument("log_path", type=str, help="Path to a log.txt (or rerun-style log file)")
    _add_stream_flags(p_single)
    p_single.add_argument("--out", type=str, default=None,
                          help="Output image path (default: "
                               "<log_dir>/plots/<log_stem>.<split>[_<chrom>].png)")

    # compare
    p_cmp = sub.add_parser("compare", help="Compare control (h_weight=0) vs experiment")
    p_cmp.add_argument("control_log", type=str, help="Path to the control log file")
    p_cmp.add_argument("experiment_log", type=str, help="Path to the experiment log file")
    _add_stream_flags(p_cmp)
    p_cmp.add_argument("--epoch", type=int, default=None, help="Epoch for violin plot (default: peak mean h² epoch)")
    p_cmp.add_argument("--mask-log-ctrl", type=str, default=None,
                       help="Optional companion log (e.g. MoM-rerun log) for the control run; "
                            "the same stream is used as a per-dim, per-epoch mask via --mask-threshold")
    p_cmp.add_argument("--mask-log-exp", type=str, default=None,
                       help="Optional companion log for the experiment run (paired with --mask-log-ctrl)")
    p_cmp.add_argument("--mask-threshold", type=float, default=None,
                       help="Threshold applied to the mask logs: include only latent dims where "
                            "the mask stream is >= this value at the matching epoch")
    p_cmp.add_argument("--value-name", type=str, default="h^2",
                       help="LaTeX-style label for the plotted quantity (default: h^2; use e.g. "
                            "'\\hat{\\rho}' for genetic correlation)")
    p_cmp.add_argument("--abs", action="store_true", default=False,
                       help="Plot |value| instead of signed value (useful for genetic "
                            "correlation, whose sign is latent-orientation-arbitrary)")
    p_cmp.add_argument("--out", type=str, default=None,
                       help="Output image path (default: "
                            "<exp_log_dir>/plots/<exp_log_stem>.<split>[_<chrom>].compare.png)")

    # linear_max
    p_lm = sub.add_parser("linear_max", help="Compare raw per-dim h² vs linear heritability maximizer (theoretical + empirical)")
    p_lm.add_argument("experiment_dir", type=str, help="Experiment output directory")
    p_lm.add_argument("--out", type=str, default=None)

    args = parser.parse_args()

    if args.command == "single":
        data = _load_log(args.log_path)
        stream_key, is_split = _resolve_stream_key(data["streams"], args.split, args.chrom)
        out_path = args.out or _default_plot_path(
            args.log_path, suffix=_stream_suffix(args.split, args.chrom, is_split),
        )
        plot_heritability(data, out_path, stream_key)

    elif args.command == "compare":
        ctrl_data = _load_log(args.control_log)
        exp_data = _load_log(args.experiment_log)
        stream_key, is_split = _resolve_stream_key(exp_data["streams"], args.split, args.chrom)
        # Sanity-check that the same stream is available in the control log
        if stream_key not in ctrl_data["streams"]:
            raise ValueError(
                f"stream {stream_key!r} not in control log "
                f"(available: {sorted(ctrl_data['streams'])})"
            )
        ctrl_mask_data = None
        exp_mask_data = None
        if args.mask_threshold is not None:
            if args.mask_log_ctrl is None or args.mask_log_exp is None:
                parser.error("--mask-threshold requires --mask-log-ctrl and --mask-log-exp")
            ctrl_mask_data = _load_log(args.mask_log_ctrl)
            exp_mask_data = _load_log(args.mask_log_exp)
            for mname, mdata in (("ctrl", ctrl_mask_data), ("exp", exp_mask_data)):
                if stream_key not in mdata["streams"]:
                    raise ValueError(
                        f"stream {stream_key!r} not in {mname} mask log "
                        f"(available: {sorted(mdata['streams'])})"
                    )
        suffix = _stream_suffix(args.split, args.chrom, is_split) + ".compare"
        if args.mask_threshold is not None:
            suffix += f".mask{args.mask_threshold}"
        if args.abs:
            suffix += ".abs"
        out_path = args.out or _default_plot_path(args.experiment_log, suffix=suffix)
        plot_compare(
            ctrl_data, exp_data, out_path, stream_key,
            epoch=args.epoch,
            ctrl_mask_data=ctrl_mask_data,
            exp_mask_data=exp_mask_data,
            mask_threshold=args.mask_threshold,
            value_name=args.value_name,
            absolute=args.abs,
        )

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
