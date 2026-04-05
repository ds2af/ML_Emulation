"""
scripts/plot_training_curves.py
================================
Generate training/validation loss curves and RMSE / Relative-L2 bar charts
for all five surrogate models.

Figures produced
----------------
figures/training_loss_curves.png   – train loss vs epoch (log y)
figures/val_loss_curves.png        – val loss vs epoch   (log y, validation-only epochs)
figures/rmse_bar.png               – RMSE bar chart across models
figures/rel_l2_bar.png             – Relative L2 bar chart across models
figures/metrics_combined.png       – 2x2 combined panel (loss curves + both bars)

Usage
-----
    python scripts/plot_training_curves.py
"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
})

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette (colorblind-safe IBM Design) ───────────────────────────────
COLOURS = {
    "mlp":        "#648FFF",   # blue
    "cnn":        "#FE6100",   # orange
    "unet":       "#DC267F",   # magenta
    "unet_lomix": "#785EF0",   # purple
    "unet_lomix_pinn": "#1F6F3D",  # deep green
    "fno":        "#FFB000",   # gold
    "pinn":       "#009E73",   # teal
}

LABELS = {
    "mlp":        "MLP (coord-based)",
    "cnn":        "CNN",
    "unet":       "U-Net",
    "unet_lomix": "U-Net LoMix",
    "unet_lomix_pinn": "U-Net LoMix + PINN",
    "fno":        "FNO",
    "pinn":       "PINN",
}

# Order for display
DEFAULT_MODEL_ORDER = ["mlp", "cnn", "unet", "unet_lomix", "unet_lomix_pinn", "fno", "pinn"]
MODEL_ORDER = list(DEFAULT_MODEL_ORDER)

LOG_DIR     = ROOT / "results" / "logs"
METRICS_DIR = ROOT / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_log(model: str) -> pd.DataFrame:
    path = LOG_DIR / f"{model}_log.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def load_metrics(model: str) -> dict:
    path = METRICS_DIR / f"{model}_metrics.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exclude_models",
        default="",
        help="Comma-separated model keys to exclude from all generated figures",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Training loss curves (one panel per model group)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_loss():
    fig, ax = plt.subplots(figsize=(9, 5))

    for model in MODEL_ORDER:
        df = load_log(model)
        if df.empty:
            continue
        ax.plot(
            df["epoch"],
            df["train_loss"],
            color=COLOURS[model],
            label=LABELS[model],
            linewidth=1.5,
            alpha=0.88,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (MSE, log scale)")
    ax.set_title("Training Loss Curves – All Models")
    ax.legend(loc="upper right", framealpha=0.85)
    plt.tight_layout()
    out = FIG_DIR / "training_loss_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Validation loss curves (only epochs where val was computed)
# ─────────────────────────────────────────────────────────────────────────────

def plot_val_loss():
    fig, ax = plt.subplots(figsize=(9, 5))

    plotted = False
    for model in MODEL_ORDER:
        df = load_log(model)
        if df.empty:
            continue
        # Keep only rows where validation was actually computed (val_loss > 0)
        val_df = df[df["val_loss"] > 0].copy()
        if val_df.empty:
            continue
        ax.plot(
            val_df["epoch"],
            val_df["val_loss"],
            color=COLOURS[model],
            label=LABELS[model],
            linewidth=1.8,
            marker="o",
            markersize=3,
            alpha=0.88,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        print("[plot] No validation data found – skipping val loss plot.")
        return

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss (MSE, log scale)")
    ax.set_title("Validation Loss Curves – All Models")
    ax.legend(loc="upper right", framealpha=0.85)
    plt.tight_layout()
    out = FIG_DIR / "val_loss_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – RMSE bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_rmse_bar():
    names, values, colours = [], [], []
    for model in MODEL_ORDER:
        m = load_metrics(model)
        if not m or "rmse" not in m:
            continue
        names.append(LABELS[model])
        values.append(m["rmse"])
        colours.append(COLOURS[model])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    bars = ax.bar(x, values, color=colours, edgecolor="white", linewidth=0.8, width=0.55)

    # Value labels on top of bars
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.012,
            f"{v:.4f}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Test RMSE – All Models")
    ax.set_ylim(0, max(values) * 1.18)
    plt.tight_layout()
    out = FIG_DIR / "rmse_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – Relative L2 bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_rel_l2_bar():
    names, values, colours = [], [], []
    for model in MODEL_ORDER:
        m = load_metrics(model)
        if not m or "rel_l2" not in m:
            continue
        names.append(LABELS[model])
        values.append(m["rel_l2"])
        colours.append(COLOURS[model])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    bars = ax.bar(x, values, color=colours, edgecolor="white", linewidth=0.8, width=0.55)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.012,
            f"{v:.4f}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("Test Relative L2 Error – All Models")
    ax.set_ylim(0, max(values) * 1.18)
    plt.tight_layout()
    out = FIG_DIR / "rel_l2_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – 2×2 combined panel
# ─────────────────────────────────────────────────────────────────────────────

def plot_combined():
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32)

    ax_train = fig.add_subplot(gs[0, 0])
    ax_val   = fig.add_subplot(gs[0, 1])
    ax_rmse  = fig.add_subplot(gs[1, 0])
    ax_l2    = fig.add_subplot(gs[1, 1])

    # ── (A) Training loss ────────────────────────────────────────────────────
    for model in MODEL_ORDER:
        df = load_log(model)
        if df.empty:
            continue
        ax_train.plot(
            df["epoch"], df["train_loss"],
            color=COLOURS[model], label=LABELS[model],
            linewidth=1.5, alpha=0.88,
        )
    ax_train.set_yscale("log")
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Train Loss (MSE)")
    ax_train.set_title("(a) Training Loss")
    ax_train.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_train.grid(True, alpha=0.3)
    ax_train.spines["top"].set_visible(False)
    ax_train.spines["right"].set_visible(False)

    # ── (B) Validation loss ──────────────────────────────────────────────────
    any_val = False
    for model in MODEL_ORDER:
        df = load_log(model)
        if df.empty:
            continue
        val_df = df[df["val_loss"] > 0]
        if val_df.empty:
            continue
        ax_val.plot(
            val_df["epoch"], val_df["val_loss"],
            color=COLOURS[model], label=LABELS[model],
            linewidth=1.8, marker="o", markersize=3, alpha=0.88,
        )
        any_val = True
    ax_val.set_yscale("log")
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Val Loss (MSE)")
    ax_val.set_title("(b) Validation Loss")
    if any_val:
        ax_val.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_val.grid(True, alpha=0.3)
    ax_val.spines["top"].set_visible(False)
    ax_val.spines["right"].set_visible(False)

    # ── (C) RMSE bar ─────────────────────────────────────────────────────────
    rmse_names, rmse_vals, rmse_cols = [], [], []
    for model in MODEL_ORDER:
        m = load_metrics(model)
        if m and "rmse" in m:
            rmse_names.append(LABELS[model])
            rmse_vals.append(m["rmse"])
            rmse_cols.append(COLOURS[model])

    x_r = np.arange(len(rmse_names))
    bars_r = ax_rmse.bar(x_r, rmse_vals, color=rmse_cols, edgecolor="white", linewidth=0.8, width=0.55)
    for bar, v in zip(bars_r, rmse_vals):
        ax_rmse.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(rmse_vals) * 0.012,
            f"{v:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
        )
    ax_rmse.set_xticks(x_r)
    ax_rmse.set_xticklabels(rmse_names, rotation=22, ha="right", fontsize=8.5)
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("(c) Test RMSE")
    ax_rmse.set_ylim(0, max(rmse_vals) * 1.20)
    ax_rmse.grid(True, axis="y", alpha=0.3)
    ax_rmse.spines["top"].set_visible(False)
    ax_rmse.spines["right"].set_visible(False)

    # ── (D) Relative L2 bar ──────────────────────────────────────────────────
    l2_names, l2_vals, l2_cols = [], [], []
    for model in MODEL_ORDER:
        m = load_metrics(model)
        if m and "rel_l2" in m:
            l2_names.append(LABELS[model])
            l2_vals.append(m["rel_l2"])
            l2_cols.append(COLOURS[model])

    x_l = np.arange(len(l2_names))
    bars_l = ax_l2.bar(x_l, l2_vals, color=l2_cols, edgecolor="white", linewidth=0.8, width=0.55)
    for bar, v in zip(bars_l, l2_vals):
        ax_l2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(l2_vals) * 0.012,
            f"{v:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
        )
    ax_l2.set_xticks(x_l)
    ax_l2.set_xticklabels(l2_names, rotation=22, ha="right", fontsize=8.5)
    ax_l2.set_ylabel("Relative L2 Error")
    ax_l2.set_title("(d) Test Relative L2 Error")
    ax_l2.set_ylim(0, max(l2_vals) * 1.20)
    ax_l2.grid(True, axis="y", alpha=0.3)
    ax_l2.spines["top"].set_visible(False)
    ax_l2.spines["right"].set_visible(False)

    fig.suptitle("ML Surrogate Models – Training Curves & Test Metrics\n(PDEBench 2D Radial Dam-Break SWE)",
                 fontsize=13, fontweight="bold", y=1.01)

    out = FIG_DIR / "metrics_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 – CNN 200-epoch focused: loss + accuracy
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_loss_200ep():
    """
    Single-panel training-loss comparison focused on the first 200 epochs.

    CNN, FNO, U-Net, U-Net LoMix  → full logs (0-199).
    PINN             → first 200 rows of seed-0 (global epochs 0-199).
    Log y-axis.  Saved as figures/training_loss_200ep.png.
    """
    FOCUS_MODELS = ["cnn", "unet", "unet_lomix", "unet_lomix_pinn", "fno", "pinn"]
    FOCUS_MODELS = [model for model in FOCUS_MODELS if model in MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(10, 5.2))

    for model in FOCUS_MODELS:
        df = load_log(model)
        if df.empty:
            continue
        if model == "pinn":
            # Seed 0 occupies global epochs 0–1999; compress onto 0-199 window.
            sub = df[df["epoch"] <= 1999].copy()
            x = sub["epoch"].values * (200 / 1999)
        else:
            sub = df[df["epoch"] <= 199].copy()
            x = sub["epoch"].values
        if len(sub) == 0:
            continue
        label_suffix = " (2000 ep, compressed)" if model == "pinn" else ""
        ax.plot(
            x, sub["train_loss"],
            color=COLOURS[model],
            label=LABELS[model] + label_suffix,
            linewidth=1.8,
            alpha=0.88,
        )

    ax.set_yscale("log")
    ax.set_xlim(0, 199)
    ax.set_xlabel("Epoch", fontsize=18)
    ax.set_ylabel("Training Loss (MSE, log scale)", fontsize=18)
    ax.set_title(
        "Training Loss – First 200 Epochs  (2D Radial Dam-Break SWE, PDEBench)",
        fontweight="bold",
        fontsize=15,
    )
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(fontsize=16, framealpha=0.88)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out = FIG_DIR / "training_loss_200ep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {out}")


def plot_cnn_focused():
    """
    Two-panel figure emphasising CNN's 200-epoch training dynamics:
      (a) Training loss and sparse val loss on a log y-axis.
      (b) Accuracy proxy: 100 × (1 − RMSE_epoch / RMSE_epoch0)
          shown for both train and, where available, validation.

    'Accuracy' here is the standard regression convergence indicator –
    the percentage reduction in RMSE relative to the untrained baseline.
    """
    if "cnn" not in MODEL_ORDER:
        print("[plot] Skipping CNN-focused figure because CNN is excluded.")
        return

    df = load_log("cnn")
    if df.empty:
        print("[plot] CNN log not found – skipping CNN focused plot.")
        return

    # Focus on the 200-epoch window
    df = df[df["epoch"] <= 199].copy().reset_index(drop=True)
    val_df = df[df["val_loss"] > 0].copy().reset_index(drop=True)

    df["train_rmse"] = np.sqrt(df["train_loss"])

    # Accuracy proxy (%) –  % RMSE reduction relative to epoch 0
    rmse0_train = float(df["train_rmse"].iloc[0]) or 1.0
    df["train_acc"] = (1.0 - df["train_rmse"] / rmse0_train) * 100.0

    if not val_df.empty:
        val_df["val_rmse"] = np.sqrt(val_df["val_loss"])
        rmse0_val = float(val_df["val_rmse"].iloc[0]) or 1.0
        val_df["val_acc"] = (1.0 - val_df["val_rmse"] / rmse0_val) * 100.0

    colour = COLOURS["cnn"]
    dark   = "#B84400"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # ── Panel (a): Loss curves ────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"],
            color=colour, linewidth=1.8, label="Train loss (MSE)", alpha=0.88)
    if not val_df.empty:
        ax.plot(val_df["epoch"], val_df["val_loss"],
                "--", color=dark, linewidth=1.2, alpha=0.75, label="Val loss (MSE)")
        ax.scatter(val_df["epoch"], val_df["val_loss"],
                   color=dark, s=22, zorder=5, edgecolors="white", linewidths=0.6)

    ax.set_yscale("log")
    ax.set_xlim(0, 199)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE, log scale)")
    ax.set_title("(a)  CNN – Training & Validation Loss", fontweight="bold")
    ax.legend(framealpha=0.88)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel (b): Accuracy proxy ─────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(df["epoch"], df["train_acc"],
             color=colour, linewidth=1.8, label="Train accuracy", alpha=0.88)
    if not val_df.empty:
        ax2.plot(val_df["epoch"], val_df["val_acc"],
                 "--", color=dark, linewidth=1.2, alpha=0.75, label="Val accuracy")
        ax2.scatter(val_df["epoch"], val_df["val_acc"],
                    color=dark, s=22, zorder=5, edgecolors="white", linewidths=0.6)

    # Annotate final val accuracy
    if not val_df.empty:
        last_epoch = val_df["epoch"].iloc[-1]
        last_acc   = val_df["val_acc"].iloc[-1]
        ax2.annotate(
            f"{last_acc:.1f}% @ ep {last_epoch}",
            xy=(last_epoch, last_acc),
            xytext=(-48, -14),
            textcoords="offset points",
            fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="#555", lw=0.9),
        )

    ax2.set_xlim(0, 199)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy proxy  (% RMSE reduction vs. epoch 0)")
    ax2.set_title("(b)  CNN – Prediction Accuracy (200 epochs)", fontweight="bold")
    ax2.legend(framealpha=0.88)
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        "CNN Surrogate Model – 200-Epoch Training Dynamics\n"
        "(2D Radial Dam-Break Shallow-Water Equations, PDEBench)",
        fontsize=12, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    out = FIG_DIR / "cnn_training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {out}")


def plot_pinn_seed1_compact_200():
    """
    PINN-only loss visualization for seed 1:
      (a) full 2000 training epochs compacted into 200 points
      (b) zoom on first 200 actual epochs

    Output:
      figures/pinn_seed1_loss_compact_200.png
    """
    if "pinn" not in MODEL_ORDER:
        print("[plot] Skipping PINN compact loss figure because PINN is excluded.")
        return

    df = load_log("pinn")
    if df.empty:
        print("[plot] PINN log not found - skipping seed-1 compacted loss plot.")
        return

    df = df.copy()
    for col in ["epoch", "train_loss", "val_loss"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["epoch", "train_loss"]).sort_values("epoch").reset_index(drop=True)
    if df.empty:
        print("[plot] PINN log has no numeric rows - skipping seed-1 compacted loss plot.")
        return

    seed_span = 2000
    seed1 = df[(df["epoch"] >= seed_span) & (df["epoch"] < 2 * seed_span)].copy()
    if seed1.empty:
        if len(df) >= seed_span:
            seed1 = df.tail(seed_span).copy()
            print("[plot] Seed-1 epoch window not found; using last 2000 PINN rows as fallback.")
        else:
            seed1 = df.copy()
            print("[plot] PINN log shorter than 2000 rows; using all available rows.")

    seed1 = seed1.sort_values("epoch").reset_index(drop=True)
    n_seed = len(seed1)

    n_compact = max(2, min(200, n_seed))
    bins = np.array_split(np.arange(n_seed), n_compact)
    x_comp = np.arange(n_compact)
    y_comp_train = np.array([float(seed1.iloc[idx]["train_loss"].mean()) for idx in bins])
    y_comp_data = np.array([float(seed1.iloc[idx]["val_loss"].mean()) for idx in bins])

    focus_n = min(200, n_seed)
    focus = seed1.iloc[:focus_n].copy()
    x_focus = np.arange(focus_n)
    focus_smooth = focus["train_loss"].rolling(window=9, center=True, min_periods=1).mean()

    pinn_color = COLOURS["pinn"]
    data_color = "#2F4F4F"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    ax.plot(x_comp, y_comp_train, color=pinn_color, linewidth=2.0, label="Total loss")
    ax.plot(x_comp, y_comp_data, "--", color=data_color, linewidth=1.4, alpha=0.9, label="Data term (logged as val_loss)")
    ax.set_yscale("log")
    ax.set_xlim(0, max(1, n_compact - 1))
    ax.set_xlabel("Compacted epoch index")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(f"(a) Seed 1: {n_seed} epochs compacted to {n_compact} points", fontweight="bold")
    ax.legend(framealpha=0.88)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    ax2.plot(x_focus, focus["train_loss"].values, color=pinn_color, linewidth=1.6, alpha=0.55, label="Total loss (raw)")
    ax2.plot(x_focus, focus_smooth.values, color=pinn_color, linewidth=2.1, label="Total loss (smoothed)")
    ax2.plot(x_focus, focus["val_loss"].values, "--", color=data_color, linewidth=1.2, alpha=0.9, label="Data term")
    ax2.set_yscale("log")
    ax2.set_xlim(0, max(1, focus_n - 1))
    ax2.set_xlabel("Epoch (first 200)")
    ax2.set_ylabel("Loss (log scale)")
    ax2.set_title("(b) Seed 1: focus on first 200 epochs", fontweight="bold")
    ax2.legend(framealpha=0.88)
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("PINN Loss Curve (Seed 1) - 2000 epochs compacted + 200-epoch focus", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "pinn_seed1_loss_compact_200.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {out}")


if __name__ == "__main__":
    args = parse_args()
    excluded = {item.strip() for item in args.exclude_models.split(",") if item.strip()}
    MODEL_ORDER = [model for model in DEFAULT_MODEL_ORDER if model not in excluded]
    if excluded:
        print(f"[plot] Excluding models: {', '.join(sorted(excluded))}")

    print("[plot] Generating training curves and metric bar charts …")
    plot_training_loss()
    plot_val_loss()
    plot_rmse_bar()
    plot_rel_l2_bar()
    plot_combined()
    plot_training_loss_200ep()
    plot_cnn_focused()
    plot_pinn_seed1_compact_200()
    print(f"\n[plot] All figures saved to {FIG_DIR}/")
