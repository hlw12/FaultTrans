"""Publication style for the FaultTrans figure campaign. Python-only."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path("/root/autodl-tmp/nature_figures")
EXPORT = ROOT / "exports"
SOURCE = ROOT / "source_data"
CACHE = ROOT / "cache"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "red_2": "#E9A6A1",
    "teal": "#42949E",
    "gold": "#C9A227",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
    "bg": "#F4F4F4",
    "tp": "#2A9D8F",
    "fp": "#C45C26",
    "fn": "#3D5A80",
    "pgv": "#0F4D92",
    "pga": "#9A4D8E",
    "finder": "#B64342",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.6,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "mathtext.default": "regular",
    }
)


def ensure_dirs():
    for p in (EXPORT, SOURCE, CACHE):
        p.mkdir(parents=True, exist_ok=True)


def save_pub(fig, stem: str, subdir: str | None = None):
    ensure_dirs()
    stem = Path(stem).stem
    out = EXPORT / subdir if subdir else EXPORT
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.relative_to(EXPORT) / stem if subdir else stem}")


def save_panel(fig, stem: str, subdir: str | None = None):
    """Single-panel export (SVG/PDF/TIFF/PNG). Same formats as save_pub."""
    save_pub(fig, stem, subdir=subdir)


def panel_label(ax, letter, x=-0.08, y=1.08):
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=PALETTE["neutral_black"],
    )
