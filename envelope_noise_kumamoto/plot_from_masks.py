#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot 2x2 from saved masks. No torch — stays inside the 2 GB cgroup."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ROOT = Path("/root/autodl-tmp")
HERE = ROOT / "envelope_noise_kumamoto"
OUT = HERE / "outputs"
FIGDIR = OUT / "figures"
MASKDIR = OUT / "masks"
UNIFIED = ROOT / "Test_on_Event" / "unified"
EVENT_DIR = ROOT / "Test_on_Event" / "kumamoto"

sys.path.insert(0, str(ROOT / "CmpFinDer"))
sys.path.insert(0, str(ROOT / "nature_figures"))
sys.path.insert(0, str(UNIFIED))

from geo import compute_metrics  # noqa: E402
from fields import strike_delta  # noqa: E402
from regen_finder_static_panel import (  # noqa: E402
    FINDER_COLOR,
    FT_COLOR,
    GT_COLOR,
    HALF_DEG,
    WHITE_VEIL_ALPHA,
    load_dem_rgb,
    load_meta,
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def score(pred, gt):
    m = compute_metrics(pred, gt)
    return float(m["IoU"]), float(m["F1"]), float(strike_delta(pred, gt))


def load_clean(gt, idx):
    ft = np.load(UNIFIED / "outputs" / "infer_max" / "kumamoto" / "pred_binary.npy")
    fd = np.load(EVENT_DIR / "finder_pred_line_mask.npy")
    if ft.ndim == 3:
        ft = ft[min(idx, ft.shape[0] - 1)]
    if fd.ndim == 3:
        fd = fd[min(idx, fd.shape[0] - 1)]
    return ft.astype(np.uint8), fd.astype(np.uint8)


def _overlay_mask(ax, mask, color, alpha, extent):
    m = (np.asarray(mask) > 0).astype(np.float32)
    rgba = np.zeros((*m.shape, 4), dtype=np.float32)
    rgba[m > 0] = (*to_rgb(color), alpha)
    ax.imshow(rgba, extent=extent, origin="lower", aspect="equal", interpolation="nearest", zorder=3)


def _legend_handles(kind):
    pred = (
        Line2D([0], [0], color=FINDER_COLOR, lw=2.2, label="FinDer line")
        if kind == "finder"
        else Patch(facecolor=FT_COLOR, edgecolor="white", linewidth=0.5, alpha=0.7, label="FaultTrans")
    )
    return [
        Patch(facecolor=GT_COLOR, edgecolor="white", linewidth=0.5, alpha=0.7, label="GT (Curved)"),
        pred,
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gold",
            markeredgecolor="#333333",
            markersize=10,
            linestyle="None",
            label="Epicentre",
        ),
    ]


def _extent(art):
    return [art["lon_min"], art["lon_max"], art["lat_min"], art["lat_max"]]


def _base_map(ax, art):
    lon_min, lon_max = art["lon_min"], art["lon_max"]
    lat_min, lat_max = art["lat_min"], art["lat_max"]
    extent = _extent(art)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal")
    ax.imshow(art["dem_rgb"], extent=extent, origin="upper", aspect="equal", zorder=0, interpolation="bilinear")
    ax.add_patch(
        Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            facecolor="white",
            edgecolor="none",
            alpha=WHITE_VEIL_ALPHA,
            zorder=1,
        )
    )
    ax.add_patch(
        Rectangle(
            (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
            linewidth=0.8, edgecolor="#333333", facecolor="none", linestyle="--", zorder=4,
        )
    )
    ax.scatter(
        art["epi_lon"], art["epi_lat"], s=48, c="gold", marker="*",
        edgecolors="#333333", linewidths=0.4, zorder=7,
    )
    return extent


def _letter(ax, letter):
    if not letter:
        return
    ax.text(
        0.03, 0.97, letter, transform=ax.transAxes, va="top", ha="left",
        fontsize=8, fontweight="bold", color="#111111", zorder=12,
        bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none", alpha=0.88),
    )


# Muted earth sequential (YlOrBr family, no fluorescent yellow / white-hot peak).
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "ylorbr_mute",
    ["#F3EADF", "#E6CFA3", "#CDA35A", "#A56B32", "#6E4324"],
)


def _axis_labels(ax, show_x, show_y):
    ax.tick_params(axis="x", labelsize=5.5, rotation=25)
    ax.tick_params(axis="y", labelsize=5.5)
    ax.set_xlabel("Longitude (°E)" if show_x else "", fontsize=6.5)
    ax.set_ylabel("Latitude (°N)" if show_y else "", fontsize=6.5)
    if not show_x:
        ax.set_xticklabels([])
    if not show_y:
        ax.set_yticklabels([])


def draw_field(ax, art, field, title, letter, vmax, show_x=True, show_y=True):
    lon_min, lon_max = art["lon_min"], art["lon_max"]
    lat_min, lat_max = art["lat_min"], art["lat_max"]
    extent = _extent(art)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal")
    ax.set_facecolor("#F4F1EA")
    vis = np.log1p(np.maximum(np.asarray(field, np.float64), 0.0))
    im = ax.imshow(
        vis,
        extent=extent,
        origin="lower",
        aspect="equal",
        cmap=FIELD_CMAP,
        vmin=0.0,
        vmax=float(np.log1p(vmax)),
        interpolation="nearest",
        zorder=2,
    )
    ax.add_patch(
        Rectangle(
            (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
            linewidth=0.8, edgecolor="#4D4D4D", facecolor="none", linestyle="--", zorder=4,
        )
    )
    ax.scatter(
        art["epi_lon"], art["epi_lat"], s=42, c="#FFD700", marker="*",
        edgecolors="#272727", linewidths=0.35, zorder=7,
    )
    _letter(ax, letter)
    ax.set_title(title, fontsize=7.5, fontweight="bold", pad=3)
    _axis_labels(ax, show_x, show_y)
    return im


def draw_panel(ax, art, mask_gt, pred_mask, kind, title, letter, show_legend, show_x=True, show_y=True):
    extent = _base_map(ax, art)
    _overlay_mask(ax, mask_gt, GT_COLOR, 0.45, extent)
    if kind == "finder":
        _overlay_mask(ax, pred_mask, FINDER_COLOR, 0.95, extent)
    else:
        _overlay_mask(ax, pred_mask, FT_COLOR, 0.55, extent)
    _letter(ax, letter)
    if show_legend:
        leg = ax.legend(
            handles=_legend_handles(kind),
            loc="upper left",
            bbox_to_anchor=(0.14, 0.98),
            fontsize=5.5,
            frameon=True,
            fancybox=False,
            edgecolor="#cccccc",
            facecolor="white",
            framealpha=0.92,
            borderpad=0.3,
            handlelength=1.2,
            labelspacing=0.22,
        )
        leg.set_zorder(10)
    ax.set_title(title, fontsize=7.5, fontweight="bold", pad=3)
    _axis_labels(ax, show_x, show_y)


def save_pub(fig, stem: Path):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=250, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_name(stem.name + "_preview").with_suffix(".jpg"), dpi=120, bbox_inches="tight", facecolor="white")
    print("Wrote", stem.with_suffix(".png"), flush=True)


def main():
    times = np.load(EVENT_DIR / "times_mid_valid.npy")
    meta = np.load(EVENT_DIR / "event_metadata.npz", allow_pickle=True)
    t_first = float(meta["t_first_trigger_s"])
    t_end = float(meta["effective_end_inf_s"])
    t_abs = float(meta["effective_end_abs_s"])
    if abs(t_end - t_abs) < 0.1:
        t_end = max(0.0, t_abs - t_first)
    valid = np.asarray(times) <= t_end
    idx = int(np.where(valid)[0][-1])
    t_eval = float(times[idx])
    print(f"idx={idx} t={t_eval:.3f}", flush=True)

    gt = np.load(MASKDIR / "gt_curved.npy")
    ft0, fd0 = load_clean(gt, idx)
    ft1 = np.load(MASKDIR / "sigma04_seed20261220_ft.npy")
    fd1 = np.load(MASKDIR / "sigma04_seed20261220_fd.npy")

    iou0, _f10, ft_d0 = score(ft0, gt)
    _iou_fd0, _f1, fd_d0 = score(fd0, gt)
    iou1, _f11, ft_d1 = score(ft1, gt)
    _iou_fd1, _f1b, fd_d1 = score(fd1, gt)
    print(f"σ=0  FT IoU={iou0:.3f} Δθ={ft_d0:.1f}  FinDer Δθ={fd_d0:.1f}", flush=True)
    print(f"σ=0.4 seed20  FT IoU={iou1:.3f} Δθ={ft_d1:.1f}  FinDer Δθ={fd_d1:.1f}", flush=True)

    epi = load_meta(EVENT_DIR)
    epi_lat, epi_lon = float(epi["eq_lat"]), float(epi["eq_lon"])
    lon_min, lon_max = epi_lon - HALF_DEG, epi_lon + HALF_DEG
    lat_min, lat_max = epi_lat - HALF_DEG, epi_lat + HALF_DEG
    rgb = load_dem_rgb(EVENT_DIR, lon_min, lon_max, lat_min, lat_max)
    art = dict(
        epi_lat=epi_lat, epi_lon=epi_lon,
        lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
        dem_rgb=rgb,
    )

    pga0 = np.load(MASKDIR / "pga_sigma0.npy")
    pga1 = np.load(MASKDIR / "pga_sigma04_seed20261220.npy")
    pgv0 = np.load(MASKDIR / "pgv_sigma0.npy")
    pgv1 = np.load(MASKDIR / "pgv_sigma04_seed20261220.npy")
    vmax_pga = float(np.nanpercentile(np.maximum(np.stack([pga0, pga1]), 0), 92))
    vmax_pgv = float(np.nanpercentile(np.maximum(np.stack([pgv0, pgv1]), 0), 92))
    vmax_pga = max(vmax_pga, 1e-3)
    vmax_pgv = max(vmax_pgv, 1e-3)
    print(f"vmax PGA={vmax_pga:.4g}  PGV={vmax_pgv:.4g}", flush=True)

    fig, axes = plt.subplots(2, 4, figsize=(14.4, 7.25), constrained_layout=True)
    im_pga = draw_field(
        axes[0, 0], art, pga0, r"PGA,  $\sigma=0$", "a", vmax=vmax_pga,
        show_x=False, show_y=True,
    )
    draw_field(
        axes[0, 1], art, pga1, r"PGA,  $\sigma=0.4$", "b", vmax=vmax_pga,
        show_x=False, show_y=False,
    )
    draw_panel(
        axes[0, 2], art, gt, fd0, "finder",
        r"FinDer,  $\sigma=0$", "c", True, show_x=False, show_y=False,
    )
    draw_panel(
        axes[0, 3], art, gt, fd1, "finder",
        r"FinDer,  $\sigma=0.4$", "d", False, show_x=False, show_y=False,
    )
    im_pgv = draw_field(
        axes[1, 0], art, pgv0, r"PGV,  $\sigma=0$", "e", vmax=vmax_pgv,
        show_x=True, show_y=True,
    )
    draw_field(
        axes[1, 1], art, pgv1, r"PGV,  $\sigma=0.4$", "f", vmax=vmax_pgv,
        show_x=True, show_y=False,
    )
    draw_panel(
        axes[1, 2], art, gt, ft0, "faulttrans",
        r"FaultTrans,  $\sigma=0$", "g", True, show_x=True, show_y=False,
    )
    draw_panel(
        axes[1, 3], art, gt, ft1, "faulttrans",
        r"FaultTrans,  $\sigma=0.4$", "h", False, show_x=True, show_y=False,
    )
    cb0 = fig.colorbar(im_pga, ax=axes[0, 0:2], shrink=0.88, pad=0.02)
    cb0.set_label(r"$\ln(1+\mathrm{PGA})$", fontsize=7)
    cb1 = fig.colorbar(im_pgv, ax=axes[1, 0:2], shrink=0.88, pad=0.02)
    cb1.set_label(r"$\ln(1+\mathrm{PGV})$", fontsize=7)
    save_pub(fig, FIGDIR / "envelope_diagnostic_2x4")
    plt.close(fig)
    print("[plot done]", FIGDIR, flush=True)


if __name__ == "__main__":
    main()
