#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""2x2 Kumamoto Curved maps: clean σ=0 vs σ=0.4 seed 20261220.

Same native interpolators as run_sweep. Loads saved ε_k so the plotted
realisation is the sweep row, not a redraw. Isolated to envelope_noise_kumamoto.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import to_rgb

ROOT = Path("/root/autodl-tmp")
HERE = ROOT / "envelope_noise_kumamoto"
OUT = HERE / "outputs"
FIGDIR = OUT / "figures"
MASKDIR = OUT / "masks"
UNIFIED = ROOT / "Test_on_Event" / "unified"
EVENT_DIR = ROOT / "Test_on_Event" / "kumamoto"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UNIFIED))
sys.path.insert(0, str(ROOT / "CmpFinDer"))
sys.path.insert(0, str(ROOT / "nature_figures"))
sys.path.insert(0, str(HERE))

from fields import EVENTS, build_station_traces, envelopes, interpolate_env  # noqa: E402
from finder_native import build_japan  # noqa: E402
from infer import CKPT_PGV, GRID_SIZE, gt_masks, load_model, load_norm, run_finder, run_ft  # noqa: E402
from run_sweep import (  # noqa: E402
    gains_for,
    interpolate_finder,
    loc_key,
    pga_envelopes_finder,
    score_curved,
)
from regen_finder_static_panel import (  # noqa: E402
    FINDER_COLOR,
    FT_COLOR,
    GT_COLOR,
    HALF_DEG,
    WHITE_VEIL_ALPHA,
    load_dem_rgb,
    load_meta,
)

EXPECTED = {
    (0.0, 20260818): dict(iou=0.564, ft=11.3, fd=12.0),
    (0.4, 20261220): dict(iou=0.626, ft=8.5, fd=47.6),
}

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


def load_clean_masks(ft_packed, gt):
    """Table 1 / Table 2 masks already on disk; do not re-infer σ=0 on CPU."""
    from fields import strike_delta
    from geo import compute_metrics

    idx = int(ft_packed["final_idx"])
    ft_path = UNIFIED / "outputs" / "infer_max" / "kumamoto" / "pred_binary.npy"
    fd_path = EVENT_DIR / "finder_pred_line_mask.npy"
    ft = np.load(ft_path)
    fd = np.load(fd_path)
    if ft.ndim == 3:
        ft = ft[min(idx, ft.shape[0] - 1)]
    if fd.ndim == 3:
        fd = fd[min(idx, fd.shape[0] - 1)]
    m = compute_metrics(ft, gt)
    row = dict(
        iou=float(m["IoU"]),
        f1=float(m["F1"]),
        ft_dtheta=float(strike_delta(ft, gt)),
        finder_dtheta=float(strike_delta(fd, gt)),
    )
    print(
        f"[clean disk] IoU={row['iou']:.3f}  "
        f"FT Δθ={row['ft_dtheta']:.1f}  FinDer Δθ={row['finder_dtheta']:.1f}"
    )
    exp = EXPECTED[(0.0, 20260818)]
    if abs(row["iou"] - exp["iou"]) > 0.02 or abs(row["ft_dtheta"] - exp["ft"]) > 1.5:
        raise SystemExit(f"clean FT disk masks do not reprint Table 1: {row}")
    if abs(row["finder_dtheta"] - exp["fd"]) > 1.0:
        raise SystemExit(f"clean FinDer disk mask does not reprint Table 2: {row}")
    return ft.astype(np.uint8), fd.astype(np.uint8), row


def load_eps(sigma: float, seed: int) -> dict:
    path = OUT / "gains" / f"eps_sigma{sigma:.1f}_seed{int(seed)}.npz"
    z = np.load(path)
    keys = [
        loc_key(lon, lat)
        for lon, lat in zip(z["keys_lon"].tolist(), z["keys_lat"].tolist())
    ]
    return dict(zip(keys, np.asarray(z["eps"], np.float32).tolist()))


def infer_saved(ft_packed, pgv_env, fd_packed, pga_env, model, device, mean, std, gt, sigma, seed):
    print(f"[infer] load eps σ={sigma} seed={seed}", flush=True)
    eps_by_key = load_eps(sigma, seed)
    print("[infer] interpolate PGV", flush=True)
    pgv_f = interpolate_env(
        pgv_env * gains_for(ft_packed["stations"], eps_by_key)[None, :], ft_packed
    )
    print("[infer] interpolate PGA", flush=True)
    pga_f = interpolate_finder(
        pga_env * gains_for(fd_packed["stations"], eps_by_key)[None, :],
        fd_packed["stations"],
        fd_packed,
    )
    print("[infer] run_ft", flush=True)
    _c, binary = run_ft(pgv_f, ft_packed["times"], ft_packed, model, device, mean, std)
    print("[infer] run_finder", flush=True)
    _res, finder_masks, _pk = run_finder(pga_f, fd_packed)
    iou, f1, ft_d, fd_d = score_curved(binary, finder_masks, ft_packed["final_idx"], gt)
    idx = int(ft_packed["final_idx"])
    row = dict(iou=float(iou), f1=float(f1), ft_dtheta=float(ft_d), finder_dtheta=float(fd_d))
    print(
        f"  σ={sigma:.1f} seed={seed}  IoU={iou:.3f}  "
        f"FT Δθ={ft_d:.1f}  FinDer Δθ={fd_d:.1f}"
    )
    exp = EXPECTED[(sigma, seed)]
    if abs(row["iou"] - exp["iou"]) > 0.02:
        raise SystemExit(f"IoU mismatch vs sweep: {row['iou']:.3f} vs {exp['iou']}")
    if abs(row["ft_dtheta"] - exp["ft"]) > 1.5:
        raise SystemExit(f"FT Δθ mismatch: {row['ft_dtheta']:.1f} vs {exp['ft']}")
    if abs(row["finder_dtheta"] - exp["fd"]) > 1.0:
        raise SystemExit(f"FinDer Δθ mismatch: {row['finder_dtheta']:.1f} vs {exp['fd']}")
    return binary[idx].astype(np.uint8), finder_masks[idx].astype(np.uint8), row


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


def draw_panel(ax, art, mask_gt, pred_mask, kind, title, letter, show_legend):
    lon_min, lon_max = art["lon_min"], art["lon_max"]
    lat_min, lat_max = art["lat_min"], art["lat_max"]
    extent = [lon_min, lon_max, lat_min, lat_max]
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
    _overlay_mask(ax, mask_gt, GT_COLOR, 0.45, extent)
    if kind == "finder":
        _overlay_mask(ax, pred_mask, FINDER_COLOR, 0.95, extent)
    else:
        _overlay_mask(ax, pred_mask, FT_COLOR, 0.55, extent)
    ax.scatter(
        art["epi_lon"],
        art["epi_lat"],
        s=70,
        c="gold",
        marker="*",
        edgecolors="#333333",
        linewidths=0.5,
        zorder=5,
    )
    ax.add_patch(
        Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=0.9,
            edgecolor="#333333",
            facecolor="none",
            linestyle="--",
            zorder=4,
        )
    )
    ax.text(
        0.03,
        0.97,
        letter,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        fontweight="bold",
        color="#111111",
        zorder=12,
        bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
    )
    if show_legend:
        leg = ax.legend(
            handles=_legend_handles(kind),
            loc="upper left",
            bbox_to_anchor=(0.12, 0.98),
            fontsize=6.5,
            frameon=True,
            fancybox=False,
            edgecolor="#cccccc",
            facecolor="white",
            framealpha=0.92,
            borderpad=0.35,
            handlelength=1.4,
            labelspacing=0.28,
        )
        leg.set_zorder(10)
    ax.set_title(title, fontsize=8, fontweight="bold", pad=4)
    ax.tick_params(axis="x", labelsize=6, rotation=25)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xlabel("Longitude (°E)", fontsize=7)
    ax.set_ylabel("Latitude (°N)", fontsize=7)


def build_art():
    meta = load_meta(EVENT_DIR)
    epi_lat, epi_lon = float(meta["eq_lat"]), float(meta["eq_lon"])
    lon_min, lon_max = epi_lon - HALF_DEG, epi_lon + HALF_DEG
    lat_min, lat_max = epi_lat - HALF_DEG, epi_lat + HALF_DEG
    rgb = load_dem_rgb(EVENT_DIR, lon_min, lon_max, lat_min, lat_max)
    return dict(
        epi_lat=epi_lat,
        epi_lon=epi_lon,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        dem_rgb=rgb,
    )


def save_pub(fig, stem: Path):
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_name(stem.name + "_preview").with_suffix(".jpg"), dpi=160, bbox_inches="tight", facecolor="white")
    print("Wrote", stem.with_suffix(".png"))


def main():
    try:
        _main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise


def _main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    MASKDIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    mean, std = load_norm("pgv")
    model = load_model(CKPT_PGV, device)

    ft_packed = build_station_traces(EVENTS["kumamoto"])
    _pga_unused, pgv_env, _ = envelopes(ft_packed)
    fd_packed, _fd0 = build_japan("kumamoto")
    pga_env = pga_envelopes_finder(fd_packed["stations"], fd_packed["times"], fd_packed["t_first"])
    gt = gt_masks("kumamoto", ft_packed)["Curved"]
    t_eval = float(ft_packed["times"][ft_packed["final_idx"]])
    print(f"final_idx={ft_packed['final_idx']} t={t_eval:.3f}s")

    torch.set_num_threads(8)
    ft0, fd0, r0 = load_clean_masks(ft_packed, gt)
    ft1, fd1, r1 = infer_saved(
        ft_packed, pgv_env, fd_packed, pga_env, model, device, mean, std, gt, 0.4, 20261220
    )
    np.save(MASKDIR / "sigma0_ft.npy", ft0)
    np.save(MASKDIR / "sigma0_fd.npy", fd0)
    np.save(MASKDIR / "sigma04_seed20261220_ft.npy", ft1)
    np.save(MASKDIR / "sigma04_seed20261220_fd.npy", fd1)
    np.save(MASKDIR / "gt_curved.npy", gt.astype(np.uint8))

    art = build_art()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.05), constrained_layout=True)
    draw_panel(
        axes[0, 0],
        art,
        gt,
        fd0,
        "finder",
        rf"$\sigma=0$  FinDer  $\Delta\theta={r0['finder_dtheta']:.1f}^\circ$",
        "a",
        True,
    )
    draw_panel(
        axes[0, 1],
        art,
        gt,
        ft0,
        "faulttrans",
        rf"$\sigma=0$  FaultTrans  IoU ${r0['iou']:.3f}$  $\Delta\theta={r0['ft_dtheta']:.1f}^\circ$",
        "b",
        True,
    )
    draw_panel(
        axes[1, 0],
        art,
        gt,
        fd1,
        "finder",
        rf"$\sigma=0.4$  seed 20  FinDer  $\Delta\theta={r1['finder_dtheta']:.1f}^\circ$",
        "c",
        False,
    )
    draw_panel(
        axes[1, 1],
        art,
        gt,
        ft1,
        "faulttrans",
        rf"$\sigma=0.4$  seed 20  FaultTrans  IoU ${r1['iou']:.3f}$  $\Delta\theta={r1['ft_dtheta']:.1f}^\circ$",
        "d",
        False,
    )
    fig.suptitle(
        rf"Kumamoto Curved, last valid frame ($t={t_eval:.1f}$ s). "
        r"Panels c–d are one realisation, not the $\sigma=0.4$ mean.",
        fontsize=8,
        y=1.02,
    )
    save_pub(fig, FIGDIR / "envelope_diagnostic_2x2")
    plt.close(fig)

    for stem, mask, kind, title in (
        ("envelope_sigma0_finder", fd0, "finder", rf"$\sigma=0$  FinDer  $\Delta\theta={r0['finder_dtheta']:.1f}^\circ$"),
        ("envelope_sigma0_ft", ft0, "faulttrans", rf"$\sigma=0$  FaultTrans  IoU ${r0['iou']:.3f}$  $\Delta\theta={r0['ft_dtheta']:.1f}^\circ$"),
        ("envelope_sigma04_seed20_finder", fd1, "finder", rf"$\sigma=0.4$  seed 20  FinDer  $\Delta\theta={r1['finder_dtheta']:.1f}^\circ$"),
        ("envelope_sigma04_seed20_ft", ft1, "faulttrans", rf"$\sigma=0.4$  seed 20  FaultTrans  IoU ${r1['iou']:.3f}$  $\Delta\theta={r1['ft_dtheta']:.1f}^\circ$"),
    ):
        fig, ax = plt.subplots(figsize=(5.0, 4.9))
        draw_panel(ax, art, gt, mask, kind, title, "", True)
        fig.tight_layout()
        save_pub(fig, FIGDIR / stem)
        plt.close(fig)
    print("[plot done]", FIGDIR)


if __name__ == "__main__":
    main()
