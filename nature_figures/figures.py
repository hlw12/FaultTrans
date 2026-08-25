"""FaultTrans campaign figures, exported as individual small panels.

A 1x4 composite is saved as four files (figXX_a, figXX_b, ...), not one plate.
Does not edit the manuscript LaTeX.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Patch

from cache_gt import preferred_mask
from catalog import EVENTS, PAPER_SIX
from geo import (
    GRID_SIZE,
    RESOLUTION,
    compute_metrics,
    delta_theta,
    ellipse_major_angle,
    final_pred,
    increment_field,
    load_event_arrays,
    load_stations,
    overlap_rgb,
    parse_nied,
    parse_pku,
    rasterize_subfaults,
)
from style import EXPORT, PALETTE, SOURCE, save_panel

ORIGIN = "lower"
MAP_SIZE = 2.35
INFER_OUT = Path("/root/autodl-tmp/Test_on_Event/unified/outputs")


def load_infer_bundle(key, combine="max"):
    """Clock from the event folder; pred/fields from infer.py or infer_chichi.py."""
    ev = EVENTS[key]
    bundle = load_event_arrays(ev["pgv_dir"])
    sub = "infer_chichi" if combine == "geomean" else "infer_max"
    d = INFER_OUT / sub / key
    pred_p = d / "pred_binary.npy"
    if not pred_p.exists():
        raise FileNotFoundError(pred_p)
    pred = np.load(pred_p)
    if pred.shape[0] != len(bundle["t_axis"]):
        raise RuntimeError(f"{key}: infer pred T={pred.shape[0]} vs clock T={len(bundle['t_axis'])}")
    bundle["pred"] = pred
    fields_p = d / "pgv_fields.npy"
    if fields_p.exists():
        bundle["fields"] = np.load(fields_p)
    bundle["infer_dir"] = d
    bundle["combine"] = combine
    return bundle


def _box_image(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.4)
        s.set_color(PALETTE["neutral_light"])


def _legend_overlap(ax, loc="lower right"):
    handles = [
        Patch(facecolor=PALETTE["tp"], edgecolor="none", label="TP"),
        Patch(facecolor=PALETTE["fp"], edgecolor="none", label="FP"),
        Patch(facecolor=PALETTE["fn"], edgecolor="none", label="FN"),
    ]
    ax.legend(handles=handles, loc=loc, frameon=False, handlelength=0.9, borderaxespad=0.2)


def _new_map():
    fig, ax = plt.subplots(figsize=(MAP_SIZE, MAP_SIZE))
    return fig, ax


def _show_overlay(ax, pred, gt, title=None, star=True):
    ax.imshow(overlap_rgb(pred, gt, PALETTE), origin=ORIGIN, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    _box_image(ax)
    if star:
        ax.plot(74.5, 74.5, marker="*", color=PALETTE["gold"], markersize=7, markeredgecolor="k", markeredgewidth=0.3)
    if title:
        ax.set_title(title, pad=3)


def _inset_label(ax, text):
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.5,
        color=PALETTE["neutral_black"],
        linespacing=1.3,
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="none", alpha=0.88),
    )


def _metrics_line(pred, gt):
    m = compute_metrics(pred, gt)
    dth = delta_theta(ellipse_major_angle(pred), ellipse_major_angle(gt))
    return m, dth


def _save_overlay(stem, pred, gt, title=None, legend=False, inset=None):
    fig, ax = _new_map()
    _show_overlay(ax, pred, gt, title=title)
    if inset:
        _inset_label(ax, inset)
    if legend:
        _legend_overlap(ax, loc="lower left")
    fig.tight_layout()
    save_panel(fig, stem)


def _save_mask(stem, mask, title, cmap="gray_r"):
    fig, ax = _new_map()
    ax.imshow(mask, cmap=cmap, origin=ORIGIN, interpolation="nearest")
    ax.plot(74.5, 74.5, marker="*", color=PALETTE["gold"], markersize=7, markeredgecolor="k", markeredgewidth=0.3)
    _box_image(ax)
    ax.set_title(title, pad=3)
    fig.tight_layout()
    save_panel(fig, stem)


def fig01_six_event_gallery():
    """Six independent overlays, one file per event."""
    rows = []
    letters = "abcdef"
    for i, key in enumerate(PAPER_SIX):
        ev = EVENTS[key]
        bundle = load_event_arrays(ev["pgv_dir"])
        gt, gt_name = preferred_mask(key)
        pred = final_pred(bundle)
        m, dth = _metrics_line(pred, gt)
        title = f"{ev['label']}  $M_w$ {ev['mw']:.1f}\nIoU {m['IoU']:.2f}   $\\Delta\\theta$ {dth:.0f}$^\\circ$"
        _save_overlay(f"fig01_{letters[i]}_{key}", pred, gt, title, legend=(i == 0))
        rows.append(
            dict(
                event=key,
                label=ev["label"],
                gt=gt_name,
                IoU=m["IoU"],
                F1=m["F1"],
                Precision=m["Precision"],
                Recall=m["Recall"],
                delta_theta_deg=dth,
                paper_iou=ev["paper_iou"],
                paper_dtheta=ev["paper_dtheta"],
            )
        )
    pd.DataFrame(rows).to_csv(SOURCE / "fig01_six_event_metrics.csv", index=False)


def fig02_kumamoto_realtime():
    ev = EVENTS["kumamoto"]
    bundle = load_infer_bundle("kumamoto", combine="max")
    df = parse_nied(ev["nied_cv"])
    gt_static, _ = preferred_mask("kumamoto")
    pred = bundle["pred"]
    t = bundle["t_axis"]
    valid = bundle["valid"]
    iou_tv, dth = [], []
    t_first = bundle["t_first"]
    lat, lon = bundle["eq_lat"], bundle["eq_lon"]
    for k in range(len(t)):
        t_origin = float(t[k]) + t_first
        dfa = df[df["ttrg_s"] <= t_origin].copy()
        gt_t, _ = rasterize_subfaults(dfa, lat, lon)
        m = compute_metrics(pred[k], gt_t)
        iou_tv.append(m["IoU"])
        dth.append(delta_theta(ellipse_major_angle(pred[k]), ellipse_major_angle(gt_static)))
    iou_tv = np.asarray(iou_tv)
    dth = np.asarray(dth)
    best = int(np.nanargmax(np.where(valid, iou_tv, -1)))
    fractions = [0.25, 0.50, 0.75, 1.0]
    idx = np.where(valid)[0]
    frames = [int(idx[int(round((len(idx) - 1) * f))]) for f in fractions]
    m_final, dth_final = _metrics_line(pred[bundle["final_idx"]], gt_static)
    print(
        f"[fig02] infer_max/kumamoto  t_end={bundle['t_end']:.3f} final_idx={bundle['final_idx']} "
        f"static IoU={m_final['IoU']:.3f} Δθ={dth_final:.1f}  "
        f"IoUmax_tv={iou_tv[best]:.3f} @ {t[best]:.1f}s"
    )

    fig, ax = plt.subplots(figsize=(3.6, 2.45))
    (ln_iou,) = ax.plot(
        t[valid], iou_tv[valid], color=PALETTE["blue_main"], lw=1.4, label="IoU (left)"
    )
    ax.set_ylabel("IoU")
    ax.set_xlabel("Time since first trigger (s)")
    ax.set_ylim(0, 0.65)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    (ln_dth,) = ax2.plot(
        t[valid],
        dth[valid],
        color=PALETTE["fp"],
        lw=1.0,
        ls="--",
        label=r"$\Delta\theta$ (right)",
    )
    ax2.set_ylabel(r"$\Delta\theta$ ($^\circ$)")
    ln_best = ax.axvline(
        t[best],
        color=PALETTE["red_strong"],
        lw=0.8,
        label=r"$t_{\mathrm{best}}$",
    )
    (pt_max,) = ax.plot(
        t[best],
        iou_tv[best],
        "o",
        color=PALETTE["red_strong"],
        ms=3.5,
        label=r"IoU$_{\mathrm{max}}$",
        zorder=5,
    )
    ln_end = ax.axvline(
        bundle["t_end"],
        color=PALETTE["neutral_mid"],
        lw=0.7,
        ls=":",
        label=r"$t_{\mathrm{end}}$",
    )
    ax.legend(
        handles=[ln_iou, ln_dth, pt_max, ln_best, ln_end],
        loc="upper left",
        fontsize=6,
        frameon=False,
        handlelength=1.6,
        borderaxespad=0.3,
        labelspacing=0.25,
    )
    fig.tight_layout()
    save_panel(fig, "fig02_a_iou_curve")

    letters = "bcde"
    for j, fi in enumerate(frames):
        t_origin = float(t[fi]) + t_first
        dfa = df[df["ttrg_s"] <= t_origin].copy()
        gt_t, _ = rasterize_subfaults(dfa, lat, lon)
        _save_overlay(
            f"fig02_{letters[j]}_t{int(round(float(t[fi])))}s",
            pred[fi],
            gt_t,
            title=None,
            legend=(j == 3),
            inset=(
                f"{t[fi]:.0f} s\n"
                f"IoU {iou_tv[fi]:.2f}\n"
                rf"$\Delta\theta$ {dth[fi]:.1f}$^\circ$"
            ),
        )
    pd.DataFrame({"t_s": t, "iou_tv": iou_tv, "dtheta": dth, "valid": valid.astype(int)}).to_csv(
        SOURCE / "fig02_kumamoto_curves.csv", index=False
    )
    _fig02_realtime_fields(bundle, frames, t)


def _fig02_field_frames(bundle):
    t = bundle["t_axis"]
    idx = np.where(bundle["valid"])[0]
    frames = [int(idx[int(round((len(idx) - 1) * f))]) for f in (0.25, 0.50, 0.75, 1.0)]
    return frames, t


def fig02_realtime_fields_only():
    """PGV / ΔPGV maps only; does not touch overlay panels or pic/."""
    bundle = load_infer_bundle("kumamoto", combine="max")
    frames, t = _fig02_field_frames(bundle)
    _fig02_realtime_fields(bundle, frames, t)


def _save_standalone_cbar(cmap, vmax, stem, subdir, label):
    """Vertical colorbar only; same log1p scale as the field maps."""
    fig = plt.figure(figsize=(0.55, MAP_SIZE))
    ax = fig.add_axes([0.45, 0.08, 0.28, 0.84])
    sm = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(0.0, np.log1p(vmax)),
        cmap=cmap,
    )
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax, orientation="vertical")
    cb.ax.tick_params(labelsize=6)
    cb.set_label(label, fontsize=6)
    save_panel(fig, stem, subdir=subdir)


def _grid_lonlat(bundle):
    """1.5° FaultTrans grid centred on the epicentre."""
    half = GRID_SIZE * RESOLUTION * 0.5
    lon0, lat0 = float(bundle["eq_lon"]), float(bundle["eq_lat"])
    return lon0 - half, lon0 + half, lat0 - half, lat0 + half, lon0, lat0


def _lonlat_axes(ax, lon_min, lon_max, lat_min, lat_max):
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    ax.tick_params(axis="both", which="major", length=2.2, width=0.5, pad=1.5, labelsize=6, color=PALETTE["neutral_dark"])
    ax.set_xlabel("Longitude (°E)", fontsize=7, labelpad=1.5)
    ax.set_ylabel("Latitude (°N)", fontsize=7, labelpad=1.5)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.45)
        s.set_color(PALETTE["neutral_dark"])


def _fig02_realtime_fields(bundle, frames, t):
    """Cumulative PGV and ΔPGV at the same four times as fig02 overlay panels.

    Maps have no station markers and no colorbar. One shared bar is saved
    per quantity in exports/pgv and exports/dpgv.
    """
    fields = bundle.get("fields")
    if fields is None:
        raise RuntimeError("infer_max/kumamoto pgv_fields.npy missing")
    cmap = "YlOrBr"
    cbar_label = r"$\ln(1+\mathrm{cm\,s}^{-1})$"
    letters = "bcde"
    lon_min, lon_max, lat_min, lat_max, lon0, lat0 = _grid_lonlat(bundle)
    vmax_pgv = float(np.nanpercentile(np.maximum(fields[frames], 0), 99))
    dpgv_stack = np.stack([increment_field(fields, fi) for fi in frames], axis=0)
    vmax_dpgv = float(np.nanpercentile(dpgv_stack, 99))
    vmax_pgv = max(vmax_pgv, 1e-3)
    vmax_dpgv = max(vmax_dpgv, 1e-3)

    def _one(stem, img, vmax, subdir):
        fig, ax = plt.subplots(figsize=(MAP_SIZE + 0.55, MAP_SIZE + 0.42))
        ax.imshow(
            np.log1p(np.maximum(img, 0)),
            origin=ORIGIN,
            cmap=cmap,
            interpolation="nearest",
            vmin=0,
            vmax=np.log1p(vmax),
            extent=[lon_min, lon_max, lat_min, lat_max],
        )
        ax.plot(
            lon0,
            lat0,
            marker="*",
            color=PALETTE["gold"],
            markersize=7,
            markeredgecolor="k",
            markeredgewidth=0.3,
        )
        _lonlat_axes(ax, lon_min, lon_max, lat_min, lat_max)
        fig.tight_layout()
        save_panel(fig, stem, subdir=subdir)

    _save_standalone_cbar(cmap, vmax_pgv, "fig02_pgv_colorbar", "pgv", cbar_label)
    _save_standalone_cbar(cmap, vmax_dpgv, "fig02_dpgv_colorbar", "dpgv", cbar_label)
    for j, fi in enumerate(frames):
        tag = f"{letters[j]}_t{int(round(float(t[fi])))}s"
        pgv = np.maximum(fields[fi], 0.0)
        dpgv = increment_field(fields, fi)
        _one(f"fig02_pgv_{tag}", pgv, vmax_pgv, "pgv")
        _one(f"fig02_dpgv_{tag}", dpgv, vmax_dpgv, "dpgv")


def fig03_ridgecrest_models():
    """Ridgecrest vs JIN/ROSS/XU on an Esri topo base, matching Fig. 9 furniture."""
    from basemap import draw_topo_base, fetch_continuous_terrain, map_furniture
    from style import CACHE

    ev = EVENTS["ridgecrest"]
    try:
        bundle = load_infer_bundle("ridgecrest", combine="max")
    except (FileNotFoundError, RuntimeError):
        bundle = load_event_arrays(ev["pgv_dir"])
    pred = final_pred(bundle)
    half = GRID_SIZE * RESOLUTION * 0.5
    lon0, lat0 = float(bundle["eq_lon"]), float(bundle["eq_lat"])
    lon_min, lon_max = lon0 - half, lon0 + half
    lat_min, lat_max = lat0 - half, lat0 + half
    topo = fetch_continuous_terrain(lon_min, lon_max, lat_min, lat_max, zoom=11)
    n = pred.shape[0]
    dlon = lon_max - lon_min
    dlat = lat_max - lat_min

    def _px_ellipse(mask):
        """Same cv2.fitEllipse as Δθ, in pixel space, then mapped to lon/lat."""
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
        rows, cols = np.where(mask_u8)
        if len(rows) < 20:
            return None
        try:
            import cv2

            pts = np.column_stack([cols, rows]).astype(np.float32)
            (cx, cy), (w_px, h_px), ang = cv2.fitEllipse(pts)
        except Exception:
            return None
        lon = lon_min + float(cx) / (n - 1) * dlon
        lat = lat_min + float(cy) / (n - 1) * dlat
        return lon, lat, float(w_px) / (n - 1) * dlon, float(h_px) / (n - 1) * dlat, float(ang)

    def _show_mask(ax, mask, color, alpha=0.62):
        from matplotlib.colors import to_rgb

        filled = np.asarray(mask).astype(bool)
        if not filled.any():
            return
        rgb = np.asarray(to_rgb(color), dtype=np.float32)
        rgba = np.zeros(filled.shape + (4,), dtype=np.float32)
        rgba[..., :3] = rgb
        rgba[filled, 3] = alpha
        ax.imshow(
            rgba,
            extent=[lon_min, lon_max, lat_min, lat_max],
            origin=ORIGIN,
            interpolation="none",
            zorder=3,
            aspect="equal",
        )

    def _show_overlap(ax, pred_m, gt_m, alpha=0.62):
        pred_m = np.asarray(pred_m).astype(bool)
        gt_m = np.asarray(gt_m).astype(bool)
        _show_mask(ax, pred_m & gt_m, PALETTE["tp"], alpha)
        _show_mask(ax, pred_m & ~gt_m, PALETTE["fp"], alpha)
        _show_mask(ax, ~pred_m & gt_m, PALETTE["fn"], alpha)

    def _base(ax):
        draw_topo_base(ax, topo, lon_min, lon_max, lat_min, lat_max, veil=0.18)
        map_furniture(ax, lon_min, lon_max, lat_min, lat_max, scale_km=50.0)
        ax.plot(
            lon0,
            lat0,
            marker="*",
            color=PALETTE["gold"],
            markersize=10,
            markeredgecolor="#333333",
            markeredgewidth=0.5,
            zorder=7,
        )

    def _add_ellipse(ax, mask, color, lw=1.4):
        spec = _px_ellipse(mask)
        if spec is None:
            return
        lon, lat, w_deg, h_deg, ang = spec
        ax.add_patch(
            Ellipse(
                (lon, lat),
                w_deg,
                h_deg,
                angle=ang,
                fill=False,
                edgecolor=color,
                lw=lw,
                zorder=6,
            )
        )

    def _overlap_legend(ax):
        handles = [
            Patch(facecolor=PALETTE["tp"], edgecolor="none", label="TP"),
            Patch(facecolor=PALETTE["fp"], edgecolor="none", label="FP"),
            Patch(facecolor=PALETTE["fn"], edgecolor="none", label="FN"),
        ]
        leg = ax.legend(
            handles=handles,
            loc="lower right",
            fontsize=8,
            frameon=True,
            fancybox=False,
            edgecolor="#cccccc",
            facecolor="white",
            framealpha=0.92,
            borderpad=0.35,
            handlelength=1.1,
            labelspacing=0.3,
        )
        leg.set_zorder(10)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    _base(ax)
    _show_mask(ax, pred, PALETTE["tp"], alpha=0.70)
    _add_ellipse(ax, pred, PALETTE["gold"], lw=1.5)
    fig.tight_layout()
    save_panel(fig, "fig03_a_prediction")

    rows = []
    letters = "bcd"
    for i, key in enumerate(["JIN", "ROSS", "XU"]):
        gt = np.load(CACHE / f"ridgecrest_{key}.npy")
        m, dth = _metrics_line(pred, gt)
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        _base(ax)
        _show_overlap(ax, pred, gt)
        _add_ellipse(ax, pred, PALETTE["gold"], lw=1.4)
        _add_ellipse(ax, gt, PALETTE["teal"], lw=1.4)
        _inset_label(ax, f"vs {key}\nIoU {m['IoU']:.3f}\n$\\Delta\\theta$ {dth:.1f}$^\\circ$")
        if i == 2:
            _overlap_legend(ax)
        fig.tight_layout()
        save_panel(fig, f"fig03_{letters[i]}_vs_{key}")
        rows.append(
            dict(model=key, IoU=m["IoU"], F1=m["F1"], Precision=m["Precision"], Recall=m["Recall"], dtheta=dth)
        )
    pd.DataFrame(rows).to_csv(SOURCE / "fig03_ridgecrest_models.csv", index=False)


def fig04_ridgecrest_timing():
    """PKU order and FaultTrans growth on independent clocks; same terrain as Fig. 6."""
    from matplotlib.colors import to_rgb

    from basemap import draw_topo_base, fetch_continuous_terrain, map_furniture

    ev = EVENTS["ridgecrest"]
    try:
        bundle = load_infer_bundle("ridgecrest", combine="max")
    except (FileNotFoundError, RuntimeError):
        bundle = load_event_arrays(ev["pgv_dir"])
    df = parse_pku(ev["pku_txt"])
    pred = bundle["pred"]
    t = bundle["t_axis"]
    t_first = float(bundle["t_first"])
    lat, lon = float(bundle["eq_lat"]), float(bundle["eq_lon"])
    half = GRID_SIZE * RESOLUTION * 0.5
    lon_min, lon_max = lon - half, lon + half
    lat_min, lat_max = lat - half, lat + half
    topo = fetch_continuous_terrain(lon_min, lon_max, lat_min, lat_max, zoom=11)

    times_gt = [2.0, 4.0, 6.0, 9.0, 12.0]
    times_pred = [8.0, 12.0, 14.0, 16.0, 20.0]
    paper_gt = [
        "pku_timing_0_t0.0s.png",
        "pku_timing_1_t4.0s.png",
        "pku_timing_3_t12.0s.png",
        "pku_timing_4_t16.0s.png",
        "pku_timing_5_t20.0s.png",
    ]
    paper_pred = [
        "pred_timing_0_t0.0s.png",
        "pred_timing_4_t12.0s.png",
        "pred_timing_5_t13.0s.png",
        "pred_timing_6_t16.0s.png",
        "pred_timing_8_t20.0s.png",
    ]

    def _show_mask(ax, mask, color, alpha=0.70):
        filled = np.asarray(mask).astype(bool)
        if not filled.any():
            return
        rgb = np.asarray(to_rgb(color), dtype=np.float32)
        rgba = np.zeros(filled.shape + (4,), dtype=np.float32)
        rgba[..., :3] = rgb
        rgba[filled, 3] = alpha
        ax.imshow(
            rgba,
            extent=[lon_min, lon_max, lat_min, lat_max],
            origin=ORIGIN,
            interpolation="none",
            zorder=3,
            aspect="equal",
        )

    def _base(ax):
        draw_topo_base(ax, topo, lon_min, lon_max, lat_min, lat_max, veil=0.18)
        map_furniture(ax, lon_min, lon_max, lat_min, lat_max, scale_km=50.0)
        ax.plot(
            lon,
            lat,
            marker="*",
            color=PALETTE["gold"],
            markersize=10,
            markeredgecolor="#333333",
            markeredgewidth=0.5,
            zorder=7,
        )

    def _one(stem, mask, color):
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        _base(ax)
        _show_mask(ax, mask, color)
        fig.tight_layout()
        save_panel(fig, stem)

    rows = []
    for tt_gt, tt_pr, gt_name, pred_name in zip(times_gt, times_pred, paper_gt, paper_pred):
        dfa = df[np.isfinite(df["ttrg_s"]) & (df["ttrg_s"] <= tt_gt)].copy()
        gt_t, _ = rasterize_subfaults(dfa, lat, lon)
        k = int(np.argmin(np.abs(t - tt_pr)))
        k = min(k, int(bundle["final_idx"]))
        _one(f"fig04_gt_{int(tt_gt):02d}s", gt_t, PALETTE["fp"])
        _one(f"fig04_pred_{int(tt_pr):02d}s", pred[k], PALETTE["tp"])
        rows.append(
            dict(
                t_gt_s=tt_gt,
                t_pred_s=tt_pr,
                t_first_s=t_first,
                pred_idx=k,
                pred_t_s=float(t[k]),
                gt_pixels=int(np.asarray(gt_t).astype(bool).sum()),
                pred_pixels=int(np.asarray(pred[k]).astype(bool).sum()),
                paper_gt=gt_name,
                paper_pred=pred_name,
            )
        )
    pd.DataFrame(rows).to_csv(SOURCE / "fig04_ridgecrest_timing.csv", index=False)



def fig05_chichi():
    """Chi-Chi vs Chi on World Topo. z=10–11 stamps 'Map data not yet available' on Taiwan; z=9 does not."""
    from basemap import draw_topo_base, fetch_esri_topo, map_furniture
    from style import CACHE

    ev = EVENTS["chichi"]
    try:
        bundle = load_infer_bundle("chichi", combine="geomean")
    except (FileNotFoundError, RuntimeError):
        bundle = load_event_arrays(ev["pgv_dir"])
    pred = final_pred(bundle)
    gt = np.load(CACHE / "chichi_CHI.npy")
    half = GRID_SIZE * RESOLUTION * 0.5
    lon0, lat0 = float(bundle["eq_lon"]), float(bundle["eq_lat"])
    lon_min, lon_max = lon0 - half, lon0 + half
    lat_min, lat_max = lat0 - half, lat0 + half
    topo = fetch_esri_topo(lon_min, lon_max, lat_min, lat_max, zoom=9)
    n = pred.shape[0]
    dlon = lon_max - lon_min
    dlat = lat_max - lat_min

    def _px_ellipse(mask):
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
        rows, cols = np.where(mask_u8)
        if len(rows) < 20:
            return None
        try:
            import cv2

            pts = np.column_stack([cols, rows]).astype(np.float32)
            (cx, cy), (w_px, h_px), ang = cv2.fitEllipse(pts)
        except Exception:
            return None
        lon = lon_min + float(cx) / (n - 1) * dlon
        lat = lat_min + float(cy) / (n - 1) * dlat
        return lon, lat, float(w_px) / (n - 1) * dlon, float(h_px) / (n - 1) * dlat, float(ang)

    def _show_mask(ax, mask, color, alpha=0.62):
        from matplotlib.colors import to_rgb

        filled = np.asarray(mask).astype(bool)
        if not filled.any():
            return
        rgb = np.asarray(to_rgb(color), dtype=np.float32)
        rgba = np.zeros(filled.shape + (4,), dtype=np.float32)
        rgba[..., :3] = rgb
        rgba[filled, 3] = alpha
        ax.imshow(
            rgba,
            extent=[lon_min, lon_max, lat_min, lat_max],
            origin=ORIGIN,
            interpolation="none",
            zorder=3,
            aspect="equal",
        )

    def _show_overlap(ax, pred_m, gt_m, alpha=0.62):
        pred_m = np.asarray(pred_m).astype(bool)
        gt_m = np.asarray(gt_m).astype(bool)
        _show_mask(ax, pred_m & gt_m, PALETTE["tp"], alpha)
        _show_mask(ax, pred_m & ~gt_m, PALETTE["fp"], alpha)
        _show_mask(ax, ~pred_m & gt_m, PALETTE["fn"], alpha)

    def _base(ax):
        draw_topo_base(ax, topo, lon_min, lon_max, lat_min, lat_max, veil=0.30)
        map_furniture(ax, lon_min, lon_max, lat_min, lat_max, scale_km=50.0)
        ax.plot(
            lon0,
            lat0,
            marker="*",
            color=PALETTE["gold"],
            markersize=10,
            markeredgecolor="#333333",
            markeredgewidth=0.5,
            zorder=7,
        )

    def _add_ellipse(ax, mask, color, lw=1.4):
        spec = _px_ellipse(mask)
        if spec is None:
            return
        lon, lat, w_deg, h_deg, ang = spec
        ax.add_patch(
            Ellipse(
                (lon, lat),
                w_deg,
                h_deg,
                angle=ang,
                fill=False,
                edgecolor=color,
                lw=lw,
                zorder=6,
            )
        )

    def _overlap_legend(ax):
        handles = [
            Patch(facecolor=PALETTE["tp"], edgecolor="none", label="TP"),
            Patch(facecolor=PALETTE["fp"], edgecolor="none", label="FP"),
            Patch(facecolor=PALETTE["fn"], edgecolor="none", label="FN"),
        ]
        leg = ax.legend(
            handles=handles,
            loc="lower right",
            fontsize=8,
            frameon=True,
            fancybox=False,
            edgecolor="#cccccc",
            facecolor="white",
            framealpha=0.92,
            borderpad=0.35,
            handlelength=1.1,
            labelspacing=0.3,
        )
        leg.set_zorder(10)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    _base(ax)
    _show_mask(ax, pred, PALETTE["tp"], alpha=0.70)
    _add_ellipse(ax, pred, PALETTE["gold"], lw=1.5)
    fig.tight_layout()
    save_panel(fig, "fig05_a_prediction")

    m, dth = _metrics_line(pred, gt)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    _base(ax)
    _show_overlap(ax, pred, gt)
    _add_ellipse(ax, pred, PALETTE["gold"], lw=1.4)
    _add_ellipse(ax, gt, PALETTE["teal"], lw=1.4)
    _inset_label(ax, f"vs Chi\nIoU {m['IoU']:.3f}\n$\\Delta\\theta$ {dth:.1f}$^\\circ$")
    _overlap_legend(ax)
    fig.tight_layout()
    save_panel(fig, "fig05_b_vs_chi")
    pd.DataFrame(
        [dict(model="CHI", IoU=m["IoU"], F1=m["F1"], Precision=m["Precision"], Recall=m["Recall"], dtheta=dth)]
    ).to_csv(SOURCE / "fig05_chichi.csv", index=False)


def fig06_pga_vs_pgv():
    keys = ["kumamoto", "Miyagi", "Tottori", "ridgecrest", "chichi", "Menyuan"]
    show = ["kumamoto", "Miyagi", "Tottori", "chichi"]
    rows = []
    letters = "abcdefgh"
    n = 0
    for key in show:
        ev = EVENTS[key]
        gt, _ = preferred_mask(key)
        pgv = load_event_arrays(ev["pgv_dir"])
        pga = load_event_arrays(ev["pga_dir"])
        pred_v = final_pred(pgv)
        pred_a = final_pred(pga)
        mv, _ = _metrics_line(pred_v, gt)
        ma, _ = _metrics_line(pred_a, gt)
        _save_overlay(
            f"fig06_{letters[n]}_{key}_pgv",
            pred_v,
            gt,
            title=f"{ev['label']}\nPGV  IoU {mv['IoU']:.2f}",
            legend=(n == 0),
        )
        n += 1
        _save_overlay(
            f"fig06_{letters[n]}_{key}_pga",
            pred_a,
            gt,
            title=f"{ev['label']}\nPGA  IoU {ma['IoU']:.2f}",
        )
        n += 1

    labels, iou_v, iou_a = [], [], []
    for key in keys:
        ev = EVENTS[key]
        gt, gt_name = preferred_mask(key)
        pgv = load_event_arrays(ev["pgv_dir"])
        pga = load_event_arrays(ev["pga_dir"])
        mv, dv = _metrics_line(final_pred(pgv), gt)
        ma, da = _metrics_line(final_pred(pga), gt)
        labels.append(ev["label"].split()[0])
        iou_v.append(mv["IoU"])
        iou_a.append(ma["IoU"])
        rows.append(
            dict(
                event=key,
                gt=gt_name,
                pgv_iou=mv["IoU"],
                pga_iou=ma["IoU"],
                pgv_dtheta=dv,
                pga_dtheta=da,
                pgv_f1=mv["F1"],
                pga_f1=ma["F1"],
            )
        )
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, iou_v, w, color=PALETTE["pgv"], label="PGV")
    ax.bar(x + w / 2, iou_a, w, color=PALETTE["pga"], label="PGA")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Static IoU")
    ax.set_ylim(0, 0.75)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_panel(fig, "fig06_i_iou_bars")
    pd.DataFrame(rows).to_csv(SOURCE / "fig06_pga_vs_pgv.csv", index=False)


def fig07_failures():
    ev = EVENTS["Tottori"]
    bundle = load_event_arrays(ev["pgv_dir"])
    gt, _ = preferred_mask("Tottori")
    pred = final_pred(bundle)
    lons, lats = load_stations(ev["h5"])
    half = 0.75
    lon0, lat0 = bundle["eq_lon"], bundle["eq_lat"]
    keep = (lons >= lon0 - half) & (lons <= lon0 + half) & (lats >= lat0 - half) & (lats <= lat0 + half)
    lons, lats = lons[keep], lats[keep]
    fig, ax = plt.subplots(figsize=(MAP_SIZE, MAP_SIZE))
    ax.set_aspect("equal")
    ax.set_xlim(lon0 - half, lon0 + half)
    ax.set_ylim(lat0 - half, lat0 + half)
    ax.scatter(lons, lats, s=10, c=PALETTE["blue_secondary"], zorder=3, linewidths=0)
    ax.plot(bundle["eq_lon"], bundle["eq_lat"], marker="*", color=PALETTE["gold"], ms=9, markeredgecolor="k", markeredgewidth=0.3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Tottori stations")
    fig.tight_layout()
    save_panel(fig, "fig07_a_tottori_stations")

    m, dth = _metrics_line(pred, gt)
    _save_overlay("fig07_b_tottori", pred, gt, title=f"Tottori  IoU {m['IoU']:.2f}\nRecall 1.00")

    evm = EVENTS["Miyagi"]
    bm = load_event_arrays(evm["pgv_dir"])
    gtm, _ = preferred_mask("Miyagi")
    pm = final_pred(bm)
    mm, dm = _metrics_line(pm, gtm)
    _save_overlay(
        "fig07_c_miyagi",
        pm,
        gtm,
        title=f"Iwate–Miyagi  IoU {mm['IoU']:.2f}\n$\\Delta\\theta$ {dm:.0f}°",
        legend=True,
    )
    pd.DataFrame(
        [
            dict(event="Tottori", **m, dtheta=dth),
            dict(event="Miyagi", **mm, dtheta=dm),
        ]
    ).to_csv(SOURCE / "fig07_failures.csv", index=False)


def fig08_finder():
    pairs = [
        ("kumamoto", Path("/root/autodl-tmp/Test_on_Event/kumamoto")),
        ("Tottori", Path("/root/autodl-tmp/Test_on_Event/Tottori")),
    ]
    rows = []
    letters = "abcd"
    k = 0
    for key, d in pairs:
        gt, _ = preferred_mask(key)
        bundle = load_event_arrays(EVENTS[key]["pgv_dir"])
        pred = final_pred(bundle)
        line_path = d / "finder_pred_line_mask.npy"
        line = np.load(line_path) if line_path.exists() else None
        if line is not None and line.ndim == 3:
            line = line[bundle["final_idx"]]
        fig, ax = _new_map()
        ax.imshow(np.full(gt.shape + (3,), 0.96), origin=ORIGIN)
        ax.contour(gt, levels=[0.5], colors=[PALETTE["fn"]], linewidths=1.0, origin=ORIGIN)
        if line is not None:
            ys, xs = np.where(np.asarray(line).astype(bool))
            if len(xs):
                ax.plot(xs, ys, ".", color=PALETTE["finder"], ms=1.2, mew=0)
        ax.plot(74.5, 74.5, marker="*", color=PALETTE["gold"], ms=6, markeredgecolor="k", markeredgewidth=0.25)
        _box_image(ax)
        ax.set_title(f"{EVENTS[key]['label']}\nFinDer / GT")
        fig.tight_layout()
        save_panel(fig, f"fig08_{letters[k]}_{key}_finder")
        k += 1
        m, dth = _metrics_line(pred, gt)
        _save_overlay(
            f"fig08_{letters[k]}_{key}_faulttrans",
            pred,
            gt,
            title=f"FaultTrans  IoU {m['IoU']:.2f}",
            legend=(k == 3),
        )
        k += 1
        rows.append(dict(event=key, ft_iou=m["IoU"], ft_dtheta=dth))
    pd.DataFrame(rows).to_csv(SOURCE / "fig08_finder.csv", index=False)


def fig09_synthetic():
    h5_path = Path("/root/autodl-tmp/data/data/events.h5")
    ids = ["event_00007", "event_00022"]
    with h5py.File(h5_path, "r") as f:
        for r, eid in enumerate(ids):
            g = f[eid]
            pgv = g["inputs"]["pgv_field"][:]
            lab = g["labels"]["rupture_grid"][:]
            T = pgv.shape[0]
            frames = np.linspace(0, T - 1, 6).astype(int)
            for j, fi in enumerate(frames):
                dpgv = pgv[fi] if fi == 0 else np.maximum(pgv[fi] - pgv[fi - 1], 0)
                fig, ax = _new_map()
                ax.imshow(dpgv, origin=ORIGIN, cmap="YlOrBr", interpolation="nearest")
                ax.contour(lab[fi] if lab.ndim == 3 else lab, levels=[0.5], colors=[PALETTE["blue_main"]], linewidths=0.6, origin=ORIGIN)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{eid.replace('event_', 'E')}  t{fi}")
                fig.tight_layout()
                save_panel(fig, f"fig09_{'ab'[r]}{j+1}_{eid}_t{fi}")


def fig10_gt_pipeline():
    from style import CACHE

    slip = np.load(CACHE / "kumamoto_cv_slip.npy")
    mask = np.load(CACHE / "kumamoto_cv.npy")
    fig, ax = _new_map()
    im = ax.imshow(slip, origin=ORIGIN, cmap="YlOrBr", interpolation="nearest")
    ax.plot(74.5, 74.5, marker="*", color=PALETTE["gold"], ms=7, markeredgecolor="k", markeredgewidth=0.3)
    _box_image(ax)
    ax.set_title("Slip (fraction of peak)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    save_panel(fig, "fig10_a_slip")
    _save_mask("fig10_b_binary", mask, r"Binary mask  (slip $\geq$ 10% of peak)")


ALL = [
    fig01_six_event_gallery,
    fig02_kumamoto_realtime,
    fig03_ridgecrest_models,
    fig04_ridgecrest_timing,
    fig05_chichi,
    fig06_pga_vs_pgv,
    fig07_failures,
    fig08_finder,
    fig09_synthetic,
    fig10_gt_pipeline,
]
