#!/usr/bin/env python3
"""FinDer static maps with Esri World Topo basemap (same as record-section maps)."""
from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image

EVENTS_ROOT = Path("/root/autodl-tmp/Test_on_Event")
OUT = Path("/root/autodl-tmp/CmpFinDer/figures")
OUT.mkdir(parents=True, exist_ok=True)
EXPORTS = Path("/root/autodl-tmp/nature_figures/exports")
EXPORTS.mkdir(parents=True, exist_ok=True)
TILE_CACHE = Path("/root/autodl-tmp/nature_figures/tile_cache")
ESRI_TOPO = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Topo_Map/MapServer/tile/{z}/{y}/{x}.jpg"
)
TILE_PX = 256
TOPO_ZOOM = 10
WHITE_VEIL_ALPHA = 0.30

GRID_SIZE = 150
RESOLUTION = 0.01
HALF_DEG = 0.75
MIN_PIXELS_FOR_ANGLE = 20

# Orange GT stays distinct from red FinDer line on the beige/green topo map
GT_COLOR = "#e6550d"
FINDER_COLOR = "#d62728"
FT_COLOR = "#2A9D8F"
INFER_MAX = Path("/root/autodl-tmp/Test_on_Event/unified/outputs/infer_max")

import sys

sys.path.insert(0, "/root/autodl-tmp/nature_figures")
from geo import delta_theta, ellipse_major_angle  # noqa: E402


def mask_major_axis_angle(binary_mask, min_pixels=MIN_PIXELS_FOR_ANGLE):
    ang = ellipse_major_angle(binary_mask, min_pixels=min_pixels)
    if not np.isfinite(ang):
        ang = ellipse_major_angle(binary_mask, min_pixels=5)
    return None if not np.isfinite(ang) else float(ang)


def angle_difference(a, b):
    if a is None or b is None:
        return None
    d = abs(float(a) - float(b)) % 180.0
    return float(min(d, 180.0 - d))


def _fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{float(v):.1f}"


def load_meta(event_dir: Path):
    meta_path = event_dir / "event_metadata.npz"
    if meta_path.exists():
        z = np.load(meta_path, allow_pickle=True)
        meta = {k: z[k].item() if getattr(z[k], "shape", ()) == () else z[k] for k in z.files}
        if "eq_lat" in meta:
            return {"eq_lat": float(meta["eq_lat"]), "eq_lon": float(meta["eq_lon"])}
        if "latitude" in meta:
            return {"eq_lat": float(meta["latitude"]), "eq_lon": float(meta["longitude"])}
    h5s = sorted(event_dir.glob("*.h5"))
    with h5py.File(h5s[0], "r") as f:
        return {
            "eq_lat": float(f["earthquake"].attrs["latitude"]),
            "eq_lon": float(f["earthquake"].attrs["longitude"]),
        }


def _lonlat_to_global_px(lon, lat, zoom):
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n * TILE_PX
    lat = max(min(lat, 85.05112878), -85.05112878)
    siny = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * n * TILE_PX
    return x, y


def _fetch_tile(z, x, y, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / f"esri_topo_{z}_{x}_{y}.jpg"
    if fp.exists() and fp.stat().st_size > 800:
        return Image.open(fp).convert("RGB")
    url = ESRI_TOPO.format(z=z, x=x, y=y)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 finder-static-plot"})
    with urlopen(req, timeout=40) as resp:
        data = resp.read()
    fp.write_bytes(data)
    return Image.open(BytesIO(data)).convert("RGB")


def fetch_esri_topo(lon_min, lon_max, lat_min, lat_max, zoom=TOPO_ZOOM, cache_dir=TILE_CACHE):
    x0, y0 = _lonlat_to_global_px(lon_min, lat_max, zoom)
    x1, y1 = _lonlat_to_global_px(lon_max, lat_min, zoom)
    tx0, ty0 = int(x0 // TILE_PX), int(y0 // TILE_PX)
    tx1, ty1 = int(x1 // TILE_PX), int(y1 // TILE_PX)
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * TILE_PX, (ty1 - ty0 + 1) * TILE_PX))
    for iy, ty in enumerate(range(ty0, ty1 + 1)):
        for ix, tx in enumerate(range(tx0, tx1 + 1)):
            mosaic.paste(_fetch_tile(zoom, tx, ty, cache_dir), (ix * TILE_PX, iy * TILE_PX))
    cropped = mosaic.crop(
        (
            int(math.floor(x0 - tx0 * TILE_PX)),
            int(math.floor(y0 - ty0 * TILE_PX)),
            int(math.ceil(x1 - tx0 * TILE_PX)),
            int(math.ceil(y1 - ty0 * TILE_PX)),
        )
    )
    return np.asarray(cropped).astype(np.float32) / 255.0


def load_dem_rgb(event_dir: Path, lon_min, lon_max, lat_min, lat_max):
    """Esri World Topo; DEM hillshade without blue ocean is the fallback."""
    try:
        rgb = fetch_esri_topo(lon_min, lon_max, lat_min, lat_max)
        print(f"  Esri World Topo OK  {event_dir.name}  shape={rgb.shape}")
        return rgb
    except Exception as e:
        print(f"  [Esri fallback -> DEM] {event_dir.name}: {e}")

    import rasterio
    from matplotlib.colors import LightSource, LinearSegmentedColormap
    from rasterio.windows import from_bounds

    dems = sorted(event_dir.glob("*srtm*.tif")) + sorted(event_dir.glob("*dem*.tif"))
    if not dems:
        raise FileNotFoundError(f"no DEM in {event_dir}")
    land_cmap = LinearSegmentedColormap.from_list(
        "land_navyfree",
        ["#efe8d2", "#d5e0b0", "#8aaa62", "#c4a056", "#8a5a32", "#f5f5f5"],
    )
    with rasterio.open(dems[0]) as src:
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, transform=src.transform)
        win = window.round_offsets().round_lengths()
        h = min(900, max(int(win.height), 1))
        w = min(900, max(int(win.width), 1))
        dem = src.read(
            1,
            window=win,
            out_shape=(h, w),
            resampling=rasterio.enums.Resampling.bilinear,
        ).astype(float)
        nodata = src.nodata
        if nodata is not None:
            dem[dem == nodata] = np.nan
        finite = dem[np.isfinite(dem)]
        fill = float(np.nanmedian(finite)) if finite.size else 0.0
        dem_filled = np.where(np.isfinite(dem), dem, fill)
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(
        dem_filled,
        cmap=land_cmap,
        blend_mode="overlay",
        vert_exag=1.5,
        dx=1,
        dy=1,
    )
    rgb = np.asarray(rgb[..., :3], dtype=np.float32)
    ocean = ~np.isfinite(dem) | (dem <= 0)
    rgb[ocean] = np.array([0.93, 0.91, 0.84], dtype=np.float32)
    return rgb


def eval_final_idx(event_dir: Path):
    meta = np.load(event_dir / "event_metadata.npz", allow_pickle=True)
    times_path = event_dir / "times_mid_valid.npy"
    times = np.load(times_path) if times_path.exists() else np.load(event_dir / "finder_times_mid.npy")
    t_first = float(meta["t_first_trigger_s"])
    t_end = float(meta["effective_end_inf_s"])
    t_abs = float(meta["effective_end_abs_s"])
    if abs(t_end - t_abs) < 0.1:
        t_end = max(0.0, t_abs - t_first)
    valid = np.asarray(times) <= t_end
    if not np.any(valid):
        valid = np.ones(len(times), dtype=bool)
    final_idx = int(np.where(valid)[0][-1])
    return final_idx, np.asarray(times), t_end


def load_art(event_name: str):
    event_dir = EVENTS_ROOT / event_name
    pred = np.load(event_dir / "finder_pred_line_mask.npy")
    mask_rp = np.load(event_dir / "finder_gt_mask_rp.npy")
    cv_path = event_dir / "finder_gt_mask_cv.npy"
    mask_cv = np.load(cv_path) if cv_path.exists() else None
    final_idx, times, t_end = eval_final_idx(event_dir)
    if pred.ndim == 3:
        final_idx = min(final_idx, pred.shape[0] - 1)
        mask_finder = pred[final_idx]
    else:
        mask_finder = pred
    t_final = float(times[min(final_idx, len(times) - 1)])
    theta_gt_rp = mask_major_axis_angle(mask_rp)
    theta_gt_cv = mask_major_axis_angle(mask_cv) if mask_cv is not None else None
    theta_p = mask_major_axis_angle(mask_finder)
    dtheta_rp = angle_difference(theta_p, theta_gt_rp)
    dtheta_cv = angle_difference(theta_p, theta_gt_cv)
    print(
        f"[{event_name}] final_idx={final_idx} t={t_final:.3f} t_end={t_end:.3f} "
        f"line_px={int(np.asarray(mask_finder).astype(bool).sum())} "
        f"Δθ_RP={_fmt(dtheta_rp)} Δθ_CV={_fmt(dtheta_cv)}"
    )
    meta = load_meta(event_dir)
    epi_lat, epi_lon = meta["eq_lat"], meta["eq_lon"]
    lon_min, lon_max = epi_lon - HALF_DEG, epi_lon + HALF_DEG
    lat_min, lat_max = epi_lat - HALF_DEG, epi_lat + HALF_DEG
    rgb = load_dem_rgb(event_dir, lon_min, lon_max, lat_min, lat_max)
    print(f"[{event_name}] DEM rgb {rgb.shape}  epi=({epi_lat:.3f},{epi_lon:.3f})")

    return {
        "event_name": event_name,
        "mask_rp": mask_rp,
        "mask_cv": mask_cv,
        "mask_finder": mask_finder,
        "theta_gt_rp": theta_gt_rp,
        "theta_gt_cv": theta_gt_cv,
        "dtheta_rp": dtheta_rp,
        "dtheta_cv": dtheta_cv,
        "t_final": t_final,
        "epi_lat": epi_lat,
        "epi_lon": epi_lon,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "dem_rgb": rgb,
        "final_idx": final_idx,
        "times": times,
    }


def load_ft_mask(event_name: str, final_idx: int, times: np.ndarray):
    """Committed map at the same last-valid-frame index as Table 1."""
    infer_p = INFER_MAX / event_name / "pred_binary.npy"
    live_p = EVENTS_ROOT / event_name / "pred_binary.npy"
    path = infer_p if infer_p.exists() else live_p
    pred = np.load(path)
    idx = min(int(final_idx), pred.shape[0] - 1)
    t_final = float(times[min(idx, len(times) - 1)])
    print(f"[{event_name}] FaultTrans pred={path} idx={idx} t={t_final:.3f} px={int(pred[idx].astype(bool).sum())}")
    return pred[idx].astype(np.uint8), t_final


def _overlay_mask(ax, mask, color, alpha, extent):
    """Mask grid: row0=south → origin='lower' with geographic extent."""
    m = (np.asarray(mask) > 0).astype(np.float32)
    rgba = np.zeros((*m.shape, 4), dtype=np.float32)
    rgba[m > 0] = (*to_rgb(color), alpha)
    ax.imshow(
        rgba,
        extent=extent,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        zorder=3,
    )


def _legend_handles(kind="finder"):
    pred = (
        Line2D([0], [0], color=FINDER_COLOR, lw=2.2, label="FinDer line")
        if kind == "finder"
        else Patch(facecolor=FT_COLOR, edgecolor="white", linewidth=0.5, alpha=0.7, label="FaultTrans")
    )
    return [
        Patch(facecolor=GT_COLOR, edgecolor="white", linewidth=0.5, alpha=0.7, label="GT mask"),
        pred,
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gold",
            markeredgecolor="#333333",
            markersize=12,
            linestyle="None",
            label="Epicenter",
        ),
    ]


def draw_map_panel(ax, art, mask_gt, title, pred_mask=None, kind="finder"):
    """Geographic panel: World Topo + GT + FinDer line or FaultTrans footprint."""
    lon_min, lon_max = art["lon_min"], art["lon_max"]
    lat_min, lat_max = art["lat_min"], art["lat_max"]
    extent = [lon_min, lon_max, lat_min, lat_max]

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal")

    ax.imshow(
        art["dem_rgb"],
        extent=extent,
        origin="upper",
        aspect="equal",
        zorder=0,
        interpolation="bilinear",
    )
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
        _overlay_mask(ax, art["mask_finder"] if pred_mask is None else pred_mask, FINDER_COLOR, 0.95, extent)
    else:
        _overlay_mask(ax, pred_mask, FT_COLOR, 0.55, extent)

    ax.scatter(
        art["epi_lon"],
        art["epi_lat"],
        s=90,
        c="gold",
        marker="*",
        edgecolors="#333333",
        linewidths=0.6,
        zorder=5,
    )

    ax.add_patch(
        Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=1.2,
            edgecolor="#333333",
            facecolor="none",
            linestyle="--",
            zorder=4,
        )
    )

    ax.annotate(
        "N",
        xy=(0.92, 0.90),
        xytext=(0.92, 0.78),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
        color="#333333",
        zorder=6,
    )

    scale_km = 50.0
    scale_deg = scale_km / 111.0
    bar_x0 = lon_min + 0.06
    bar_y0 = lat_min + 0.05
    ax.plot(
        [bar_x0, bar_x0 + scale_deg],
        [bar_y0, bar_y0],
        color="#333333",
        lw=2.0,
        solid_capstyle="butt",
        zorder=6,
    )
    ax.text(
        bar_x0 + scale_deg / 2,
        bar_y0 + 0.025,
        f"{scale_km:.0f} km",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="#333333",
        zorder=6,
    )

    # in-axes legend: upper-left, white background
    leg = ax.legend(
        handles=_legend_handles(kind),
        loc="upper left",
        fontsize=8,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        facecolor="white",
        framealpha=0.92,
        borderpad=0.4,
        handlelength=1.6,
        labelspacing=0.35,
    )
    leg.set_zorder(10)

    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=8)
    for lab in ax.get_xticklabels():
        lab.set_fontweight("bold")
        lab.set_ha("right")
    for lab in ax.get_yticklabels():
        lab.set_fontweight("bold")
    ax.set_xlabel("Longitude (°E)", fontsize=8, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=8, fontweight="bold")


def save_single_map(art, mask_gt, title, stem, pred_mask=None, kind="finder"):
    """Save one standalone map (png + pdf)."""
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    draw_map_panel(ax, art, mask_gt, title=title, pred_mask=pred_mask, kind=kind)
    fig.tight_layout()

    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(EXPORTS / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", png)
    print("Wrote", pdf)
    print("Wrote", EXPORTS / f"{stem}.png")


def main():
    k = load_art("kumamoto")
    t = load_art("Tottori")
    k_ft, k_t = load_ft_mask("kumamoto", k["final_idx"], k["times"])
    t_ft, t_t = load_ft_mask("Tottori", t["final_idx"], t["times"])
    k_ft_rp = float(delta_theta(ellipse_major_angle(k_ft), k["theta_gt_rp"]))
    k_ft_cv = float(delta_theta(ellipse_major_angle(k_ft), k["theta_gt_cv"]))
    t_ft_rp = float(delta_theta(ellipse_major_angle(t_ft), t["theta_gt_rp"]))
    print(f"[kumamoto] FaultTrans Δθ RP={k_ft_rp:.1f} Curved={k_ft_cv:.1f}")
    print(f"[Tottori] FaultTrans Δθ RP={t_ft_rp:.1f}")

    save_single_map(
        k,
        k["mask_rp"],
        title=(
            f"Kumamoto (dense)  RP\n"
            f"θ={_fmt(k['theta_gt_rp'])}°  |  Δθ={_fmt(k['dtheta_rp'])}°  |  "
            f"t={k['t_final']:.1f} s"
        ),
        stem="finder_static_kumamoto_rp",
    )
    save_single_map(
        k,
        k["mask_cv"],
        title=(
            f"Kumamoto (dense)  Curved\n"
            f"θ={_fmt(k['theta_gt_cv'])}°  |  Δθ={_fmt(k['dtheta_cv'])}°  |  "
            f"t={k['t_final']:.1f} s"
        ),
        stem="finder_static_kumamoto_curved",
    )
    save_single_map(
        t,
        t["mask_rp"],
        title=(
            f"Tottori (off-network)  RP\n"
            f"θ={_fmt(t['theta_gt_rp'])}°  |  Δθ={_fmt(t['dtheta_rp'])}°  |  "
            f"t={t['t_final']:.1f} s"
        ),
        stem="finder_static_tottori_rp",
    )
    save_single_map(
        k,
        k["mask_rp"],
        title=(
            f"Kumamoto (dense)  RP\n"
            f"FaultTrans  |  Δθ={k_ft_rp:.1f}°  |  t={k_t:.1f} s"
        ),
        stem="ft_static_kumamoto_rp",
        pred_mask=k_ft,
        kind="faulttrans",
    )
    save_single_map(
        k,
        k["mask_cv"],
        title=(
            f"Kumamoto (dense)  Curved\n"
            f"FaultTrans  |  Δθ={k_ft_cv:.1f}°  |  t={k_t:.1f} s"
        ),
        stem="ft_static_kumamoto_curved",
        pred_mask=k_ft,
        kind="faulttrans",
    )
    save_single_map(
        t,
        t["mask_rp"],
        title=(
            f"Tottori (off-network)  RP\n"
            f"FaultTrans  |  Δθ={t_ft_rp:.1f}°  |  t={t_t:.1f} s"
        ),
        stem="ft_static_tottori_rp",
        pred_mask=t_ft,
        kind="faulttrans",
    )


if __name__ == "__main__":
    main()