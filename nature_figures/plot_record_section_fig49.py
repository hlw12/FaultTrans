#!/usr/bin/env python
"""Regenerate paper Fig.4 (Kumamoto) and Fig.9 (Tottori) record sections.

Changes vs the original record_section.ipynb cell 6:
  1. Drop the cyan S-wave-coverage vertical line on the left panel.
  2. Replace the matplotlib `terrain` DEM (blue ocean) with Esri World Topo
     tiles, matching kumamoto_fault_comparison.ipynb, so K-NET blue dots
     no longer collide with the basemap.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image
from scipy.signal import butter, filtfilt

# ---------------------------------------------------------------------------
# Event list
# ---------------------------------------------------------------------------
EVENTS = [
    {
        "h5": "/root/autodl-tmp/Test_on_Event/kumamoto/kumamoto.h5",
        "title": "Kumamoto",
        "out_name": "record_section_kumamoto.png",
    },
    {
        "h5": "/root/autodl-tmp/Test_on_Event/Tottori/Tottori.h5",
        "title": "Tottori",
        "out_name": "record_section_Tottori.png",
    },
]

COMPONENT = "UD"
HALF_DEG = 0.75
T_WINDOW = (-5, 60)
SCALE = 0.8
FREQ_BAND = (1.0, 10.0)
MANUAL_ORIGIN_TIMES = {
    "kumamoto": "2016-04-16T01:25:05+09:00",
    "Miyagi": "2008-06-14T08:43:45+09:00",
    "Tottori": "2016-10-21T14:07:22+09:00",
}
RECORD_TIME_IS_TRIGGER = True
RECORD_PRETRIGGER_S = 15.0
ASSUME_JST = True
PHASENET_PROB_THRESHOLD = 0.3
V_P_KM_S = 6.0
T_P_TOLERANCE_S = 3.0
FAULT_LENGTH_A = 0.59
FAULT_LENGTH_B = -2.44
RUPTURE_VELOCITY_KM_S = 3.0
PHYSICAL_END_BUFFER_S = 4.0

OUT_DIR = Path("/root/autodl-tmp/nature_figures/exports")
TILE_CACHE = Path("/root/autodl-tmp/nature_figures/tile_cache")
ESRI_TOPO = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Topo_Map/MapServer/tile/{z}/{y}/{x}.jpg"
)
TILE_PX = 256
TOPO_ZOOM = 10
WHITE_VEIL_ALPHA = 0.30  # same wash as kumamoto_fault_comparison.ipynb


# ---------------------------------------------------------------------------
# Helpers copied from record_section.ipynb
# ---------------------------------------------------------------------------
def parse_time(s, assume_jst=False):
    s = str(s)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    if dt.tzinfo is None:
        tz = timezone(timedelta(hours=9)) if assume_jst else timezone.utc
        dt = dt.replace(tzinfo=tz)
    return dt


def bandpass(data, fs, fmin, fmax, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [fmin / nyq, fmax / nyq], btype="band")
    return filtfilt(b, a, data)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def load_picks(cache_path):
    df = pd.read_csv(cache_path, index_col="station_id")
    picks = {
        sid: (None if pd.isna(row["p_pick_s"]) else float(row["p_pick_s"]))
        for sid, row in df.iterrows()
    }
    print(f"Loaded picks from cache: {len(picks)} stations ({cache_path})")
    return picks, df


def load_records(h5_path, picks):
    records = []
    event_name = Path(h5_path).parent.name
    with h5py.File(h5_path, "r") as f:
        eq = f["earthquake"]
        epi_lat = float(eq.attrs["latitude"])
        epi_lon = float(eq.attrs["longitude"])
        mag = float(eq.attrs["magnitude"])
        if event_name in MANUAL_ORIGIN_TIMES:
            ori_time = parse_time(MANUAL_ORIGIN_TIMES[event_name])
        else:
            ori_time = parse_time(eq.attrs["origin_time"], assume_jst=ASSUME_JST)

        for sid, grp in f["stations"].items():
            if COMPONENT not in grp.get("waveforms", {}):
                continue
            sta_lat = float(grp.attrs["latitude"])
            sta_lon = float(grp.attrs["longitude"])
            if abs(sta_lat - epi_lat) > HALF_DEG or abs(sta_lon - epi_lon) > HALF_DEG:
                continue
            trigger_abs_s = picks.get(sid, None)
            if trigger_abs_s is None:
                continue
            dist_km = haversine_km(epi_lat, epi_lon, sta_lat, sta_lon)
            sr = float(grp.attrs["sampling_rate"])
            acc_raw = np.asarray(grp["waveforms"][COMPONENT][:], dtype=np.float32)
            network = str(grp.attrs.get("network", "Unknown"))
            rec_time = parse_time(grp.attrs["record_time"], assume_jst=ASSUME_JST)
            rec_start = (
                rec_time - timedelta(seconds=RECORD_PRETRIGGER_S)
                if RECORD_TIME_IS_TRIGGER
                else rec_time
            )
            offset_s = (rec_start - ori_time).total_seconds()
            n = len(acc_raw)
            t = np.arange(n) / sr + offset_s
            try:
                acc_disp = (
                    bandpass(acc_raw, sr, FREQ_BAND[0], FREQ_BAND[1])
                    if FREQ_BAND and len(acc_raw) > 20
                    else acc_raw.copy()
                )
            except Exception:
                acc_disp = acc_raw.copy()
            mask = (t >= T_WINDOW[0]) & (t <= T_WINDOW[1])
            if mask.sum() < 10:
                continue
            records.append(
                {
                    "sid": sid,
                    "network": network,
                    "dist_km": dist_km,
                    "trigger_s": float(trigger_abs_s),
                    "t": t[mask],
                    "acc": acc_disp[mask],
                    "pga": float(np.max(np.abs(acc_disp[mask]))),
                }
            )
    records.sort(key=lambda r: r["trigger_s"])
    return records, epi_lat, epi_lon, mag


def collect_stations(h5_path, epi_lat, epi_lon):
    sta_kiknet_lon, sta_kiknet_lat = [], []
    sta_knet_lon, sta_knet_lat = [], []
    with h5py.File(h5_path, "r") as f:
        for sid, grp in f["stations"].items():
            sta_lat = float(grp.attrs["latitude"])
            sta_lon = float(grp.attrs["longitude"])
            if abs(sta_lat - epi_lat) > HALF_DEG or abs(sta_lon - epi_lon) > HALF_DEG:
                continue
            network = str(grp.attrs.get("network", "Unknown"))
            if network == "KiK-net":
                sta_kiknet_lon.append(sta_lon)
                sta_kiknet_lat.append(sta_lat)
            else:
                sta_knet_lon.append(sta_lon)
                sta_knet_lat.append(sta_lat)
    return sta_kiknet_lon, sta_kiknet_lat, sta_knet_lon, sta_knet_lat


# ---------------------------------------------------------------------------
# Esri World Topo (same tile source as kumamoto_fault_comparison.ipynb)
# ---------------------------------------------------------------------------
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
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 record-section-plot"})
    with urlopen(req, timeout=40) as resp:
        data = resp.read()
    fp.write_bytes(data)
    return Image.open(BytesIO(data)).convert("RGB")


def fetch_esri_topo(lon_min, lon_max, lat_min, lat_max, zoom=TOPO_ZOOM, cache_dir=TILE_CACHE):
    x0, y0 = _lonlat_to_global_px(lon_min, lat_max, zoom)
    x1, y1 = _lonlat_to_global_px(lon_max, lat_min, zoom)
    tx0, ty0 = int(x0 // TILE_PX), int(y0 // TILE_PX)
    tx1, ty1 = int(x1 // TILE_PX), int(y1 // TILE_PX)
    cols = tx1 - tx0 + 1
    rows = ty1 - ty0 + 1
    mosaic = Image.new("RGB", (cols * TILE_PX, rows * TILE_PX))
    for iy, ty in enumerate(range(ty0, ty1 + 1)):
        for ix, tx in enumerate(range(tx0, tx1 + 1)):
            mosaic.paste(_fetch_tile(zoom, tx, ty, cache_dir), (ix * TILE_PX, iy * TILE_PX))
    left = x0 - tx0 * TILE_PX
    upper = y0 - ty0 * TILE_PX
    right = x1 - tx0 * TILE_PX
    lower = y1 - ty0 * TILE_PX
    cropped = mosaic.crop(
        (int(math.floor(left)), int(math.floor(upper)), int(math.ceil(right)), int(math.ceil(lower)))
    )
    return np.asarray(cropped).astype(np.float32) / 255.0


def dem_fallback_rgb(h5_path, lon_min, lon_max, lat_min, lat_max):
    """Hillshade without a blue ocean, used only if Esri tiles fail."""
    import rasterio
    from matplotlib.colors import LightSource, LinearSegmentedColormap
    from rasterio.windows import from_bounds

    event_dir = Path(h5_path).parent
    candidates = list(event_dir.glob("*srtm*.tif"))
    if not candidates:
        raise FileNotFoundError(f"no srtm tif in {event_dir}")
    dem_path = candidates[0]
    land_cmap = LinearSegmentedColormap.from_list(
        "land_navyfree",
        ["#efe8d2", "#d5e0b0", "#8aaa62", "#c4a056", "#8a5a32", "#f5f5f5"],
    )
    with rasterio.open(dem_path) as src:
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, transform=src.transform)
        dem = src.read(1, window=window).astype(float)
        nodata = src.nodata
        if nodata is not None:
            dem[dem == nodata] = np.nan
        dem_filled = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(
        dem_filled,
        cmap=land_cmap,
        blend_mode="overlay",
        vert_exag=1.5,
        dx=1,
        dy=1,
    )[..., :3]
    ocean = ~np.isfinite(dem) | (dem <= 0)
    rgb[ocean] = np.array([0.93, 0.91, 0.84])
    return rgb


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_one(event):
    h5_path = Path(event["h5"])
    event_name = h5_path.parent.name
    cache_path = h5_path.parent / f"{event_name}_phasenet_picks.csv"
    picks, df_picks = load_picks(cache_path)
    records, epi_lat, epi_lon, mag = load_records(str(h5_path), picks)
    n_total = len(records)
    triggered = [r for r in records if np.isfinite(r["trigger_s"])]
    print(f"{event_name}: stations={n_total} picked={len(triggered)} Mw={mag}")

    sta_kiknet_lon, sta_kiknet_lat, sta_knet_lon, sta_knet_lat = collect_stations(
        str(h5_path), epi_lat, epi_lon
    )
    dist_vals = [r["dist_km"] for r in records]

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.4, 1], hspace=0.4, wspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[1, 1])
    net_colors = {"KiK-net": "#d62728", "K-NET": "#1f77b4"}

    for rank, rec in enumerate(records):
        color = net_colors.get(rec["network"], "gray")
        pga = rec["pga"]
        if pga < 1e-6:
            continue
        ax.plot(
            rec["t"],
            rank + rec["acc"] / pga * SCALE,
            lw=0.5,
            color=color,
            alpha=0.75,
        )
    if triggered:
        ax.plot(
            [r["trigger_s"] for r in triggered],
            list(range(len(triggered))),
            "k--",
            lw=1.2,
            alpha=0.7,
        )
    ax.axvline(0, color="dimgray", lw=1.2, alpha=0.8, ls="-")

    length_km = 10 ** (FAULT_LENGTH_A * mag + FAULT_LENGTH_B)
    t_end_mag = length_km / RUPTURE_VELOCITY_KM_S + PHYSICAL_END_BUFFER_S
    ax.axvline(
        t_end_mag,
        color="#e377c2",
        lw=1.5,
        alpha=0.85,
        ls="-.",
        label=f"Mag-based end ($t={t_end_mag:.1f}$s)",
    )
    ax.axvspan(t_end_mag, T_WINDOW[1], alpha=0.06, color="#e377c2")

    ax.legend(
        handles=[
            Line2D([0], [0], color="#d62728", lw=2.0, label="KiK-net"),
            Line2D([0], [0], color="#1f77b4", lw=2.0, label="K-NET"),
            Line2D([0], [0], color="k", lw=1.5, ls="--", label="PhaseNet P-pick"),
            Line2D([0], [0], color="dimgray", lw=1.5, ls="-", label="Origin time ($t=0$)"),
            Line2D(
                [0],
                [0],
                color="#e377c2",
                lw=2.0,
                ls="-.",
                label=f"Mag-based end ({t_end_mag:.1f}s)",
            ),
        ],
        loc="upper left",
        fontsize=11,
        prop={"weight": "normal", "size": 11},
    )
    ax.set_xlabel("Time relative to origin (s)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Trigger order (early to late)", fontsize=18, fontweight="bold")
    ax.set_xlim(T_WINDOW)
    ax.set_ylim(-1, n_total + 1)
    ax.set_title(
        f"Record Section — {event['title']} M$_w${mag}\n"
        f"Grid: 1.5°×1.5° | Component: {COMPONENT} | "
        f"N={len(triggered)} picked / {n_total} total",
        fontsize=17,
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=15, width=1.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(axis="x", lw=0.4, alpha=0.4)

    ax_inset = ax.inset_axes([0.70, 0.03, 0.28, 0.26])
    net_counts = {}
    for rec in records:
        net_counts[rec["network"]] = net_counts.get(rec["network"], 0) + 1
    networks = list(net_counts.keys())
    counts = [net_counts[n] for n in networks]
    colors = [net_colors.get(n, "gray") for n in networks]
    x_pos = np.arange(len(networks))
    ax_inset.bar(x_pos, counts, color=colors, edgecolor="white", alpha=0.85)
    for i, c in enumerate(counts):
        ax_inset.text(i, c + 0.8, str(c), ha="center", va="bottom", fontsize=12, fontweight="bold")
    for i, (n, c) in enumerate(zip(networks, counts)):
        ax_inset.text(i, c / 2, n, ha="center", va="center", fontsize=11, color="black", fontweight="bold")
    ax_inset.set_ylim(0, max(counts) * 1.25)
    ax_inset.text(
        0.5,
        0.97,
        "Network",
        ha="center",
        va="top",
        transform=ax_inset.transAxes,
        fontsize=12,
        fontweight="bold",
    )
    ax_inset.set_xticks([])
    ax_inset.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax_inset.tick_params(labelsize=11)
    for label in ax_inset.get_yticklabels():
        label.set_fontweight("bold")
    ax_inset.set_facecolor("white")
    ax_inset.patch.set_alpha(1.0)
    ax_inset.grid(axis="y", lw=0.3, alpha=0.4)

    lon_min = epi_lon - HALF_DEG
    lon_max = epi_lon + HALF_DEG
    lat_min = epi_lat - HALF_DEG
    lat_max = epi_lat + HALF_DEG
    ax_d.set_xlim(lon_min, lon_max)
    ax_d.set_ylim(lat_min, lat_max)
    ax_d.set_aspect("equal")

    try:
        rgb = fetch_esri_topo(lon_min, lon_max, lat_min, lat_max)
        print(f"  Esri World Topo tiles OK  shape={rgb.shape}")
    except Exception as e:
        print(f"  [Esri fallback -> DEM] {e}")
        rgb = dem_fallback_rgb(str(h5_path), lon_min, lon_max, lat_min, lat_max)

    ax_d.imshow(
        rgb,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="upper",
        aspect="equal",
        zorder=0,
        interpolation="bilinear",
    )
    ax_d.add_patch(
        plt.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            facecolor="white",
            edgecolor="none",
            alpha=WHITE_VEIL_ALPHA,
            zorder=1,
        )
    )
    ax_d.add_patch(
        plt.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=1.5,
            edgecolor="#333333",
            facecolor="none",
            linestyle="--",
            zorder=4,
        )
    )
    def scatter_stations(lons, lats, color, label):
        # black halo so K-NET blue remains readable over pale Esri water
        ax_d.scatter(lons, lats, s=54, c="k", marker="o", linewidths=0, zorder=3, alpha=0.9)
        ax_d.scatter(
            lons, lats, s=34, c=color, marker="o", alpha=0.95,
            edgecolors="white", linewidths=0.35, zorder=4, label=label,
        )

    scatter_stations(sta_knet_lon, sta_knet_lat, "#1f77b4", f"K-NET (N={len(sta_knet_lon)})")
    scatter_stations(sta_kiknet_lon, sta_kiknet_lat, "#d62728", f"KiK-net (N={len(sta_kiknet_lon)})")
    ax_d.scatter(
        epi_lon,
        epi_lat,
        s=160,
        c="gold",
        marker="*",
        edgecolors="#333333",
        linewidths=0.8,
        zorder=5,
        label="Epicenter",
    )
    def draw_isochron(t_s, az_deg=22.0):
        r_deg = (V_P_KM_S * t_s) / 111.0
        ang = np.linspace(0, 2 * np.pi, 360)
        cx = epi_lon + r_deg * np.cos(ang)
        cy = epi_lat + r_deg * np.sin(ang)
        ax_d.plot(cx, cy, color="white", lw=3.6, ls="-", alpha=0.95, zorder=4.4, solid_capstyle="round")
        ax_d.plot(cx, cy, color="#111111", lw=1.9, ls="--", alpha=1.0, zorder=4.5)
        r_lab = min(r_deg, HALF_DEG - 0.10)
        lab_lon = epi_lon + r_lab * np.cos(np.radians(az_deg))
        lab_lat = epi_lat + r_lab * np.sin(np.radians(az_deg))
        ax_d.text(
            lab_lon,
            lab_lat,
            f"{t_s} s",
            fontsize=11,
            fontweight="bold",
            color="#111111",
            ha="center",
            va="center",
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="#222222",
                linewidth=0.7,
                alpha=0.95,
            ),
        )

    for t_s in (5, 10, 15):
        draw_isochron(t_s)
    ax_d.set_xlabel("Longitude (°E)", fontsize=14, fontweight="bold")
    ax_d.set_ylabel("Latitude (°N)", fontsize=14, fontweight="bold")
    ax_d.set_title("Station distribution\n(1.5°×1.5° grid)", fontsize=15, fontweight="bold")
    ax_d.tick_params(axis="x", labelsize=10, width=1.5, rotation=30)
    ax_d.tick_params(axis="y", labelsize=12, width=1.5)
    for label in ax_d.get_xticklabels():
        label.set_fontweight("bold")
        label.set_ha("right")
    for label in ax_d.get_yticklabels():
        label.set_fontweight("bold")
    ax_d.legend(
        fontsize=11,
        loc="lower right",
        framealpha=0.9,
        edgecolor="#cccccc",
        markerscale=1.2,
        prop={"weight": "normal", "size": 11},
    )
    ax_d.annotate(
        "N",
        xy=(0.95, 0.90),
        xytext=(0.95, 0.78),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        fontsize=14,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5),
        color="#333333",
    )
    scale_km = 50.0
    scale_deg = scale_km / 111.0
    bar_x0 = lon_min + 0.05
    bar_y0 = lat_min + 0.04
    ax_d.plot(
        [bar_x0, bar_x0 + scale_deg],
        [bar_y0, bar_y0],
        color="#333333",
        lw=2.5,
        solid_capstyle="butt",
        zorder=5,
    )
    ax_d.text(
        bar_x0 + scale_deg / 2,
        bar_y0 + 0.02,
        f"{scale_km} km",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#333333",
        zorder=5,
    )

    consistent, inconsistent, theoretical = [], [], []
    for rec in records:
        if rec["sid"] not in df_picks.index:
            continue
        row = df_picks.loc[rec["sid"]]
        t_theory = rec["dist_km"] / V_P_KM_S
        source = str(row.get("source", ""))
        residual = rec["trigger_s"] - t_theory
        if source == "theoretical":
            theoretical.append((rec["dist_km"], rec["trigger_s"]))
        elif abs(residual) <= T_P_TOLERANCE_S:
            consistent.append((rec["dist_km"], rec["trigger_s"]))
        else:
            inconsistent.append((rec["dist_km"], rec["trigger_s"]))
    dist_range = np.linspace(0, max(dist_vals) * 1.05, 200)
    ax_r.plot(
        dist_range,
        dist_range / V_P_KM_S,
        "k--",
        lw=1.5,
        alpha=0.6,
        label=f"Theoretical ($V_P$={V_P_KM_S} km/s)",
    )
    ax_r.fill_between(
        dist_range,
        dist_range / V_P_KM_S - T_P_TOLERANCE_S,
        dist_range / V_P_KM_S + T_P_TOLERANCE_S,
        color="gray",
        alpha=0.15,
        label=f"±{T_P_TOLERANCE_S}s tolerance",
    )
    if consistent:
        dx, ty = zip(*consistent)
        ax_r.scatter(
            dx, ty, s=28, color="#2ca02c", alpha=0.8, zorder=3,
            label=f"PhaseNet consistent (N={len(consistent)})",
        )
    if inconsistent:
        dx, ty = zip(*inconsistent)
        ax_r.scatter(
            dx, ty, s=28, color="#ff7f0e", alpha=0.8, zorder=3, marker="^",
            label=f"PhaseNet inconsistent (N={len(inconsistent)})",
        )
    if theoretical:
        dx, ty = zip(*theoretical)
        ax_r.scatter(
            dx, ty, s=24, color="red", alpha=0.6, zorder=2, marker="x",
            label=f"Theoretical fallback (N={len(theoretical)})",
        )
    ax_r.set_xlabel("Epicentral distance (km)", fontsize=14, fontweight="bold")
    ax_r.set_ylabel("P-pick time (s)", fontsize=14, fontweight="bold")
    ax_r.set_title("P-pick vs theoretical", fontsize=15, fontweight="bold")
    ax_r.legend(fontsize=11, loc="upper left", prop={"weight": "normal", "size": 11})
    ax_r.tick_params(labelsize=12, width=1.5)
    for label in ax_r.get_xticklabels() + ax_r.get_yticklabels():
        label.set_fontweight("bold")
    ax_r.grid(lw=0.4, alpha=0.4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / event["out_name"]
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    also = h5_path.parent / event["out_name"]
    fig.savefig(also, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")
    print(f"  saved {also}")
    return out_path


def main():
    paths = []
    for event in EVENTS:
        paths.append(plot_one(event))
    print("DONE", paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
