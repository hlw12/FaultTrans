"""Esri basemap helper for campaign maps.

World Topo (Fig. 9) composites map sheets and land-unit fills, which show up as
irregular gray blocks around Ridgecrest. Continuous terrain uses Terrain Base
modulated by hillshade so those fills do not appear.
"""
from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from style import CACHE

TILE_CACHE = CACHE.parent / "tile_cache"
TILE_PX = 256
TOPO_ZOOM = 10
WHITE_VEIL_ALPHA = 0.30

ESRI_TOPO = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
)
ESRI_TERRAIN = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Terrain_Base/MapServer/tile/{z}/{y}/{x}"
)
ESRI_HILLSHADE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}"
)
ESRI_PLACES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
)


def _lonlat_to_global_px(lon, lat, zoom):
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n * TILE_PX
    lat = max(min(lat, 85.05112878), -85.05112878)
    siny = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * n * TILE_PX
    return x, y


def _tile_cache_path(cache_dir, prefix, z, x, y):
    for ext in (".jpg", ".png", ".bin"):
        fp = cache_dir / f"{prefix}_{z}_{x}_{y}{ext}"
        if fp.exists() and fp.stat().st_size > 800:
            return fp
    return None


def _fetch_tile(url_template, prefix, z, x, y, cache_dir, mode="RGB"):
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = _tile_cache_path(cache_dir, prefix, z, x, y)
    if existing is not None:
        return Image.open(existing).convert(mode)
    url = url_template.format(z=z, x=x, y=y)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 nature-figures"})
    with urlopen(req, timeout=40) as resp:
        data = resp.read()
    ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
    fp = cache_dir / f"{prefix}_{z}_{x}_{y}{ext}"
    fp.write_bytes(data)
    return Image.open(BytesIO(data)).convert(mode)


def _mosaic_layer(url_template, prefix, lon_min, lon_max, lat_min, lat_max, zoom, cache_dir, mode="RGB"):
    x0, y0 = _lonlat_to_global_px(lon_min, lat_max, zoom)
    x1, y1 = _lonlat_to_global_px(lon_max, lat_min, zoom)
    tx0, ty0 = int(x0 // TILE_PX), int(y0 // TILE_PX)
    tx1, ty1 = int(x1 // TILE_PX), int(y1 // TILE_PX)
    mosaic = Image.new(mode, ((tx1 - tx0 + 1) * TILE_PX, (ty1 - ty0 + 1) * TILE_PX))
    for iy, ty in enumerate(range(ty0, ty1 + 1)):
        for ix, tx in enumerate(range(tx0, tx1 + 1)):
            mosaic.paste(
                _fetch_tile(url_template, prefix, zoom, tx, ty, cache_dir, mode=mode),
                (ix * TILE_PX, iy * TILE_PX),
            )
    cropped = mosaic.crop(
        (
            int(math.floor(x0 - tx0 * TILE_PX)),
            int(math.floor(y0 - ty0 * TILE_PX)),
            int(math.ceil(x1 - tx0 * TILE_PX)),
            int(math.ceil(y1 - ty0 * TILE_PX)),
        )
    )
    arr = np.asarray(cropped).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3 if mode == "RGB" else 4, axis=2)
    return arr


def fetch_esri_topo(lon_min, lon_max, lat_min, lat_max, zoom=TOPO_ZOOM, cache_dir=TILE_CACHE):
    return _mosaic_layer(ESRI_TOPO, "esri_topo", lon_min, lon_max, lat_min, lat_max, zoom, cache_dir, mode="RGB")


def fetch_continuous_terrain(lon_min, lon_max, lat_min, lat_max, zoom=11, cache_dir=TILE_CACHE):
    """Hypsometric terrain × hillshade, without World Topo land-unit fills."""
    terrain = _mosaic_layer(
        ESRI_TERRAIN, "esri_terrain", lon_min, lon_max, lat_min, lat_max, zoom, cache_dir, mode="RGB"
    )
    hill = _mosaic_layer(
        ESRI_HILLSHADE, "esri_hs", lon_min, lon_max, lat_min, lat_max, zoom, cache_dir, mode="RGB"
    )
    if hill.shape[:2] != terrain.shape[:2]:
        hill_img = Image.fromarray((np.clip(hill, 0, 1) * 255).astype(np.uint8))
        hill_img = hill_img.resize((terrain.shape[1], terrain.shape[0]), Image.Resampling.BILINEAR)
        hill = np.asarray(hill_img).astype(np.float32) / 255.0
    shade = 0.42 + 0.58 * hill
    rgb = np.clip(terrain * shade, 0.0, 1.0)
    try:
        places = _mosaic_layer(
            ESRI_PLACES, "esri_places", lon_min, lon_max, lat_min, lat_max, zoom, cache_dir, mode="RGBA"
        )
        if places.shape[:2] == rgb.shape[:2]:
            alpha = places[..., 3:4]
            rgb = rgb * (1.0 - alpha) + places[..., :3] * alpha
    except Exception:
        pass
    return rgb


def draw_topo_base(ax, rgb, lon_min, lon_max, lat_min, lat_max, veil=WHITE_VEIL_ALPHA):
    extent = [lon_min, lon_max, lat_min, lat_max]
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal")
    ax.imshow(rgb, extent=extent, origin="upper", aspect="equal", interpolation="bilinear", zorder=0)
    ax.add_patch(
        Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            facecolor="white",
            edgecolor="none",
            alpha=veil,
            zorder=1,
        )
    )
    return extent


def map_furniture(ax, lon_min, lon_max, lat_min, lat_max, scale_km=50.0):
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
    scale_deg = scale_km / 111.0
    bar_x0 = lon_min + 0.06
    bar_y0 = lat_min + 0.05
    ax.plot([bar_x0, bar_x0 + scale_deg], [bar_y0, bar_y0], color="#333333", lw=2.0, solid_capstyle="butt", zorder=6)
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
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=8)
    for lab in ax.get_xticklabels():
        lab.set_fontweight("bold")
        lab.set_ha("right")
    for lab in ax.get_yticklabels():
        lab.set_fontweight("bold")
    ax.set_xlabel("Longitude (°E)", fontsize=8, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=8, fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#333333")
