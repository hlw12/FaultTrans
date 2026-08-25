"""Geometry, metrics, and ground-truth loaders. Does not plot."""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

GRID_SIZE = 150
RESOLUTION = 0.01
SLIP_RATIO = 0.10
SUBFAULT_SIZE_KM = 2.0


def compute_metrics(pred, gt):
    pred = np.asarray(pred).astype(bool).ravel()
    gt = np.asarray(gt).astype(bool).ravel()
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    return {
        "IoU": float(tp / (tp + fp + fn + 1e-8)),
        "F1": float(2 * tp / (2 * tp + fp + fn + 1e-8)),
        "Precision": float(tp / (tp + fp + 1e-8)),
        "Recall": float(tp / (gt.sum() + 1e-8)),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def ellipse_major_angle(mask, min_pixels=20):
    """Major-axis angle in [0, 180), matching Test_on_Event cv2.fitEllipse."""
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    if int(mask_u8.sum()) < min_pixels:
        return np.nan
    rows, cols = np.where(mask_u8)
    if len(rows) < 5:
        return np.nan
    try:
        import cv2

        pts = np.column_stack([cols, rows]).astype(np.float32)
        _, (_, _), angle = cv2.fitEllipse(pts)
        return float(angle % 180.0)
    except Exception:
        pts = np.stack([cols.astype(np.float64), rows.astype(np.float64)], axis=1)
        pts -= pts.mean(axis=0, keepdims=True)
        cov = np.cov(pts, rowvar=False)
        if not np.all(np.isfinite(cov)):
            return np.nan
        w, v = np.linalg.eigh(cov)
        axis = v[:, int(np.argmax(w))]
        return float(np.degrees(np.arctan2(axis[1], axis[0]))) % 180.0


def delta_theta(a, b):
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    d = abs(float(a) - float(b))
    return float(min(d, 180.0 - d))


def overlap_rgb(pred, gt, colors):
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    rgb = np.ones(pred.shape + (3,), dtype=np.float32)
    rgb[:] = 0.96
    def hex_to_rgb(h):
        h = h.lstrip("#")
        return np.array([int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)

    rgb[pred & gt] = hex_to_rgb(colors["tp"])
    rgb[pred & ~gt] = hex_to_rgb(colors["fp"])
    rgb[~pred & gt] = hex_to_rgb(colors["fn"])
    return rgb


def rasterize_subfaults(df, eq_lat, eq_lon, size_dip_key="size_dip_km", size_strike_key="size_strike_km"):
    half = GRID_SIZE * RESOLUTION * 0.5
    lon_min, lon_max = eq_lon - half, eq_lon + half
    lat_min, lat_max = eq_lat - half, eq_lat + half
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    if df is None or len(df) == 0:
        return mask.astype(np.uint8), mask
    sm = float(df["slip_m"].max())
    if sm <= 0:
        return mask.astype(np.uint8), mask
    dfa = df[df["slip_m"] >= sm * SLIP_RATIO]
    dpx_lon = (lon_max - lon_min) / (GRID_SIZE - 1)
    dpx_lat = (lat_max - lat_min) / (GRID_SIZE - 1)
    m_lat = 111.32
    m_lon = 111.32 * np.cos(np.radians(eq_lat))
    for _, row in dfa.iterrows():
        cf = (row["lon"] - lon_min) / (lon_max - lon_min) * (GRID_SIZE - 1)
        rf = (row["lat"] - lat_min) / (lat_max - lat_min) * (GRID_SIZE - 1)
        sd = float(row.get(size_dip_key, SUBFAULT_SIZE_KM) or SUBFAULT_SIZE_KM)
        ss = float(row.get(size_strike_key, SUBFAULT_SIZE_KM) or SUBFAULT_SIZE_KM)
        rx = (ss / m_lon) / dpx_lon * 0.5
        ry = (sd / m_lat) / dpx_lat * 0.5
        c0 = max(0, int(np.floor(cf - rx)))
        c1 = min(GRID_SIZE - 1, int(np.ceil(cf + rx)))
        r0 = max(0, int(np.floor(rf - ry)))
        r1 = min(GRID_SIZE - 1, int(np.ceil(rf + ry)))
        mask[r0 : r1 + 1, c0 : c1 + 1] = np.maximum(mask[r0 : r1 + 1, c0 : c1 + 1], row["slip_m"] / sm)
    return (mask > 0).astype(np.uint8), mask


def parse_nied(filepath):
    records = []
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()
    pat = re.compile(r"^\s*(\d+)\s+(\d+)\s+")
    for line in lines:
        if line.strip().startswith("#") or not pat.match(line):
            continue
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            records.append(
                dict(
                    lat=float(parts[2]),
                    lon=float(parts[3]),
                    size_strike_km=float(parts[7]) / 1000.0,
                    size_dip_km=float(parts[8]) / 1000.0,
                    slip_m=float(parts[10]),
                    ttrg_s=float(parts[12]),
                )
            )
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(records)


def parse_pku(filepath):
    sr = 2.0
    rows = []
    in_model = False
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("# Sample Rate"):
                continue
            if s and not s.startswith("#") and "Sample Rate" not in s:
                parts = s.split()
                if len(parts) == 1:
                    try:
                        sr = float(parts[0])
                    except ValueError:
                        pass
            if "Rupture Model" in s:
                in_model = True
                continue
            if not in_model or not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 9:
                continue
            stf = np.array([float(x) for x in parts[9:]], dtype=np.float64)
            pos = np.where(stf > 1e-8)[0]
            t_rup = float(pos[0] / sr) if len(pos) else np.nan
            rows.append(
                dict(
                    lat=float(parts[0]),
                    lon=float(parts[1]),
                    size_dip_km=float(parts[3]),
                    size_strike_km=float(parts[4]),
                    slip_m=float(parts[8]),
                    ttrg_s=t_rup,
                )
            )
    return pd.DataFrame(rows)


def _deep_flatten(v):
    if v is None:
        return np.array([], dtype=np.float64)
    try:
        a = np.asarray(v)
    except Exception:
        return np.array([], dtype=np.float64)
    if a.size == 0:
        return np.array([], dtype=np.float64)
    if a.dtype == object:
        parts = [_deep_flatten(c) for c in a.flat if c is not None]
        return np.concatenate(parts) if parts else np.array([], dtype=np.float64)
    try:
        return a.astype(np.float64).flatten()
    except Exception:
        return np.array([], dtype=np.float64)


def _safe_pair(v, default=(0.0, 0.0)):
    a = _deep_flatten(v)
    return (float(a[0]), float(a[1])) if a.size >= 2 else default


def _seg_field_index(s, fields, aliases):
    for alias in aliases:
        pat = re.compile(r"^seg(\d+)" + re.escape(alias) + r"$")
        items = sorted(
            [(int(m.group(1)), f) for f in fields if (m := pat.match(f))],
            key=lambda x: x[0],
        )
        if items:
            return items, alias
    return [], None


def parse_srcmod_mat(mat_path):
    data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    top = [k for k in data if not k.startswith("__")]
    if not top:
        return pd.DataFrame()
    s = data[top[0]]
    fields = getattr(s, "_fieldnames", [])
    fld = lambda n: getattr(s, n) if n in fields else None
    sd_g, ss_g = _safe_pair(fld("invDzDx"), (SUBFAULT_SIZE_KM, SUBFAULT_SIZE_KM))
    sd_g = SUBFAULT_SIZE_KM if sd_g <= 0 or sd_g > 50 else sd_g
    ss_g = SUBFAULT_SIZE_KM if ss_g <= 0 or ss_g > 50 else ss_g
    if "slipSPL" in fields and "geoLAT" in fields:
        slip = _deep_flatten(fld("slipSPL"))
        lat = _deep_flatten(fld("geoLAT"))
        lon = _deep_flatten(fld("geoLON"))
        sd_arr = np.full(len(slip), sd_g)
        ss_arr = np.full(len(slip), ss_g)
    else:
        slip_i, _ = _seg_field_index(s, fields, ["SLIP"])
        lat_i, _ = _seg_field_index(s, fields, ["geoLAT", "LAT", "Lat"])
        lon_i, _ = _seg_field_index(s, fields, ["geoLON", "LON", "Lon"])
        dim_i, _ = _seg_field_index(s, fields, ["DimWL", "invDzDx"])
        if not slip_i or not lat_i or not lon_i:
            return pd.DataFrame()
        slip_a = [_deep_flatten(getattr(s, name)) for _, name in slip_i]
        lat_a = [_deep_flatten(getattr(s, name)) for _, name in lat_i]
        lon_a = [_deep_flatten(getattr(s, name)) for _, name in lon_i]
        dim_a = [_deep_flatten(getattr(s, name)) for _, name in dim_i] if dim_i else [np.array([])] * len(slip_a)
        all_s, all_la, all_lo, all_sd, all_ss = [], [], [], [], []
        for k in range(len(slip_a)):
            sa = slip_a[k]
            la = lat_a[k] if k < len(lat_a) else np.array([])
            lo = lon_a[k] if k < len(lon_a) else np.array([])
            nn = min(sa.size, la.size, lo.size)
            if nn == 0:
                continue
            all_s.append(sa[:nn])
            all_la.append(la[:nn])
            all_lo.append(lo[:nn])
            di = np.asarray(dim_a[k] if k < len(dim_a) else np.array([]), dtype=np.float64).flatten()
            sd_s, ss_s = (float(di[0]), float(di[1])) if di.size >= 2 else (sd_g, ss_g)
            all_sd.append(np.full(nn, sd_s))
            all_ss.append(np.full(nn, ss_s))
        if not all_s:
            return pd.DataFrame()
        slip = np.concatenate(all_s)
        lat = np.concatenate(all_la)
        lon = np.concatenate(all_lo)
        sd_arr = np.concatenate(all_sd)
        ss_arr = np.concatenate(all_ss)
    if slip.size > 0 and np.nanmean(slip) > 10:
        slip = slip / 100.0
    nn = min(len(slip), len(lat), len(lon))
    return pd.DataFrame(
        [
            dict(
                lat=float(lat[k]),
                lon=float(lon[k]),
                size_dip_km=float(sd_arr[k]) if k < len(sd_arr) else sd_g,
                size_strike_km=float(ss_arr[k]) if k < len(ss_arr) else ss_g,
                slip_m=float(slip[k]),
            )
            for k in range(nn)
        ]
    )


def load_event_arrays(pgv_dir: Path):
    meta = np.load(pgv_dir / "event_metadata.npz", allow_pickle=True)
    pred = np.load(pgv_dir / "pred_binary.npy")
    t_axis = np.load(pgv_dir / "times_mid_valid.npy") if (pgv_dir / "times_mid_valid.npy").exists() else np.arange(pred.shape[0], dtype=np.float32)
    if len(t_axis) != pred.shape[0]:
        t_axis = np.arange(pred.shape[0], dtype=np.float32)
    t_first = float(meta["t_first_trigger_s"])
    if "effective_end_inf_s" in meta:
        t_end = float(meta["effective_end_inf_s"])
        t_abs = float(meta["effective_end_abs_s"])
        if abs(t_end - t_abs) < 0.1:
            t_end = max(0.0, t_abs - t_first)
    else:
        t_end = max(0.0, float(meta["effective_end_abs_s"]) - t_first)
    valid = t_axis <= t_end
    if valid.sum() == 0:
        valid = np.ones(len(t_axis), dtype=bool)
    final_idx = int(np.where(valid)[0][-1])
    fields = None
    for name in ("pgv_fields.npy", "pga_fields.npy"):
        p = pgv_dir / name
        if p.exists():
            fields = np.load(p)
            break
    return dict(
        meta=meta,
        pred=pred,
        t_axis=t_axis,
        t_first=t_first,
        t_end=t_end,
        valid=valid,
        final_idx=final_idx,
        eq_lat=float(meta["eq_lat"]),
        eq_lon=float(meta["eq_lon"]),
        eq_mag=float(meta["eq_mag"]),
        fields=fields,
        prob=np.load(pgv_dir / "pred_prob.npy") if (pgv_dir / "pred_prob.npy").exists() else None,
    )


def load_stations(h5_path: Path):
    lons, lats = [], []
    if h5_path is None or not Path(h5_path).exists():
        return np.array([]), np.array([])
    with h5py.File(h5_path, "r") as f:
        for _, g in f["stations"].items():
            lon = float(g.attrs.get("longitude", 0.0))
            lat = float(g.attrs.get("latitude", 0.0))
            if lon == 0.0 and lat == 0.0:
                continue
            lons.append(lon)
            lats.append(lat)
    return np.asarray(lons), np.asarray(lats)


def final_pred(bundle):
    return bundle["pred"][bundle["final_idx"]].astype(np.uint8)


def increment_field(fields, idx):
    if fields is None:
        return None
    idx = min(int(idx), fields.shape[0] - 1)
    if idx <= 0:
        return np.maximum(fields[0], 0)
    return np.maximum(fields[idx] - fields[idx - 1], 0)
