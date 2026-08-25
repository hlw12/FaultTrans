#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Native FinDer PGA fields for Table 2. Two modes, do not mix.

Japan (Kumamoto, Tottori): Gaussian interpolation, zeros kept; Table 2 Δθ =
ellipse of the painted line (13.5 / 12.0 / 35.7).
USA (Ridgecrest): Delaunay interpolation; Table 2 Δθ = geographic strike
(strike−90) vs GT ellipse (15.5 / 13.9 / 9.7).

Does not modify FaultTrans pred_binary / pred_prob / pgv_fields.
Writes pga_fields_finder.npy and finder_pred_line_mask.npy only.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import griddata
from scipy.signal import butter, detrend, filtfilt

ROOT = Path("/root/autodl-tmp")
HERE = ROOT / "Test_on_Event" / "unified"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "CmpFinDer"))
sys.path.insert(0, str(ROOT / "nature_figures"))
sys.path.insert(0, str(HERE))

from fields import EVENTS, strike_delta  # noqa: E402
from geo import delta_theta, ellipse_major_angle  # noqa: E402
from infer import GRID_SIZE, gt_masks, run_finder  # noqa: E402

JST = timezone(timedelta(hours=9))
UTC = timezone.utc
INTERP_SIGMA_DEG = 0.15
INTERP_CUTOFF_SIGMA = 4.0
RECORD_PRETRIGGER_S = 15.0

# Table 2 reprint gates for the two-mode recipe below.
PAPER = {
    ("kumamoto", "RP"): 13.5,
    ("kumamoto", "Curved"): 12.0,
    ("Tottori", "RP"): 35.7,
    ("ridgecrest", "JIN"): 15.5,
    ("ridgecrest", "ROSS"): 13.9,
    ("ridgecrest", "XU"): 9.7,
}


def parse_event_time(value, naive_tz=JST):
    if value is None:
        return None
    text = value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
    text = str(text).strip().replace("Z", "+00:00")
    if not text or text.upper() == "UNKNOWN":
        return None
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=naive_tz)
    return dt


def preprocess_acc(acc, sr, pre_event_s=5.0, hp_hz=0.10, lp_hz=10.0, filt_order=3):
    x = np.asarray(acc, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        return x.astype(np.float32)
    n_pre = min(int(pre_event_s * sr), len(x))
    baseline = float(np.median(x[:n_pre])) if n_pre > 16 else float(np.median(x))
    x = detrend(x - baseline, type="linear").astype(np.float32)
    nyq = 0.5 * sr
    hp = max(hp_hz / nyq, 1e-5)
    lp = min(lp_hz / nyq, 0.999)
    b, a = butter(filt_order, [hp, lp], btype="band") if hp < lp else butter(filt_order, hp, btype="high")
    return filtfilt(b, a, x).astype(np.float32)


def gauss_interp(lons, lats, values, grid_lon, grid_lat, sigma_deg=INTERP_SIGMA_DEG, cutoff_sigma=INTERP_CUTOFF_SIGMA):
    """Japan / compare_finder: zeros are interpolated, not dropped."""
    lons = np.asarray(lons, dtype=np.float32)
    lats = np.asarray(lats, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(grid_lon, dtype=np.float32)
    wsum = np.zeros_like(grid_lon, dtype=np.float32)
    if len(values) < 3:
        return out
    cutoff2 = (float(cutoff_sigma) * float(sigma_deg)) ** 2
    inv_2s2 = 1.0 / (2.0 * float(sigma_deg) ** 2)
    for lo, la, val in zip(lons, lats, values):
        dist2 = (grid_lon - lo) ** 2 + (grid_lat - la) ** 2
        m = dist2 <= cutoff2
        if not np.any(m):
            continue
        w = np.exp(-dist2[m] * inv_2s2)
        out[m] += w * val
        wsum[m] += w
    ok = wsum > 0
    out[ok] /= wsum[ok]
    return out


def delaunay_interp(lons, lats, values, grid_lon, grid_lat):
    pts = np.column_stack([np.asarray(lons, np.float64), np.asarray(lats, np.float64)])
    vals = np.asarray(values, np.float64).reshape(-1)
    corners = np.array(
        [
            [float(grid_lon.min()), float(grid_lat.min())],
            [float(grid_lon.min()), float(grid_lat.max())],
            [float(grid_lon.max()), float(grid_lat.min())],
            [float(grid_lon.max()), float(grid_lat.max())],
        ],
        dtype=np.float64,
    )
    pts = np.vstack([pts, corners])
    vals = np.concatenate([vals, np.zeros(len(corners))])
    zi = griddata(pts, vals, (grid_lon, grid_lat), method="linear", fill_value=0.0)
    return np.nan_to_num(zi, nan=0.0).astype(np.float32)


def fill_gauss_pga(stations, times, t_first, grid_lon, grid_lat):
    """Japan: every station is used; untriggered samples are zeros."""
    fields = np.zeros((len(times), GRID_SIZE, GRID_SIZE), np.float32)
    lons = np.asarray([s["lon"] for s in stations], np.float32)
    lats = np.asarray([s["lat"] for s in stations], np.float32)
    for ti, t_inf in enumerate(times):
        t_abs = float(t_inf) + t_first
        vals = []
        for s in stations:
            idx = int(np.floor((t_abs - s["t0_s"]) * s["sr"]))
            if idx < 0:
                vals.append(0.0)
                continue
            idx = min(idx, len(s["acc"]) - 1)
            vals.append(float(np.max(s["acc"][: idx + 1])))
        fields[ti] = gauss_interp(lons, lats, np.asarray(vals, np.float32), grid_lon, grid_lat)
    return fields


def fill_delaunay_pga(stations, times, t_first, grid_lon, grid_lat):
    """USA / Ridgecrest: untriggered stations are omitted; box corners padded with 0."""
    fields = np.zeros((len(times), GRID_SIZE, GRID_SIZE), np.float32)
    for ti, t_inf in enumerate(times):
        t_abs = float(t_inf) + t_first
        vals, xs, ys = [], [], []
        for s in stations:
            idx = int(np.floor((t_abs - s["t0_s"]) * s["sr"]))
            if idx < 0:
                continue
            idx = min(idx, len(s["acc"]) - 1)
            vals.append(float(np.max(s["acc"][: idx + 1])))
            xs.append(s["lon"])
            ys.append(s["lat"])
        if len(vals) < 3:
            continue
        fields[ti] = delaunay_interp(xs, ys, vals, grid_lon, grid_lat)
    return fields


def eval_clock(src: Path):
    meta = dict(np.load(src / "event_metadata.npz", allow_pickle=True))
    times = np.load(src / "times_mid_valid.npy").astype(np.float32)
    t_first = float(meta["t_first_trigger_s"])
    t_end = float(meta["effective_end_inf_s"])
    t_abs = float(meta["effective_end_abs_s"])
    if abs(t_end - t_abs) < 0.1:
        t_end = max(0.0, t_abs - t_first)
    final_idx = int(np.where(times <= t_end)[0][-1])
    return meta, times, t_first, t_end, final_idx


def pick_maxabs(wgrp):
    comps = []
    for key in ("EW", "NS", "UD", "EW1", "NS1", "e", "n", "z", "E", "N", "Z"):
        if key in wgrp:
            arr = np.asarray(wgrp[key][...], dtype=np.float32).reshape(-1)
            if arr.size:
                comps.append(arr)
    if not comps:
        return None
    n = min(len(c) for c in comps)
    return np.max(np.abs(np.stack([c[:n] for c in comps], axis=0)), axis=0).astype(np.float32)


def japan_t0(group, origin):
    start_attr = group.attrs.get("starttime", group.attrs.get("start_time", None))
    start_dt = parse_event_time(start_attr, JST)
    if start_dt is None or origin is None:
        return -RECORD_PRETRIGGER_S
    t0 = (start_dt - origin).total_seconds()
    t0 -= RECORD_PRETRIGGER_S
    return float(t0)


def build_japan(key):
    cfg = EVENTS[key]
    src = cfg["src"]
    meta, times, t_first, t_end, final_idx = eval_clock(src)
    origin = parse_event_time(cfg["origin"], JST)
    h5s = sorted(src.glob("*.h5"))
    eq_lat, eq_lon = float(meta["eq_lat"]), float(meta["eq_lon"])
    lon_min, lon_max = float(meta["lon_min"]), float(meta["lon_max"])
    lat_min, lat_max = float(meta["lat_min"]), float(meta["lat_max"])
    stations = []
    with h5py.File(h5s[0], "r") as f:
        for sid, g in f["stations"].items():
            if "waveforms" not in g:
                continue
            lo, la = float(g.attrs["longitude"]), float(g.attrs["latitude"])
            if not (lon_min <= lo <= lon_max and lat_min <= la <= lat_max):
                continue
            acc_raw = pick_maxabs(g["waveforms"])
            if acc_raw is None:
                continue
            sr = float(g.attrs.get("sampling_rate", g.attrs.get("sfreq", 100.0)))
            if sr <= 1e-6:
                continue
            acc = preprocess_acc(acc_raw, sr)
            stations.append(dict(lon=lo, lat=la, sr=sr, t0_s=japan_t0(g, origin), acc=acc))
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(lon_min, lon_max, GRID_SIZE, np.float32),
        np.linspace(lat_min, lat_max, GRID_SIZE, np.float32),
    )
    fields = fill_gauss_pga(stations, times, t_first, grid_lon, grid_lat)
    packed = dict(
        src=src,
        times=times,
        t_first=t_first,
        t_end=t_end,
        final_idx=final_idx,
        eq_lat=eq_lat,
        eq_lon=eq_lon,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        stations=stations,
    )
    print(f"[{key}/japan-gauss] stations={len(stations)} frames={len(times)} idx={final_idx} t={times[final_idx]:.3f}")
    return packed, fields


def pick_h1h2(wgrp):
    def _load(k):
        if k not in wgrp:
            return None
        a = np.asarray(wgrp[k][...], np.float32).reshape(-1)
        return a if a.size else None

    h1, h2 = _load("H1"), _load("H2")
    if h1 is not None and h2 is not None:
        n = min(len(h1), len(h2))
        return np.maximum(np.abs(h1[:n]), np.abs(h2[:n])).astype(np.float32)
    return h1 if h1 is not None else h2


def vel_to_pga(vel, sr, pga_gal_target=None):
    vel = np.asarray(vel, np.float64).reshape(-1)
    if len(vel) < 3 or sr <= 1e-6:
        return np.zeros(0, np.float32)
    acc = np.gradient(vel, 1.0 / float(sr))
    acc = preprocess_acc(acc.astype(np.float32), sr)
    peak = float(np.max(np.abs(acc))) if len(acc) else 0.0
    if pga_gal_target and np.isfinite(pga_gal_target) and pga_gal_target > 0 and peak > 1e-6:
        acc = acc * (float(pga_gal_target) / peak)
    return acc.astype(np.float32)


def build_ridgecrest():
    cfg = EVENTS["ridgecrest"]
    src = cfg["src"]
    meta, times, t_first, t_end, final_idx = eval_clock(src)
    eq_lat, eq_lon = float(meta["eq_lat"]), float(meta["eq_lon"])
    lon_min, lon_max = float(meta["lon_min"]), float(meta["lon_max"])
    lat_min, lat_max = float(meta["lat_min"]), float(meta["lat_max"])
    stations = []
    with h5py.File(src / "ridgecrest.h5", "r") as f:
        origin = parse_event_time(f["earthquake"].attrs.get("origin_time", None), UTC)
        for sid, g in f["stations"].items():
            if str(sid).upper().startswith("CHAN"):
                continue
            if "waveforms" not in g:
                continue
            lo, la = float(g.attrs["longitude"]), float(g.attrs["latitude"])
            if not (lon_min <= lo <= lon_max and lat_min <= la <= lat_max):
                continue
            vel = pick_h1h2(g["waveforms"])
            if vel is None:
                continue
            sr = float(g.attrs.get("sampling_rate", 100.0))
            start_attr = g.attrs.get("record_time", g.attrs.get("starttime", None))
            start_dt = parse_event_time(start_attr, UTC)
            t0 = 0.0 if (start_dt is None or origin is None) else (start_dt - origin).total_seconds()
            if t0 > 30.0:
                continue
            pga_gal = np.nan
            if "peak_values" in g:
                pga_gal = float(g["peak_values"].attrs.get("pga_gal", np.nan))
            acc = vel_to_pga(vel, sr, pga_gal)
            if len(acc) == 0:
                continue
            stations.append(dict(lon=lo, lat=la, sr=sr, t0_s=t0, acc=acc, peak=float(np.max(np.abs(acc)))))
    best = {}
    for s in stations:
        key = (round(s["lon"], 3), round(s["lat"], 3))
        if key not in best or s["peak"] > best[key]["peak"]:
            best[key] = s
    stations = list(best.values())
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(lon_min, lon_max, GRID_SIZE, np.float32),
        np.linspace(lat_min, lat_max, GRID_SIZE, np.float32),
    )
    fields = fill_delaunay_pga(stations, times, t_first, grid_lon, grid_lat)
    packed = dict(
        src=src,
        times=times,
        t_first=t_first,
        t_end=t_end,
        final_idx=final_idx,
        eq_lat=eq_lat,
        eq_lon=eq_lon,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        stations=stations,
    )
    print(f"[ridgecrest/usa-delaunay] stations={len(stations)} frames={len(times)} idx={final_idx}")
    return packed, fields


def geographic_strike_to_cv(strike_deg):
    """compare_ridgecrest: FinDer geographic strike (0=N, 90=E) → OpenCV ellipse angle."""
    if strike_deg is None or not np.isfinite(strike_deg):
        return np.nan
    return float((float(strike_deg) - 90.0) % 180.0)


def geo_strike_delta(strike_geo, gt_mask):
    """Table 2 Ridgecrest: (strike − 90) vs GT fitEllipse, not the painted-line ellipse."""
    return delta_theta(geographic_strike_to_cv(strike_geo), ellipse_major_angle(gt_mask))


def save_and_score(key, packed, fields):
    src = packed["src"]
    np.save(src / "pga_fields_finder.npy", fields)
    _res, masks, _pk = run_finder(fields, packed)
    np.save(src / "finder_pred_line_mask.npy", masks)
    np.save(src / "finder_times_mid.npy", packed["times"])
    idx = packed["final_idx"]
    line = masks[idx]
    gts = gt_masks(key, packed)
    rows = []
    strike_geo = float(getattr(_res[idx], "strike_deg", np.nan))
    strike_cv = geographic_strike_to_cv(strike_geo)
    # Japan Table 2 = ellipse of the line mask. Ridgecrest Table 2 = geographic strike.
    table2_kind = "geo-strike" if key == "ridgecrest" else "ellipse"
    print(
        f"  line_px={int(line.sum())} strike_geo={strike_geo} "
        f"strike_cv={strike_cv:.1f} table2={table2_kind}"
    )
    for gt_name, gt in gts.items():
        d_ell = strike_delta(line, gt)
        d_geo = geo_strike_delta(strike_geo, gt)
        d_table2 = d_geo if key == "ridgecrest" else d_ell
        paper = PAPER.get((key, gt_name))
        rows.append(
            dict(
                event=key,
                gt=gt_name,
                finder_dtheta_table2=None if not np.isfinite(d_table2) else round(float(d_table2), 3),
                table2_kind=table2_kind,
                finder_dtheta_ellipse=None if not np.isfinite(d_ell) else round(float(d_ell), 3),
                finder_dtheta_geo=None if not np.isfinite(d_geo) else round(float(d_geo), 3),
                paper=paper,
                line_px=int(line.sum()),
                strike_geo=strike_geo,
                strike_cv=None if not np.isfinite(strike_cv) else round(float(strike_cv), 3),
            )
        )
        mark = ""
        if paper is not None and np.isfinite(d_table2):
            mark = " OK" if abs(d_table2 - paper) <= 1.5 else " DRIFT"
        print(
            f"  [{key}/{gt_name}] Table2 Δθ={d_table2:.1f} ({table2_kind})  "
            f"ellipse={d_ell:.1f}  geo-strike={d_geo:.1f}  paper={paper}{mark}"
        )
    return rows


def main():
    rows = []
    for key in ("kumamoto", "Tottori"):
        packed, fields = build_japan(key)
        rows.extend(save_and_score(key, packed, fields))
    packed, fields = build_ridgecrest()
    rows.extend(save_and_score("ridgecrest", packed, fields))
    out = HERE / "outputs" / "finder_native.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("[finder_native done]", datetime.now().isoformat(), out)


if __name__ == "__main__":
    main()
