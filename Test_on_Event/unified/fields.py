#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FaultTrans real-event field construction (eval only; training untouched).

Locked FaultTrans recipe (PGV, and PGA for the fig06 control):
  - record_time (−15 s if the attribute is a trigger)
  - origin-time sampling; do not add t_first again
  - two horizontals EW/NS (H1/H2); no UD/V
  - combine: sample-wise max(|H1|,|H2|) for all events except Chi-Chi,
    which uses geometric mean of the two component envelopes (GMxy)
  - Gaussian σ=0.15°, zeros dropped (values > 0)
  - evaluation clock from existing event_metadata.npz / times_mid_valid.npy

FinDer is a reproduced baseline: do not use this interpolator for Table 2.
Native FinDer PGA fields are built by finder_native.py.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, detrend, filtfilt

GRID_SIZE = 150
RESOLUTION = 0.01
INTERP_SIGMA_DEG = 0.15
INTERP_CUTOFF_SIGMA = 4.0
HP_HZ, LP_HZ, FILT_ORD = 0.10, 10.0, 3
JST = timezone(timedelta(hours=9))
CST = timezone(timedelta(hours=8))
UTC = timezone.utc

EVENTS = {
    "kumamoto": dict(
        key="kumamoto",
        src=Path("/root/autodl-tmp/Test_on_Event/kumamoto"),
        origin="2016-04-16T01:25:05+09:00",
        style="acc",
        record_time_is_trigger=True,
        pretrigger_s=15.0,
        naive_tz=JST,
        time_attr="record_time",
        horiz=("EW", "NS", "EW1", "NS1"),
        finder=True,
        gts=("rp", "cv"),
    ),
    "Miyagi": dict(
        key="Miyagi",
        src=Path("/root/autodl-tmp/Test_on_Event/Miyagi"),
        origin="2008-06-14T08:43:45+09:00",
        style="acc",
        record_time_is_trigger=True,
        pretrigger_s=15.0,
        naive_tz=JST,
        time_attr="record_time",
        horiz=("EW", "NS", "EW1", "NS1"),
        finder=False,
        gts=("rp",),
    ),
    "Tottori": dict(
        key="Tottori",
        src=Path("/root/autodl-tmp/Test_on_Event/Tottori"),
        origin="2016-10-21T14:07:22+09:00",
        style="acc",
        record_time_is_trigger=True,
        pretrigger_s=15.0,
        naive_tz=JST,
        time_attr="record_time",
        horiz=("EW", "NS", "EW1", "NS1"),
        finder=True,
        gts=("rp",),
    ),
    "ridgecrest": dict(
        key="ridgecrest",
        src=Path("/root/autodl-tmp/Test_on_Event/ridgecrest/Ridgecrest"),
        origin="2019-07-06T03:19:53+00:00",
        style="vel",
        record_time_is_trigger=False,
        pretrigger_s=0.0,
        naive_tz=UTC,
        time_attr="record_time",
        horiz=("H1", "H2", "EW", "NS"),
        finder=True,
        gts=("JIN", "ROSS", "XU"),
    ),
    "chichi": dict(
        key="chichi",
        src=Path("/root/autodl-tmp/Test_on_Event/chichi/Chichi"),
        origin=None,
        style="acc",
        record_time_is_trigger=False,
        pretrigger_s=0.0,
        naive_tz=UTC,
        time_attr="record_time",
        horiz=("EW", "NS"),
        finder=False,
        gts=("CHI", "MA", "ZENG"),
    ),
    "Menyuan": dict(
        key="Menyuan",
        src=Path("/root/autodl-tmp/Test_on_Event/menyuan/Menyuan"),
        origin="2022-01-07T17:45:30+00:00",
        style="acc",
        record_time_is_trigger=False,
        pretrigger_s=0.0,
        naive_tz=CST,
        time_attr="recorder_time",
        horiz=("EW", "NS"),
        finder=False,
        gts=("pku",),
    ),
}

PAPER_SIX = ["kumamoto", "Miyagi", "Tottori", "ridgecrest", "chichi", "Menyuan"]
MAX_EVENTS = ["kumamoto", "Miyagi", "Tottori", "ridgecrest", "Menyuan"]
CHICHI_EVENTS = ["chichi"]
FINDER_EVENTS = ["kumamoto", "Tottori", "ridgecrest"]
COMBINE_BY_EVENT = {
    "kumamoto": "max",
    "Miyagi": "max",
    "Tottori": "max",
    "ridgecrest": "max",
    "Menyuan": "max",
    "chichi": "geomean",
}


def parse_event_time(value, naive_tz):
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


def preprocess_acc(acc, sr, pre_event_s=5.0, hp_hz=HP_HZ, lp_hz=LP_HZ, order=FILT_ORD):
    x = np.asarray(acc, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        return x
    n_pre = min(int(pre_event_s * sr), len(x))
    baseline = float(np.median(x[:n_pre])) if n_pre > 16 else float(np.median(x))
    x = detrend(x - baseline, type="linear").astype(np.float32)
    nyq = 0.5 * sr
    hp = max(hp_hz / nyq, 1e-5)
    lp = min(lp_hz / nyq, 0.999)
    b, a = butter(order, [hp, lp], btype="band") if hp < lp else butter(order, hp, btype="high")
    return filtfilt(b, a, x).astype(np.float32)


def acc_to_vel(acc_f, sr):
    vel = cumulative_trapezoid(acc_f, dx=1.0 / sr, initial=0.0).astype(np.float32)
    vel = detrend(vel, type="linear").astype(np.float32)
    nyq = 0.5 * sr
    hp2 = max(0.05 / nyq, 1e-5)
    if hp2 < 0.999:
        b2, a2 = butter(2, hp2, btype="high")
        vel = filtfilt(b2, a2, vel).astype(np.float32)
    return vel


def gauss_interp(lons, lats, values, grid_lon, grid_lat, sigma_deg=INTERP_SIGMA_DEG, cutoff_sigma=INTERP_CUTOFF_SIGMA):
    """Finite positive samples only. Unrecorded (nan) and still-zero IMs are omitted."""
    lons = np.asarray(lons, dtype=np.float32)
    lats = np.asarray(lats, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(grid_lon, dtype=np.float32)
    wsum = np.zeros_like(grid_lon, dtype=np.float32)
    m = np.isfinite(values) & (values > 0)
    if int(m.sum()) < 3:
        return out
    cutoff2 = (float(cutoff_sigma) * float(sigma_deg)) ** 2
    inv_2s2 = 1.0 / (2.0 * float(sigma_deg) ** 2)
    for lo, la, val in zip(lons[m], lats[m], values[m]):
        dist2 = (grid_lon - lo) ** 2 + (grid_lat - la) ** 2
        hit = dist2 <= cutoff2
        if not np.any(hit):
            continue
        w = np.exp(-dist2[hit] * inv_2s2)
        out[hit] += w * val
        wsum[hit] += w
    ok = wsum > 0
    out[ok] /= wsum[ok]
    return out


def load_horizontals(wgrp, names):
    found = []
    seen = set()
    for key in names:
        if key in wgrp and key not in seen:
            arr = np.asarray(wgrp[key][...], dtype=np.float32).reshape(-1)
            if arr.size > 0:
                found.append(arr)
                seen.add(key)
        if len(found) >= 2:
            break
    if not found:
        for key in wgrp.keys():
            if str(key).upper() in ("UD", "V", "Z"):
                continue
            arr = np.asarray(wgrp[key][...], dtype=np.float32).reshape(-1)
            if arr.size > 0:
                found.append(arr)
            if len(found) >= 2:
                break
    if not found:
        return None
    n = min(len(a) for a in found)
    return [a[:n] for a in found]


def eval_clock(src: Path):
    meta = dict(np.load(src / "event_metadata.npz", allow_pickle=True))
    times = np.load(src / "times_mid_valid.npy").astype(np.float32)
    t_first = float(meta["t_first_trigger_s"])
    t_abs = float(meta["effective_end_abs_s"])
    if "effective_end_inf_s" in meta:
        t_end = float(meta["effective_end_inf_s"])
        if abs(t_end - t_abs) < 0.1:
            t_end = max(0.0, t_abs - t_first)
    else:
        t_end = max(0.0, t_abs - t_first)
    final_idx = int(np.where(times <= t_end)[0][-1])
    return meta, times, t_first, t_end, final_idx


def rec_offset_s(group, cfg, origin, picks=None, eq_lat=None, eq_lon=None):
    raw = group.attrs.get(cfg["time_attr"], group.attrs.get("record_time", None))
    start_dt = parse_event_time(raw, cfg["naive_tz"])
    if start_dt is None or origin is None:
        pre = group.attrs.get("pre_event_s", group.attrs.get("pre_event_time", None))
        try:
            return -float(pre) if pre is not None else 0.0
        except (TypeError, ValueError):
            return 0.0
    t0 = (start_dt - origin).total_seconds()
    if cfg["record_time_is_trigger"]:
        t0 -= float(cfg["pretrigger_s"])
    return float(t0)


def build_station_traces(cfg):
    src = cfg["src"]
    meta, times, t_first, t_end, final_idx = eval_clock(src)
    eq_lat, eq_lon = float(meta["eq_lat"]), float(meta["eq_lon"])
    lon_min, lon_max = float(meta["lon_min"]), float(meta["lon_max"])
    lat_min, lat_max = float(meta["lat_min"]), float(meta["lat_max"])
    h5s = sorted(src.glob("*.h5"))
    if not h5s:
        raise FileNotFoundError(src)
    origin = parse_event_time(cfg["origin"], cfg["naive_tz"]) if cfg["origin"] else None
    stations = []
    with h5py.File(h5s[0], "r") as f:
        if origin is None:
            origin = parse_event_time(f["earthquake"].attrs.get("origin_time", None), cfg["naive_tz"])
        for sid, g in f["stations"].items():
            if "waveforms" not in g:
                continue
            lon = float(g.attrs.get("longitude", 0.0))
            lat = float(g.attrs.get("latitude", 0.0))
            if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                continue
            comps = load_horizontals(g["waveforms"], cfg["horiz"])
            if not comps:
                continue
            sr = float(g.attrs.get("sampling_rate", g.attrs.get("sfreq", 100.0)))
            if sr <= 1e-6:
                continue
            t0 = rec_offset_s(g, cfg, origin)
            accs, vels = [], []
            for raw in comps:
                if cfg["style"] == "vel":
                    vel_f = preprocess_acc(raw, sr)
                    acc_f = np.gradient(vel_f, 1.0 / sr).astype(np.float32)
                    acc_f = preprocess_acc(acc_f, sr)
                    vels.append(np.abs(vel_f))
                    accs.append(np.abs(acc_f))
                else:
                    acc_f = preprocess_acc(raw, sr)
                    vel_f = acc_to_vel(acc_f, sr)
                    accs.append(np.abs(acc_f))
                    vels.append(np.abs(vel_f))
            n = min(len(a) for a in accs + vels)
            stations.append(
                dict(
                    lon=lon,
                    lat=lat,
                    sr=sr,
                    t0_s=t0,
                    accs=[a[:n] for a in accs],
                    vels=[v[:n] for v in vels],
                    acc=np.maximum.reduce([a[:n] for a in accs]),
                    vel=np.maximum.reduce([v[:n] for v in vels]),
                    name=str(sid),
                )
            )
    if not stations:
        raise RuntimeError(f"no stations in grid for {cfg['key']}")
    return dict(
        stations=stations,
        times=times,
        meta=meta,
        t_first=t_first,
        t_end=t_end,
        final_idx=final_idx,
        eq_lat=eq_lat,
        eq_lon=eq_lon,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        src=src,
        combine="max",
    )


def _one_env(times, sr, t0, acc, vel, win_s=2.0):
    t_n = len(times)
    pga = np.full(t_n, np.nan, dtype=np.float32)
    pgv = np.full(t_n, np.nan, dtype=np.float32)
    win_n = max(1, int(round(win_s * sr)))
    cur_pga = cur_pgv = 0.0
    for ti, t_abs in enumerate(times):
        t_rec = float(t_abs) - t0
        if t_rec <= 0:
            continue
        i1 = int(np.clip(np.ceil(t_rec * sr), 0, len(acc)))
        if i1 <= 0:
            continue
        cur_pga = max(cur_pga, float(acc[:i1].max()))
        pga[ti] = cur_pga
        i0 = max(0, i1 - win_n)
        local_v = float(vel[i0:i1].max()) if i1 > i0 else 0.0
        cur_pgv = max(cur_pgv, local_v)
        pgv[ti] = cur_pgv
    return pga, pgv


def envelopes(packed, win_s=2.0, combine=None):
    """Station PGA/PGV envelopes.

    combine='max': sample-wise max(|EW|,|NS|) then running-peak (Japan/Ridgecrest/Menyuan).
    combine='geomean': independent running-peaks, then sqrt(IM_E * IM_N) (Chi-Chi).
    """
    how = combine or packed.get("combine") or "max"
    stations = packed["stations"]
    times = packed["times"]
    n, t_n = len(stations), len(times)
    pga = np.full((t_n, n), np.nan, dtype=np.float32)
    pgv = np.full((t_n, n), np.nan, dtype=np.float32)
    pga_rp = np.full((t_n, n), np.nan, dtype=np.float32)
    n_two = 0
    for j, s in enumerate(stations):
        sr, t0 = float(s["sr"]), float(s["t0_s"])
        accs = s.get("accs")
        vels = s.get("vels")
        if accs is None:
            accs, vels = [s["acc"]], [s["vel"]]
        n_two += int(len(accs) >= 2)
        if how == "geomean" and len(accs) >= 2:
            pgas, pgvs = [], []
            for acc, vel in zip(accs, vels):
                a, v = _one_env(times, sr, t0, acc, vel, win_s=win_s)
                pgas.append(a)
                pgvs.append(v)
            pga[:, j] = np.where(
                np.isfinite(pgas[0]) & np.isfinite(pgas[1]),
                np.sqrt(np.maximum(pgas[0], 0.0) * np.maximum(pgas[1], 0.0)),
                np.nan,
            )
            pgv[:, j] = np.where(
                np.isfinite(pgvs[0]) & np.isfinite(pgvs[1]),
                np.sqrt(np.maximum(pgvs[0], 0.0) * np.maximum(pgvs[1], 0.0)),
                np.nan,
            )
            pga_rp[:, j] = pga[:, j]
        else:
            acc = np.maximum.reduce(accs) if len(accs) > 1 else accs[0]
            vel = np.maximum.reduce(vels) if len(vels) > 1 else vels[0]
            pga[:, j], pgv[:, j] = _one_env(times, sr, t0, acc, vel, win_s=win_s)
            pga_rp[:, j] = pga[:, j]
    packed["n_two_horiz"] = n_two
    packed["combine"] = how
    return pga, pgv, pga_rp


def interpolate_env(env, packed):
    lons = np.asarray([s["lon"] for s in packed["stations"]], dtype=np.float32)
    lats = np.asarray([s["lat"] for s in packed["stations"]], dtype=np.float32)
    grid_lons = np.linspace(packed["lon_min"], packed["lon_max"], GRID_SIZE, dtype=np.float32)
    grid_lats = np.linspace(packed["lat_min"], packed["lat_max"], GRID_SIZE, dtype=np.float32)
    grid_lon, grid_lat = np.meshgrid(grid_lons, grid_lats)
    fields = np.zeros((env.shape[0], GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for ti in range(env.shape[0]):
        fields[ti] = gauss_interp(lons, lats, env[ti], grid_lon, grid_lat)
    return fields


def pixel_size_km(eq_lat):
    dy = RESOLUTION * 111.0
    dx = RESOLUTION * 111.0 * np.cos(np.radians(eq_lat))
    return float(0.5 * (dx + dy))


def strike_delta(mask_a, mask_b, min_pixels=20, fallback_pixels=5):
    """Ellipse Δθ; drop min_pixels if the mask is too small (Tottori FinDer)."""
    import sys
    from pathlib import Path

    nf = Path("/root/autodl-tmp/nature_figures")
    if str(nf) not in sys.path:
        sys.path.insert(0, str(nf))
    from geo import delta_theta, ellipse_major_angle

    def _ang(mask):
        a = ellipse_major_angle(mask, min_pixels=min_pixels)
        if not np.isfinite(a):
            a = ellipse_major_angle(mask, min_pixels=fallback_pixels)
        return a

    return delta_theta(_ang(mask_a), _ang(mask_b))
