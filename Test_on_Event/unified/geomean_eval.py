#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FaultTrans PGV with two-horizontal geometric mean.

Does NOT touch Test_on_Event live pred_*.npy.
Writes only under Test_on_Event/unified/outputs/geomean/.

GM definition (Beyer & Bommer 2006 GMxy):
  IM = sqrt(IM_E * IM_N), peaks taken independently on each filtered
  horizontal trace, then combined. Not sample-wise GM, not 3-component.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path("/root/autodl-tmp")
HERE = ROOT / "Test_on_Event" / "unified"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "nature_figures"))

from fields import (  # noqa: E402
    EVENTS,
    acc_to_vel,
    interpolate_env,
    load_horizontals,
    parse_event_time,
    preprocess_acc,
    rec_offset_s,
    strike_delta,
)
from geo import compute_metrics  # noqa: E402
from infer import CKPT_PGV, gt_masks, load_model, load_norm, run_ft  # noqa: E402

OUT = HERE / "outputs" / "geomean"
SAFE = ["kumamoto", "Miyagi", "Tottori", "ridgecrest", "chichi"]
PAPER = {
    ("kumamoto", "RP"): (0.490, 12.8),
    ("kumamoto", "Curved"): (0.564, 11.3),
    ("Miyagi", "RP"): (0.373, 40.2),
    ("Tottori", "RP"): (0.304, 20.7),
    ("ridgecrest", "JIN"): (0.383, 13.1),
    ("ridgecrest", "ROSS"): (0.352, 11.5),
    ("ridgecrest", "XU"): (0.518, 7.3),
    ("chichi", "CHI"): (0.587, 9.6),
    ("chichi", "MA"): (0.472, 16.1),
    ("chichi", "ZENG"): (0.418, 9.7),
    ("Menyuan", "XU/PKU"): (0.281, 26.5),
    ("Menyuan", "XU"): (0.281, 26.5),
}


def eval_clock_safe(src: Path):
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


def component_traces(cfg):
    src = cfg["src"]
    meta, times, t_first, t_end, final_idx = eval_clock_safe(src)
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


def envelopes_geomean(packed, win_s=2.0):
    stations = packed["stations"]
    times = packed["times"]
    n, t_n = len(stations), len(times)
    pga = np.full((t_n, n), np.nan, dtype=np.float32)
    pgv = np.full((t_n, n), np.nan, dtype=np.float32)
    n_two = 0
    for j, s in enumerate(stations):
        sr, t0 = float(s["sr"]), float(s["t0_s"])
        pgas, pgvs = [], []
        for acc, vel in zip(s["accs"], s["vels"]):
            a, v = _one_env(times, sr, t0, acc, vel, win_s=win_s)
            pgas.append(a)
            pgvs.append(v)
        if len(pgas) >= 2:
            n_two += 1
            pga[:, j] = np.sqrt(np.maximum(pgas[0], 0.0) * np.maximum(pgas[1], 0.0))
            pgv[:, j] = np.sqrt(np.maximum(pgvs[0], 0.0) * np.maximum(pgvs[1], 0.0))
        else:
            pga[:, j] = pgas[0]
            pgv[:, j] = pgvs[0]
    packed["n_two_horiz"] = n_two
    return pga, pgv


def envelopes_max(packed, win_s=2.0):
    """Sample-wise max(|EW|, |NS|) then the same running-peak as Japan/Ridgecrest."""
    stations = packed["stations"]
    times = packed["times"]
    n, t_n = len(stations), len(times)
    pga = np.full((t_n, n), np.nan, dtype=np.float32)
    pgv = np.full((t_n, n), np.nan, dtype=np.float32)
    n_two = 0
    for j, s in enumerate(stations):
        sr, t0 = float(s["sr"]), float(s["t0_s"])
        n_two += int(len(s["accs"]) >= 2)
        acc = np.maximum.reduce(s["accs"])
        vel = np.maximum.reduce(s["vels"])
        pga[:, j], pgv[:, j] = _one_env(times, sr, t0, acc, vel, win_s=win_s)
    packed["n_two_horiz"] = n_two
    return pga, pgv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", nargs="*", default=SAFE)
    parser.add_argument("--combine", choices=("geomean", "max"), default="geomean")
    args = parser.parse_args()
    out = HERE / "outputs" / args.combine
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_norm("pgv")
    model = load_model(CKPT_PGV, device)
    env_fn = envelopes_geomean if args.combine == "geomean" else envelopes_max
    rows = []
    for key in args.events:
        cfg = EVENTS[key]
        packed = component_traces(cfg)
        pga, pgv = env_fn(packed)
        fields = interpolate_env(pgv, packed)
        committed, binary = run_ft(fields, packed["times"], packed, model, device, mean, std)
        dest = out / key
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / "pred_binary.npy", binary)
        np.save(dest / "pred_prob.npy", committed.astype(np.float32))
        np.save(dest / "pgv_fields.npy", fields)
        idx = packed["final_idx"]
        gts = gt_masks(key, packed)
        print(
            f"=== {key} combine={args.combine} stations={len(packed['stations'])} "
            f"two_horiz={packed['n_two_horiz']} idx={idx} "
            f"t={float(packed['times'][idx]):.3f} ==="
        )
        for gt_name, gt in gts.items():
            m = compute_metrics(binary[idx], gt)
            dth = strike_delta(binary[idx], gt)
            paper = PAPER.get((key, gt_name), (float("nan"), float("nan")))
            row = dict(
                event=key,
                gt=gt_name,
                combine=args.combine,
                n_stations=len(packed["stations"]),
                n_two_horiz=int(packed["n_two_horiz"]),
                final_idx=int(idx),
                iou=float(m["IoU"]),
                f1=float(m["F1"]),
                precision=float(m["Precision"]),
                recall=float(m["Recall"]),
                dtheta=float(dth),
                paper_iou=paper[0],
                paper_dtheta=paper[1],
            )
            rows.append(row)
            print(
                f"  {gt_name:8s} IoU={m['IoU']:.3f} (table {paper[0]:.3f})  "
                f"Δθ={dth:.1f} (table {paper[1]:.1f})"
            )
    path = out / "metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("[done]", datetime.now().isoformat(), "wrote", path)


if __name__ == "__main__":
    main()
