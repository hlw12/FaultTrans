#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kumamoto envelope-noise: Table 1 FaultTrans + Table 2 FinDer.

Isolated workspace. Does not write Test_on_Event live preds, CmpFinDer, or LaTeX.

σ=0 must reprint:
  FaultTrans Curved IoU 0.564, Δθ 11.3°  (unified PGV, zeros dropped)
  FinDer Curved Δθ 12.0°               (finder_native Japan PGA, zeros kept)

Same station-wise ε_i multiplies both envelopes. Training frozen.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp")
HERE = ROOT / "envelope_noise_kumamoto"
OUT = HERE / "outputs"
GAINS = OUT / "gains"
UNIFIED = ROOT / "Test_on_Event" / "unified"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UNIFIED))
sys.path.insert(0, str(ROOT / "CmpFinDer"))
sys.path.insert(0, str(ROOT / "nature_figures"))

from fields import EVENTS, build_station_traces, envelopes, interpolate_env  # noqa: E402
from finder_native import build_japan, gauss_interp  # noqa: E402
from infer import (  # noqa: E402
    CKPT_PGV,
    GRID_SIZE,
    gt_masks,
    load_model,
    load_norm,
    run_finder,
    run_ft,
)
from geo import compute_metrics  # noqa: E402
from fields import strike_delta  # noqa: E402

SIGMAS = (0.0, 0.2, 0.4)
N_SEEDS = 10
SEED_BASE = 20260818

PAPER_FT_IOU = 0.564
PAPER_FT_DTHETA = 11.3
PAPER_FINDER_DTHETA = 12.0
GATE_IOU_TOL = 0.02
GATE_FT_DTHETA_TOL = 1.5
GATE_FINDER_DTHETA_TOL = 1.0


def loc_key(lon, lat):
    return (round(float(lon), 4), round(float(lat), 4))


def pga_envelopes_finder(stations, times, t_first):
    """Same running-max clock as finder_native.fill_gauss_pga."""
    n, t_n = len(stations), len(times)
    env = np.zeros((t_n, n), np.float32)
    for j, s in enumerate(stations):
        sr = float(s["sr"])
        t0 = float(s["t0_s"])
        acc = s["acc"]
        for ti, t_inf in enumerate(times):
            t_abs = float(t_inf) + float(t_first)
            idx = int(np.floor((t_abs - t0) * sr))
            if idx < 0:
                continue
            idx = min(idx, len(acc) - 1)
            env[ti, j] = float(np.max(acc[: idx + 1]))
    return env


def interpolate_finder(env, stations, packed):
    """Japan Table 2 interpolator: zeros are kept and interpolated."""
    lons = np.asarray([s["lon"] for s in stations], np.float32)
    lats = np.asarray([s["lat"] for s in stations], np.float32)
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(packed["lon_min"], packed["lon_max"], GRID_SIZE, np.float32),
        np.linspace(packed["lat_min"], packed["lat_max"], GRID_SIZE, np.float32),
    )
    fields = np.zeros((env.shape[0], GRID_SIZE, GRID_SIZE), np.float32)
    for ti in range(env.shape[0]):
        fields[ti] = gauss_interp(lons, lats, env[ti], grid_lon, grid_lat)
    return fields


def draw_eps(keys, sigma, seed):
    keys = sorted(keys)
    rng = np.random.default_rng(int(seed))
    if sigma <= 0:
        vals = np.zeros(len(keys), np.float32)
    else:
        vals = rng.normal(0.0, float(sigma), size=len(keys)).astype(np.float32)
    return dict(zip(keys, vals.tolist())), keys, vals


def gains_for(stations, eps_by_key):
    return np.exp(
        np.asarray([eps_by_key[loc_key(s["lon"], s["lat"])] for s in stations], np.float32)
    )


def score_curved(binary, finder_masks, idx, gt):
    pred = binary[idx]
    m = compute_metrics(pred, gt)
    ft_d = float(strike_delta(pred, gt))
    line = finder_masks[idx]
    fd_d = float(strike_delta(line, gt))
    return m["IoU"], m["F1"], ft_d, fd_d


def one_run(ft_packed, pgv_env, fd_packed, pga_env, model, device, mean, std, gt, sigma, seed):
    keys = {loc_key(s["lon"], s["lat"]) for s in ft_packed["stations"]}
    keys |= {loc_key(s["lon"], s["lat"]) for s in fd_packed["stations"]}
    eps_by_key, key_order, eps_vals = draw_eps(keys, sigma, seed)
    pgv_f = interpolate_env(pgv_env * gains_for(ft_packed["stations"], eps_by_key)[None, :], ft_packed)
    pga_f = interpolate_finder(
        pga_env * gains_for(fd_packed["stations"], eps_by_key)[None, :],
        fd_packed["stations"],
        fd_packed,
    )
    _c, binary = run_ft(pgv_f, ft_packed["times"], ft_packed, model, device, mean, std)
    _res, finder_masks, _pk = run_finder(pga_f, fd_packed)
    iou, f1, ft_d, fd_d = score_curved(binary, finder_masks, ft_packed["final_idx"], gt)
    GAINS.mkdir(parents=True, exist_ok=True)
    np.savez(
        GAINS / f"eps_sigma{sigma:.1f}_seed{int(seed)}.npz",
        keys_lon=np.asarray([k[0] for k in key_order], np.float64),
        keys_lat=np.asarray([k[1] for k in key_order], np.float64),
        eps=eps_vals,
    )
    row = dict(
        sigma=float(sigma),
        seed=int(seed),
        iou=float(iou),
        f1=float(f1),
        ft_dtheta=ft_d,
        finder_dtheta=fd_d,
        n_ft=len(ft_packed["stations"]),
        n_fd=len(fd_packed["stations"]),
        n_keys=len(keys),
        final_idx=int(ft_packed["final_idx"]),
        t_eval=float(ft_packed["times"][ft_packed["final_idx"]]),
    )
    print(
        f"  σ={sigma:.1f} seed={seed}  IoU={iou:.3f}  "
        f"FT Δθ={ft_d:.1f}  FinDer Δθ={fd_d:.1f}"
    )
    return row


def gate_ok(row) -> bool:
    iou_ok = abs(row["iou"] - PAPER_FT_IOU) <= GATE_IOU_TOL
    ft_ok = abs(row["ft_dtheta"] - PAPER_FT_DTHETA) <= GATE_FT_DTHETA_TOL
    fd_ok = abs(row["finder_dtheta"] - PAPER_FINDER_DTHETA) <= GATE_FINDER_DTHETA_TOL
    print(
        f"[gate] FT IoU={row['iou']:.3f} (paper {PAPER_FT_IOU})  "
        f"FT Δθ={row['ft_dtheta']:.1f} (paper {PAPER_FT_DTHETA})  "
        f"FinDer Δθ={row['finder_dtheta']:.1f} (paper {PAPER_FINDER_DTHETA})"
    )
    print(f"[gate] iou_ok={iou_ok}  ft_dtheta_ok={ft_ok}  finder_dtheta_ok={fd_ok}")
    if not fd_ok:
        print("[gate] FAIL: FinDer σ=0 must reprint Table 2 (12.0°). Stop.")
        return False
    if not (iou_ok and ft_ok):
        print("[gate] FAIL: FaultTrans σ=0 must reprint Table 1 Curved. Stop.")
        return False
    return True


def summarize(rows):
    by = {}
    for r in rows:
        by.setdefault(r["sigma"], []).append(r)
    out = []
    for sigma, group in sorted(by.items()):
        def stats(key):
            xs = np.asarray([g[key] for g in group], np.float64)
            xs = xs[np.isfinite(xs)]
            if len(xs) == 0:
                return float("nan"), float("nan")
            return float(xs.mean()), float(xs.std(ddof=1) if len(xs) > 1 else 0.0)

        iou_m, iou_s = stats("iou")
        ft_m, ft_s = stats("ft_dtheta")
        fd_m, fd_s = stats("finder_dtheta")
        out.append(
            dict(
                sigma=sigma,
                n=len(group),
                iou_mean=iou_m,
                iou_std=iou_s,
                ft_dtheta_mean=ft_m,
                ft_dtheta_std=ft_s,
                finder_dtheta_mean=fd_m,
                finder_dtheta_std=fd_s,
            )
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--sigmas", type=float, nargs="*", default=list(SIGMAS))
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_norm("pgv")
    model = load_model(CKPT_PGV, device)

    ft_packed = build_station_traces(EVENTS["kumamoto"])
    _pga_unused, pgv_env, _ = envelopes(ft_packed)
    fd_packed, fd_fields0 = build_japan("kumamoto")
    pga_env = pga_envelopes_finder(fd_packed["stations"], fd_packed["times"], fd_packed["t_first"])
    if not np.allclose(ft_packed["times"], fd_packed["times"]):
        raise SystemExit("FT and FinDer time axes differ")
    if int(ft_packed["final_idx"]) != int(fd_packed["final_idx"]):
        raise SystemExit("FT and FinDer final_idx differ")

    gts = gt_masks("kumamoto", ft_packed)
    gt = gts["Curved"]
    pga_check = interpolate_finder(pga_env, fd_packed["stations"], fd_packed)
    max_diff = float(np.max(np.abs(pga_check - fd_fields0)))
    print(
        f"Kumamoto FT stations={len(ft_packed['stations'])}  "
        f"FinDer stations={len(fd_packed['stations'])}  "
        f"frames={len(ft_packed['times'])}  idx={ft_packed['final_idx']}  "
        f"t={float(ft_packed['times'][ft_packed['final_idx']]):.3f}  "
        f"PGA env vs fill_gauss max|Δ|={max_diff:.4g}"
    )

    rows = []
    for sigma in args.sigmas:
        n_rep = 1 if sigma == 0.0 or args.gate_only else args.n_seeds
        if args.gate_only and sigma != 0.0:
            continue
        for i in range(n_rep):
            seed = SEED_BASE if sigma == 0.0 else SEED_BASE + int(round(sigma * 10)) * 100 + i
            row = one_run(
                ft_packed, pgv_env, fd_packed, pga_env, model, device, mean, std, gt, sigma, seed
            )
            rows.append(row)
            if sigma == 0.0 and not gate_ok(row):
                (OUT / "gate.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                raise SystemExit("σ=0 gate failed")
            if sigma == 0.0:
                (OUT / "gate.json").write_text(json.dumps({"pass": True, **row}, indent=2), encoding="utf-8")
                print("[gate] PASS")

    csv_path = OUT / "sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    (OUT / "sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("[summary]")
    for s in summary:
        print(
            f"  σ={s['sigma']:.1f} n={s['n']}  "
            f"IoU={s['iou_mean']:.3f}±{s['iou_std']:.3f}  "
            f"FT Δθ={s['ft_dtheta_mean']:.1f}±{s['ft_dtheta_std']:.1f}  "
            f"FinDer Δθ={s['finder_dtheta_mean']:.1f}±{s['finder_dtheta_std']:.1f}"
        )
    print("[noise done]", datetime.now().isoformat(), csv_path)


if __name__ == "__main__":
    main()
