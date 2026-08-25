#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dump last-frame PGA/PGV fields: σ=0 and σ=0.4 seed 20261220. No matplotlib."""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp")
HERE = ROOT / "envelope_noise_kumamoto"
OUT = HERE / "outputs"
MASKDIR = OUT / "masks"
UNIFIED = ROOT / "Test_on_Event" / "unified"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UNIFIED))
sys.path.insert(0, str(ROOT / "CmpFinDer"))
sys.path.insert(0, str(HERE))

from fields import EVENTS, build_station_traces, envelopes, interpolate_env  # noqa: E402
from finder_native import build_japan  # noqa: E402
from run_sweep import (  # noqa: E402
    gains_for,
    interpolate_finder,
    loc_key,
    pga_envelopes_finder,
)


def load_eps(sigma, seed):
    z = np.load(OUT / "gains" / f"eps_sigma{sigma:.1f}_seed{int(seed)}.npz")
    keys = [loc_key(lon, lat) for lon, lat in zip(z["keys_lon"].tolist(), z["keys_lat"].tolist())]
    return dict(zip(keys, np.asarray(z["eps"], np.float32).tolist()))


def drop_waveforms(packed):
    for s in packed.get("stations", []):
        for k in ("acc", "vel", "disp", "ew", "ns", "ud"):
            s.pop(k, None)


def last_frame(fields, idx):
    return np.asarray(fields[idx], np.float32)


def main():
    MASKDIR.mkdir(parents=True, exist_ok=True)
    print("build traces", flush=True)
    ft_packed = build_station_traces(EVENTS["kumamoto"])
    _a, pgv_env, _ = envelopes(ft_packed)
    drop_waveforms(ft_packed)
    fd_packed, _ = build_japan("kumamoto")
    pga_env = pga_envelopes_finder(fd_packed["stations"], fd_packed["times"], fd_packed["t_first"])
    drop_waveforms(fd_packed)
    gc.collect()
    idx = int(ft_packed["final_idx"])
    print(f"idx={idx} t={float(ft_packed['times'][idx]):.3f}", flush=True)

    ones = {loc_key(s["lon"], s["lat"]): 0.0 for s in ft_packed["stations"]}
    pgv0 = last_frame(interpolate_env(pgv_env * gains_for(ft_packed["stations"], ones)[None, :], ft_packed), idx)
    pga0 = last_frame(
        interpolate_finder(pga_env * gains_for(fd_packed["stations"], ones)[None, :], fd_packed["stations"], fd_packed),
        idx,
    )
    eps = load_eps(0.4, 20261220)
    pgv1 = last_frame(interpolate_env(pgv_env * gains_for(ft_packed["stations"], eps)[None, :], ft_packed), idx)
    pga1 = last_frame(
        interpolate_finder(pga_env * gains_for(fd_packed["stations"], eps)[None, :], fd_packed["stations"], fd_packed),
        idx,
    )
    lons = np.asarray([s["lon"] for s in ft_packed["stations"]], np.float64)
    lats = np.asarray([s["lat"] for s in ft_packed["stations"]], np.float64)
    gain = gains_for(ft_packed["stations"], eps)
    np.save(MASKDIR / "pga_sigma0.npy", pga0)
    np.save(MASKDIR / "pga_sigma04_seed20261220.npy", pga1)
    np.save(MASKDIR / "pgv_sigma0.npy", pgv0)
    np.save(MASKDIR / "pgv_sigma04_seed20261220.npy", pgv1)
    np.savez(
        MASKDIR / "stations_gain_seed20261220.npz",
        lon=lons,
        lat=lats,
        gain=gain.astype(np.float32),
        eps=np.log(gain).astype(np.float32),
    )
    rel_pga = float(np.nanmax(np.abs(pga1 - pga0)) / (np.nanmax(pga0) + 1e-12))
    rel_pgv = float(np.nanmax(np.abs(pgv1 - pgv0)) / (np.nanmax(pgv0) + 1e-12))
    print(
        f"PGA max {float(pga0.max()):.3g} -> {float(pga1.max()):.3g}  "
        f"max|Δ|/max0={rel_pga:.3f}",
        flush=True,
    )
    print(
        f"PGV max {float(pgv0.max()):.3g} -> {float(pgv1.max()):.3g}  "
        f"max|Δ|/max0={rel_pgv:.3f}",
        flush=True,
    )
    print("gain min/median/max", float(gain.min()), float(np.median(gain)), float(gain.max()), flush=True)
    print("[fields saved]", MASKDIR, flush=True)


if __name__ == "__main__":
    main()
