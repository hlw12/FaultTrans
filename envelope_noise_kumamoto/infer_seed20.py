#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CPU-cgroup-safe inference for σ=0.4 seed 20261220. No matplotlib."""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp")
HERE = ROOT / "envelope_noise_kumamoto"
OUT = HERE / "outputs"
MASKDIR = OUT / "masks"
UNIFIED = ROOT / "Test_on_Event" / "unified"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UNIFIED))
sys.path.insert(0, str(ROOT / "CmpFinDer"))
sys.path.insert(0, str(ROOT / "nature_figures"))
sys.path.insert(0, str(HERE))

from fields import EVENTS, build_station_traces, envelopes, interpolate_env  # noqa: E402
from finder_native import build_japan  # noqa: E402
from infer import CKPT_PGV, gt_masks, load_model, load_norm, run_finder, run_ft  # noqa: E402
from model import HybridFeatureDecoder  # noqa: E402
from run_sweep import gains_for, interpolate_finder, loc_key, pga_envelopes_finder, score_curved  # noqa: E402

_ORIG_DECODER = HybridFeatureDecoder.forward


def _chunked_decoder_forward(self, x, chunk: int = 2):
    """Same per-frame decoder, chunked so 2 GB cgroup does not OOM at T×150²."""
    _b, t_n, _l, _d = x.shape
    parts = []
    for t0 in range(0, t_n, chunk):
        parts.append(_ORIG_DECODER(self, x[:, t0 : t0 + chunk]))
    return torch.cat(parts, dim=1)


HybridFeatureDecoder.forward = _chunked_decoder_forward



def rss_mb():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except Exception:
        return float("nan")
    return float("nan")


def log(msg):
    print(f"{msg}  rss={rss_mb():.0f} MB", flush=True)


def load_eps(sigma, seed):
    z = np.load(OUT / "gains" / f"eps_sigma{sigma:.1f}_seed{int(seed)}.npz")
    keys = [loc_key(lon, lat) for lon, lat in zip(z["keys_lon"].tolist(), z["keys_lat"].tolist())]
    return dict(zip(keys, np.asarray(z["eps"], np.float32).tolist()))


def drop_waveforms(packed):
    for s in packed.get("stations", []):
        for k in ("acc", "vel", "disp", "ew", "ns", "ud"):
            s.pop(k, None)


def main():
    torch.set_num_threads(4)
    MASKDIR.mkdir(parents=True, exist_ok=True)
    log("start")
    device = torch.device("cpu")

    ft_packed = build_station_traces(EVENTS["kumamoto"])
    log("ft traces")
    _pga_unused, pgv_env, _ = envelopes(ft_packed)
    log("pgv env")
    drop_waveforms(ft_packed)
    gc.collect()

    fd_packed, _fd0 = build_japan("kumamoto")
    log("finder packed")
    pga_env = pga_envelopes_finder(fd_packed["stations"], fd_packed["times"], fd_packed["t_first"])
    log("pga env")
    drop_waveforms(fd_packed)
    gc.collect()

    eps_by_key = load_eps(0.4, 20261220)
    pgv_f = interpolate_env(pgv_env * gains_for(ft_packed["stations"], eps_by_key)[None, :], ft_packed)
    del pgv_env
    log("pgv field")
    pga_f = interpolate_finder(
        pga_env * gains_for(fd_packed["stations"], eps_by_key)[None, :],
        fd_packed["stations"],
        fd_packed,
    )
    del pga_env
    gc.collect()
    log("pga field")

    gt = gt_masks("kumamoto", ft_packed)["Curved"]
    idx = int(ft_packed["final_idx"])
    mean, std = load_norm("pgv")
    log("load model")
    model = load_model(CKPT_PGV, device)
    log("run_ft")
    _c, binary = run_ft(pgv_f, ft_packed["times"], ft_packed, model, device, mean, std)
    del model, _c, pgv_f
    gc.collect()
    log("run_finder")
    _res, finder_masks, _pk = run_finder(pga_f, fd_packed)
    del pga_f, _res
    gc.collect()
    iou, f1, ft_d, fd_d = score_curved(binary, finder_masks, idx, gt)
    log(f"IoU={iou:.3f} FT Δθ={ft_d:.1f} FinDer Δθ={fd_d:.1f}")
    if abs(iou - 0.626) > 0.02 or abs(ft_d - 8.5) > 1.5 or abs(fd_d - 47.6) > 1.0:
        raise SystemExit("scores do not reprint sweep.csv seed 20261220")
    np.save(MASKDIR / "sigma04_seed20261220_ft.npy", binary[idx].astype(np.uint8))
    np.save(MASKDIR / "sigma04_seed20261220_fd.npy", finder_masks[idx].astype(np.uint8))
    np.save(MASKDIR / "gt_curved.npy", np.asarray(gt).astype(np.uint8))
    log("saved masks")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
