#!/usr/bin/env python
"""Score Menyuan max(|EW|,|NS|) pred without touching live notebook files."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp")
HERE = ROOT / "Test_on_Event" / "unified"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "nature_figures"))

from fields import eval_clock  # noqa: E402
from geo import compute_metrics, parse_pku, rasterize_subfaults  # noqa: E402
from infer import gt_masks, strike_delta  # noqa: E402
from geomean_eval import eval_clock_safe  # noqa: E402

SRC = ROOT / "Test_on_Event" / "menyuan" / "Menyuan"
PRED = HERE / "outputs" / "max" / "Menyuan" / "pred_binary.npy"


def tv_mask(df, t_inf, t_first, lat, lon):
    if df is None or len(df) == 0 or "ttrg_s" not in df.columns:
        return None
    t_origin = float(t_inf) + float(t_first)
    dfa = df[np.isfinite(df["ttrg_s"]) & (df["ttrg_s"] <= t_origin)].copy()
    mask, _ = rasterize_subfaults(dfa, lat, lon)
    return mask


def main():
    pred = np.load(PRED)
    meta, times, t_first, t_end, idx = eval_clock_safe(SRC)
    packed = dict(
        eq_lat=float(meta["eq_lat"]),
        eq_lon=float(meta["eq_lon"]),
        lon_min=float(meta["lon_min"]),
        lon_max=float(meta["lon_max"]),
        lat_min=float(meta["lat_min"]),
        lat_max=float(meta["lat_max"]),
    )
    gts = gt_masks("Menyuan", packed)
    live = np.load(SRC / "pred_binary.npy")
    df = parse_pku(SRC / "20220107174530_Menyuan_Rupture_Info.txt")
    print(f"idx={idx} t={float(times[idx]):.3f} t_end={t_end:.3f} t_first={t_first:.3f}")
    print(f"pred_max {pred.shape} live {live.shape}")
    for name, gt in gts.items():
        m = compute_metrics(pred[idx], gt)
        ml = compute_metrics(live[min(idx, len(live) - 1)], gt)
        print(
            f"{name} gt_px={int(gt.sum())}  "
            f"max pred_px={int(pred[idx].sum())} IoU={m['IoU']:.3f} F1={m['F1']:.3f} "
            f"P={m['Precision']:.3f} R={m['Recall']:.3f} dth={strike_delta(pred[idx], gt):.1f}  "
            f"| notebook pred_px={int(live[min(idx, len(live)-1)].sum())} "
            f"IoU={ml['IoU']:.3f} dth={strike_delta(live[min(idx, len(live)-1)], gt):.1f}"
        )
        iou_tv = []
        for k in range(len(times)):
            if times[k] > t_end:
                iou_tv.append(-1.0)
                continue
            gtt = tv_mask(df, times[k], t_first, packed["eq_lat"], packed["eq_lon"])
            iou_tv.append(0.0 if gtt is None else compute_metrics(pred[k], gtt)["IoU"])
        valid = np.array(iou_tv) >= 0
        kbest = int(np.argmax(np.where(valid, iou_tv, -np.inf)))
        print(f"  IoUmax_tv={iou_tv[kbest]:.3f} @ t={float(times[kbest]):.1f}s idx={kbest}")
        iou_st = [
            compute_metrics(pred[k], gt)["IoU"] if times[k] <= t_end else -1.0
            for k in range(len(times))
        ]
        ks = int(np.argmax(np.where(np.array(iou_st) >= 0, iou_st, -np.inf)))
        print(f"  IoUmax_static={iou_st[ks]:.3f} @ t={float(times[ks]):.1f}s")


if __name__ == "__main__":
    main()
