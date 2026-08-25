#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified PGA FaultTrans inference for fig06.

Uses the same eval preprocessor as PGV (fields.py). Writes only under
PGA_real_eval/outputs/. Does not touch Test_on_Event predictions.
Chi-Chi and Menyuan are skipped: their H5 cannot be read by the unified
loader, and their original PGA notebook outputs stay in place.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp")
HERE = Path("/root/autodl-tmp/Test_on_Event/unified")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "nature_figures"))

from fields import EVENTS, build_station_traces, envelopes, interpolate_env, strike_delta  # noqa: E402
from infer import load_model, load_norm, run_ft, gt_masks  # noqa: E402
from geo import compute_metrics  # noqa: E402

PGA_OUT = ROOT / "PGA_real_eval" / "outputs"
CKPT_PGA = ROOT / "checkpoints_pga" / "pga" / "best_iou_rule_hybrid.pth"
SAFE = ["kumamoto", "Miyagi", "Tottori", "ridgecrest"]


def save_pga(key, committed, binary, pga_fields, packed):
    dest = PGA_OUT / key
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "pred_binary.npy", binary)
    np.save(dest / "pred_prob.npy", committed.astype(np.float32))
    np.save(dest / "pga_fields.npy", pga_fields)
    src = packed["src"]
    for name in ("event_metadata.npz", "times_mid_valid.npy"):
        p = src / name
        if p.exists():
            shutil.copy2(p, dest / name)
    print(f"  wrote {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", nargs="*", default=SAFE)
    args = parser.parse_args()
    keys = [k for k in args.events if k in SAFE]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_norm("pga")
    model = load_model(CKPT_PGA, device)
    rows = []
    for key in keys:
        packed = build_station_traces(EVENTS[key])
        print(
            f"=== PGA {key} stations={len(packed['stations'])} "
            f"frames={len(packed['times'])} final_idx={packed['final_idx']} ==="
        )
        pga, _pgv, _rp = envelopes(packed)
        pga_fields = interpolate_env(pga, packed)
        committed, binary = run_ft(pga_fields, packed["times"], packed, model, device, mean, std)
        save_pga(key, committed, binary, pga_fields, packed)
        idx = packed["final_idx"]
        gts = gt_masks(key, packed)
        gt_name = list(gts)[0]
        gt = gts[gt_name] if key != "kumamoto" else gts.get("Curved", gts[gt_name])
        if key == "kumamoto":
            gt_name = "Curved" if "Curved" in gts else gt_name
            gt = gts[gt_name]
        elif key == "ridgecrest" and "XU" in gts:
            gt_name, gt = "XU", gts["XU"]
        m = compute_metrics(binary[idx], gt)
        dth = strike_delta(binary[idx], gt)
        row = dict(event=key, gt=gt_name, iou=m["IoU"], dtheta=dth, final_idx=idx)
        rows.append(row)
        print(f"  PGA FT IoU={m['IoU']:.3f} Δθ={dth:.1f}")
    out = HERE / "outputs" / "pga_ft_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("[pga_ft done]", datetime.now().isoformat(), out)


if __name__ == "__main__":
    main()
