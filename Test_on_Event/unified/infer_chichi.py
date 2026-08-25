#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Chi-Chi FaultTrans PGV with two-horizontal geometric mean (GMxy).

Independent running-peaks on EW and NS, then sqrt(IM_E * IM_N)
(Beyer & Bommer 2006). Japan / Ridgecrest / Menyuan belong in infer.py.

Does not write live Test_on_Event pred_*.npy unless --write-live.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path("/root/autodl-tmp")
HERE = Path("/root/autodl-tmp/Test_on_Event/unified")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nature_figures"))
sys.path.insert(0, str(HERE))

from fields import CHICHI_EVENTS  # noqa: E402
from infer import CKPT_PGV, OUT, load_model, load_norm, run_event, write_csv  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", nargs="*", default=None)
    parser.add_argument(
        "--write-live",
        action="store_true",
        help="Overwrite Test_on_Event/chichi live pred_*.npy (default writes under outputs/infer_chichi/).",
    )
    args = parser.parse_args()
    keys = list(args.events or CHICHI_EVENTS)
    bad = [k for k in keys if k not in CHICHI_EVENTS]
    if bad:
        raise SystemExit(f"Non-Chi-Chi events belong in infer.py (max combine): {bad}")
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dpgv_mean, dpgv_std = load_norm("pgv")
    model = load_model(CKPT_PGV, device)
    rows = []
    for key in keys:
        rows.extend(
            run_event(
                key,
                model,
                device,
                dpgv_mean,
                dpgv_std,
                do_finder=False,
                write_live=args.write_live,
                combine="geomean",
            )
        )
        write_csv(OUT / "metrics_chichi.csv", rows)
        (OUT / "metrics_chichi.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("[done]", datetime.now().isoformat(), "n_rows", len(rows))
    for r in rows:
        print(
            f"{r['event']:12s} {r['gt']:8s}  IoU={r['ft_iou']:.3f}  "
            f"Δθ={r['ft_dtheta']:.1f}  IoUmax={r['ft_ioumax']:.3f} @{r['t_ioumax']:.1f}s"
        )


if __name__ == "__main__":
    main()
