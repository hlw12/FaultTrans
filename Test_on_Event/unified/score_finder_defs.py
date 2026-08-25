#!/usr/bin/env python
"""Score native FinDer with both Table-2 Δθ definitions. Does not write npy."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp")
HERE = ROOT / "Test_on_Event" / "unified"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "CmpFinDer"))
sys.path.insert(0, str(ROOT / "nature_figures"))
sys.path.insert(0, str(HERE))

from fields import strike_delta  # noqa: E402
from finder_native import (  # noqa: E402
    EVENTS,
    PAPER,
    eval_clock,
    geographic_strike_to_cv,
    geo_strike_delta,
)
from infer import gt_masks, run_finder  # noqa: E402


def packed_of(key):
    src = EVENTS[key]["src"]
    meta, times, t_first, t_end, final_idx = eval_clock(src)
    return dict(
        src=src,
        times=times,
        t_first=t_first,
        t_end=t_end,
        final_idx=final_idx,
        eq_lat=float(meta["eq_lat"]),
        eq_lon=float(meta["eq_lon"]),
        lon_min=float(meta["lon_min"]),
        lon_max=float(meta["lon_max"]),
        lat_min=float(meta["lat_min"]),
        lat_max=float(meta["lat_max"]),
    )


def main():
    print(
        f"{'event':<12} {'GT':<8} {'ellipse':>8} {'geo-strike':>10} "
        f"{'paper':>6} {'ell-paper':>9} {'geo-paper':>9}  strike_geo  line_px"
    )
    for key in ("kumamoto", "Tottori", "ridgecrest"):
        packed = packed_of(key)
        src = packed["src"]
        field_path = src / "pga_fields_finder.npy"
        if not field_path.exists():
            print(f"{key}: missing {field_path}")
            continue
        fields = np.load(field_path)
        res, masks, _ = run_finder(fields, packed)
        idx = packed["final_idx"]
        line = masks[idx]
        strike = float(getattr(res[idx], "strike_deg", np.nan))
        length = float(getattr(res[idx], "length_km", np.nan))
        print(
            f"-- {key}  idx={idx}  t={float(packed['times'][idx]):.3f}s  "
            f"strike_geo={strike:.1f}  L={length:.1f} km  "
            f"strike_cv={geographic_strike_to_cv(strike):.1f}  "
            f"line_px={int(line.sum())}"
        )
        for name, gt in gt_masks(key, packed).items():
            d_ell = float(strike_delta(line, gt))
            d_geo = float(geo_strike_delta(strike, gt))
            paper = PAPER.get((key, name))
            e_off = abs(d_ell - paper) if paper is not None else np.nan
            g_off = abs(d_geo - paper) if paper is not None else np.nan
            print(
                f"{key:<12} {name:<8} {d_ell:8.1f} {d_geo:10.1f} "
                f"{paper:6} {e_off:9.1f} {g_off:9.1f}  {strike:7.1f}  {int(line.sum()):7d}"
            )


if __name__ == "__main__":
    main()
