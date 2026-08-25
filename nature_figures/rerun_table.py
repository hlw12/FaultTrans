#!/usr/bin/env python
"""Recompute Table 1 under one protocol: last committed frame inside the
effective evaluation window (inference clock, with inf≈abs correction).

Does not modify Test_on_Event predictions. Writes CSV only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path("/root/autodl-tmp/nature_figures")
sys.path.insert(0, str(HERE))

from catalog import EVENTS  # noqa: E402
from geo import (  # noqa: E402
    compute_metrics,
    load_event_arrays,
    parse_nied,
    parse_pku,
    parse_srcmod_mat,
    rasterize_subfaults,
)

OUT = HERE / "source_data"
OUT.mkdir(parents=True, exist_ok=True)

MIN_PIX_ANGLE = 20
PAPER = {
    ("Kumamoto (2016)", "RP"): dict(IoU=0.371, F1=0.541, Precision=0.421, Recall=0.759, IoUmax=0.425, t_best=20.3, dtheta=19.7),
    ("Kumamoto (2016)", "Curved"): dict(IoU=0.439, F1=0.610, Precision=0.521, Recall=0.737, IoUmax=0.492, t_best=20.3, dtheta=18.2),
    ("Iwate-Miyagi (2008)", "RP"): dict(IoU=0.368, F1=0.538, Precision=0.461, Recall=0.646, IoUmax=0.454, t_best=18.3, dtheta=42.6),
    ("Tottori (2000)", "RP"): dict(IoU=0.306, F1=0.469, Precision=0.306, Recall=1.000, IoUmax=0.434, t_best=6.8, dtheta=4.6),
    ("Ridgecrest (2019)", "JIN"): dict(IoU=0.351, F1=0.519, Precision=0.365, Recall=0.900, IoUmax=0.387, t_best=18.3, dtheta=18.0),
    ("Ridgecrest (2019)", "ROSS"): dict(IoU=0.336, F1=0.503, Precision=0.349, Recall=0.904, IoUmax=0.345, t_best=18.3, dtheta=16.4),
    ("Ridgecrest (2019)", "XU"): dict(IoU=0.521, F1=0.685, Precision=0.636, Recall=0.742, IoUmax=0.521, t_best=19.7, dtheta=12.2),
    ("Menyuan (2022)", "XU/PKU"): dict(IoU=0.281, F1=0.438, Precision=0.308, Recall=0.762, IoUmax=0.365, t_best=8.8, dtheta=26.5),
    ("Chi-Chi (1999)", "CHI"): dict(IoU=0.587, F1=0.740, Precision=0.762, Recall=0.720, IoUmax=0.589, t_best=33.9, dtheta=9.6),
    ("Chi-Chi (1999)", "MA"): dict(IoU=0.472, F1=0.641, Precision=0.702, Recall=0.590, IoUmax=0.481, t_best=32.5, dtheta=16.1),
    ("Chi-Chi (1999)", "ZENG"): dict(IoU=0.418, F1=0.589, Precision=0.699, Recall=0.509, IoUmax=0.439, t_best=30.5, dtheta=9.7),
}


def fitellipse_deg(mask, min_pixels=MIN_PIX_ANGLE):
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if int(mask.sum()) < min_pixels:
        return np.nan
    rows, cols = np.where(mask)
    if len(rows) < 5:
        return np.nan
    pts = np.column_stack([cols, rows]).astype(np.float32)
    try:
        _, (_, _), angle = cv2.fitEllipse(pts)
    except cv2.error:
        return np.nan
    return float(angle % 180.0)


def dtheta(a, b):
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    d = abs(float(a) - float(b)) % 180.0
    return float(min(d, 180.0 - d))


def tv_mask(df, t_inf, t_first, lat, lon):
    if df is None or len(df) == 0 or "ttrg_s" not in df.columns:
        return None
    t_origin = float(t_inf) + float(t_first)
    dfa = df[np.isfinite(df["ttrg_s"]) & (df["ttrg_s"] <= t_origin)].copy()
    mask, _ = rasterize_subfaults(dfa, lat, lon)
    return mask


def peak_within(values, valid, t_axis):
    arr = np.asarray(values, dtype=np.float64)
    score = np.where(valid, arr, -np.inf)
    if not np.isfinite(score).any() or np.all(score == -np.inf):
        return np.nan, np.nan, -1
    i = int(np.nanargmax(score))
    return float(arr[i]), float(t_axis[i]), i


def eval_gt(event_label, gt_name, bundle, df, paper_key):
    pred = bundle["pred"]
    t = bundle["t_axis"]
    valid = bundle["valid"]
    final_idx = bundle["final_idx"]
    lat, lon = bundle["eq_lat"], bundle["eq_lon"]
    t_first = bundle["t_first"]
    gt_static, _ = rasterize_subfaults(df, lat, lon)
    pred_final = pred[final_idx]
    m = compute_metrics(pred_final, gt_static)
    th_p = fitellipse_deg(pred_final)
    th_g = fitellipse_deg(gt_static)
    dt = dtheta(th_p, th_g)

    iou_static = [compute_metrics(pred[k], gt_static)["IoU"] for k in range(len(t))]
    has_tv = df is not None and "ttrg_s" in df.columns and np.isfinite(df["ttrg_s"]).any()
    if has_tv:
        iou_tv = []
        for k in range(len(t)):
            gtt = tv_mask(df, t[k], t_first, lat, lon)
            iou_tv.append(0.0 if gtt is None else compute_metrics(pred[k], gtt)["IoU"])
    else:
        iou_tv = [np.nan] * len(t)

    ioumax_tv, tbest_tv, _ = peak_within(iou_tv, valid, t)
    ioumax_st, tbest_st, _ = peak_within(iou_static, valid, t)

    # last-of-40s (Menyuan old) and uncorrected-inf (Tottori old CSV)
    last_all = compute_metrics(pred[-1], gt_static)
    t_abs = float(bundle["meta"]["effective_end_abs_s"])
    t_inf_raw = float(bundle["meta"]["effective_end_inf_s"]) if "effective_end_inf_s" in bundle["meta"] else t_abs
    raw_valid = t <= t_inf_raw
    if raw_valid.sum() == 0:
        raw_idx = len(t) - 1
    else:
        raw_idx = int(np.where(raw_valid)[0][-1])
    m_raw = compute_metrics(pred[raw_idx], gt_static)

    paper = PAPER.get(paper_key, {})
    row = dict(
        event=event_label,
        gt=gt_name,
        t_end_s=round(float(bundle["t_end"]), 3),
        t_end_raw_s=round(t_inf_raw, 3),
        final_idx=int(final_idx),
        final_t_s=round(float(t[final_idx]), 3),
        gt_pixels=int(gt_static.sum()),
        pred_pixels=int(pred_final.sum()),
        IoU=m["IoU"],
        F1=m["F1"],
        Precision=m["Precision"],
        Recall=m["Recall"],
        TP=m["TP"],
        FP=m["FP"],
        FN=m["FN"],
        dtheta=dt,
        IoUmax_tv=ioumax_tv,
        t_best_tv=tbest_tv,
        IoUmax_static=ioumax_st,
        t_best_static=tbest_st,
        IoU_last40=last_all["IoU"],
        IoU_raw_tend=m_raw["IoU"],
        raw_idx=int(raw_idx),
        paper_IoU=paper.get("IoU", np.nan),
        paper_F1=paper.get("F1", np.nan),
        paper_Prec=paper.get("Precision", np.nan),
        paper_Rec=paper.get("Recall", np.nan),
        paper_IoUmax=paper.get("IoUmax", np.nan),
        paper_tbest=paper.get("t_best", np.nan),
        paper_dtheta=paper.get("dtheta", np.nan),
        has_tv=bool(has_tv),
    )
    return row


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{float(x):.{nd}f}"


def main():
    rows = []

    # Japanese NIED
    for key, label, models in [
        ("kumamoto", "Kumamoto (2016)", [("RP", "nied_rp"), ("Curved", "nied_cv")]),
        ("Miyagi", "Iwate-Miyagi (2008)", [("RP", "nied_rp")]),
        ("Tottori", "Tottori (2000)", [("RP", "nied_rp")]),
    ]:
        ev = EVENTS[key]
        bundle = load_event_arrays(ev["pgv_dir"])
        print(f"\n{label}: t_first={bundle['t_first']:.3f} t_end={bundle['t_end']:.3f} "
              f"final_idx={bundle['final_idx']} t={bundle['t_axis'][bundle['final_idx']]:.2f}")
        for gt_name, attr in models:
            df = parse_nied(ev[attr])
            rows.append(eval_gt(label, gt_name, bundle, df, (label, gt_name)))

    # Ridgecrest SRCMOD
    ev = EVENTS["ridgecrest"]
    bundle = load_event_arrays(ev["pgv_dir"])
    print(f"\nRidgecrest (2019): t_first={bundle['t_first']:.3f} t_end={bundle['t_end']:.3f} "
          f"final_idx={bundle['final_idx']} t={bundle['t_axis'][bundle['final_idx']]:.2f}")
    for gt_name, path in ev["srcmod"].items():
        df = parse_srcmod_mat(path)
        rows.append(eval_gt("Ridgecrest (2019)", gt_name, bundle, df, ("Ridgecrest (2019)", gt_name)))

    # Chi-Chi SRCMOD
    ev = EVENTS["chichi"]
    bundle = load_event_arrays(ev["pgv_dir"])
    print(f"\nChi-Chi (1999): t_first={bundle['t_first']:.3f} t_end={bundle['t_end']:.3f} "
          f"final_idx={bundle['final_idx']} t={bundle['t_axis'][bundle['final_idx']]:.2f}")
    for gt_name, path in ev["srcmod"].items():
        df = parse_srcmod_mat(path)
        rows.append(eval_gt("Chi-Chi (1999)", gt_name, bundle, df, ("Chi-Chi (1999)", gt_name)))

    # Menyuan PKU (table label XU)
    ev = EVENTS["Menyuan"]
    bundle = load_event_arrays(ev["pgv_dir"])
    print(f"\nMenyuan (2022): t_first={bundle['t_first']:.3f} t_end={bundle['t_end']:.3f} "
          f"final_idx={bundle['final_idx']} t={bundle['t_axis'][bundle['final_idx']]:.2f}")
    df = parse_pku(ev["pku_txt"])
    rows.append(eval_gt("Menyuan (2022)", "XU/PKU", bundle, df, ("Menyuan (2022)", "XU/PKU")))

    df_out = pd.DataFrame(rows)
    csv_path = OUT / "table1_rerun.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    print("\n================ TABLE 1 RERUN (last valid frame) ================")
    print(f"{'Event':<22} {'GT':<8} {'IoU':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} "
          f"{'IoUmaxTV':>8} {'tTV':>6} {'IoUmaxST':>8} {'tST':>6} {'dth':>6}  "
          f"{'paperIoU':>8} {'paperMx':>8} {'paperdth':>8}")
    for r in rows:
        print(
            f"{r['event']:<22} {r['gt']:<8} "
            f"{fmt(r['IoU']):>6} {fmt(r['F1']):>6} {fmt(r['Precision']):>6} {fmt(r['Recall']):>6} "
            f"{fmt(r['IoUmax_tv']):>8} {fmt(r['t_best_tv'],1):>6} "
            f"{fmt(r['IoUmax_static']):>8} {fmt(r['t_best_static'],1):>6} "
            f"{fmt(r['dtheta'],1):>6}  "
            f"{fmt(r['paper_IoU']):>8} {fmt(r['paper_IoUmax']):>8} {fmt(r['paper_dtheta'],1):>8}"
        )

    print("\n--- delta vs paper (new - old) ---")
    for r in rows:
        diou = r["IoU"] - r["paper_IoU"]
        ddth = r["dtheta"] - r["paper_dtheta"]
        flag = ""
        if abs(diou) >= 0.015 or abs(ddth) >= 1.0:
            flag = "  <-- CHANGED"
        print(
            f"{r['event']:<22} {r['gt']:<8}  dIoU={diou:+.3f}  dF1={r['F1']-r['paper_F1']:+.3f}  "
            f"dθ={ddth:+.1f}  last40={r['IoU_last40']:.3f}  raw_tend={r['IoU_raw_tend']:.3f}{flag}"
        )


if __name__ == "__main__":
    main()
