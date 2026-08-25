#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified real-event inference: FaultTrans PGV with max(|EW|,|NS|).

Default events: Japan + Ridgecrest + Menyuan. Chi-Chi is infer_chichi.py (GMxy).
Does not write live Test_on_Event pred_*.npy unless --write-live.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path("/root/autodl-tmp")
HERE = Path("/root/autodl-tmp/Test_on_Event/unified")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "CmpFinDer"))
sys.path.insert(0, str(ROOT / "nature_figures"))
sys.path.insert(0, str(HERE))

from fields import (  # noqa: E402
    EVENTS,
    MAX_EVENTS,
    CHICHI_EVENTS,
    COMBINE_BY_EVENT,
    envelopes,
    interpolate_env,
    pixel_size_km,
    strike_delta,
    build_station_traces,
)
from model import FaultCleanRule  # noqa: E402
from finder import FinDer, FinDerConfig, finder_line_mask  # noqa: E402
from geo import compute_metrics  # noqa: E402
from catalog import EVENTS as CAT  # noqa: E402
from geo import parse_nied, parse_pku, parse_srcmod_mat, rasterize_subfaults  # noqa: E402

CKPT_PGV = ROOT / "checkpoints_pgv_aniso" / "pgv" / "best_iou_rule_hybrid.pth"
CKPT_PGA = ROOT / "checkpoints_pga" / "pga" / "best_iou_rule_hybrid.pth"
TRAIN_H5 = ROOT / "data" / "data" / "events.h5"
BACKUP = ROOT / "Test_on_Event" / "_backup_pre_unified"
OUT = HERE / "outputs"
THRESHOLD = 0.5
FINDER_LINE_HALF_WIDTH_PX = 1.5
GRID_SIZE = 150

SAVE_KEYS = [
    "pred_binary.npy",
    "pred_prob.npy",
    "pgv_fields.npy",
    "pga_fields.npy",
    "finder_pred_line_mask.npy",
    "finder_times_mid.npy",
]


def load_norm(channel="pgv"):
    with h5py.File(TRAIN_H5, "r") as f:
        ng = f["normalization_diff"]
        if channel == "pgv":
            return float(ng.attrs["dpgv_mean"]), float(ng.attrs["dpgv_std"])
        return float(ng.attrs["dpga_mean"]), float(ng.attrs["dpga_std"])


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    def g(name, default):
        return args.get(name, default) if isinstance(args, dict) else default

    model = FaultCleanRule(
        in_channels=1,
        patch_size=int(g("patch_size", 10)),
        d_model=int(g("d_model", 256)),
        n_layers=int(g("n_layers", 6)),
        grid_size=int(g("grid_size", 150)),
        decoder_dim=int(g("decoder_dim", 32)),
        decoder_mid_dim=int(g("decoder_mid_dim", 64)),
        decoder_refine_blocks=int(g("decoder_refine_blocks", 3)),
        causal=bool(g("causal", True)),
        commit_tau=float(g("commit_tau", 0.60)),
        commit_temp=float(g("commit_temp", 0.08)),
    ).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if len(state) > 0 and all(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded {ckpt_path} epoch={ckpt.get('epoch')} val_iou={ckpt.get('val_iou')}")
    return model


def run_ft(fields, times, packed, model, device, mean, std):
    d = np.maximum(np.diff(fields, axis=0, prepend=fields[:1]), 0.0)
    d_norm = (d - mean) / (std + 1e-8)
    inputs_t = torch.from_numpy(d_norm[np.newaxis, np.newaxis]).float().to(device)
    times_t = torch.from_numpy(times[np.newaxis]).float().to(device)
    seq_mask = torch.ones(1, len(times), dtype=torch.bool, device=device)
    lon_min, lon_max = packed["lon_min"], packed["lon_max"]
    lat_min, lat_max = packed["lat_min"], packed["lat_max"]
    epi_x = (packed["eq_lon"] - lon_min) / (lon_max - lon_min)
    epi_y = (packed["eq_lat"] - lat_min) / (lat_max - lat_min)
    epicenter_t = torch.tensor([[epi_x, epi_y]], dtype=torch.float32, device=device)
    with torch.no_grad():
        _, committeds = model(inputs_t, times_t, seq_mask, epicenter_t)
    committed = committeds[0].cpu().numpy()
    binary = (committed > THRESHOLD).astype(np.uint8)
    return committed, binary


def run_finder(pga_fields, packed):
    pixel_km = pixel_size_km(packed["eq_lat"])
    finder = FinDer(
        FinDerConfig(pixel_size_km=pixel_km, pga_unit="cm_s2", min_active_pixels=10, growth_only=True)
    )
    results = finder.fit_sequence(pga_fields)
    masks = []
    for res in results:
        masks.append(
            finder_line_mask(res, grid_size=GRID_SIZE, pixel_size_km=pixel_km, half_width_px=FINDER_LINE_HALF_WIDTH_PX)
        )
    return results, np.stack(masks, axis=0), pixel_km


def gt_masks(key, packed):
    cat = CAT[key]
    lat, lon = packed["eq_lat"], packed["eq_lon"]
    out = {}
    if "nied_rp" in cat and cat["nied_rp"] is not None and Path(cat["nied_rp"]).exists():
        df = parse_nied(cat["nied_rp"])
        out["RP"], _ = rasterize_subfaults(df, lat, lon)
    if cat.get("nied_cv") and Path(cat["nied_cv"]).exists():
        df = parse_nied(cat["nied_cv"])
        out["Curved"], _ = rasterize_subfaults(df, lat, lon)
    if cat.get("srcmod"):
        for name, path in cat["srcmod"].items():
            df = parse_srcmod_mat(path)
            out[name], _ = rasterize_subfaults(df, lat, lon)
    if key == "Menyuan" and cat.get("pku_txt") and Path(cat["pku_txt"]).exists():
        df = parse_pku(cat["pku_txt"])
        out["XU/PKU"], _ = rasterize_subfaults(df, lat, lon)
    return out


def backup_event(src: Path):
    dest = BACKUP / src.name
    dest.mkdir(parents=True, exist_ok=True)
    for name in SAVE_KEYS:
        p = src / name
        if p.exists() and not (dest / name).exists():
            shutil.copy2(p, dest / name)


def score_rows(key, binary, finder_masks, packed, gts):
    idx = packed["final_idx"]
    times = packed["times"]
    t = float(times[idx])
    rows = []
    pred = binary[idx]
    line = finder_masks[idx] if finder_masks is not None else None
    how = packed.get("combine", "")
    for gt_name, gt in gts.items():
        m = compute_metrics(pred, gt)
        ft_d = strike_delta(pred, gt)
        fd_d = strike_delta(line, gt) if line is not None else float("nan")
        ious = [
            compute_metrics(binary[i], gt)["IoU"] if i <= idx else -1.0
            for i in range(len(binary))
        ]
        kbest = int(np.argmax(ious))
        rows.append(
            dict(
                event=key,
                gt=gt_name,
                combine=how,
                final_idx=idx,
                t_eval=t,
                n_stations=len(packed["stations"]),
                n_two_horiz=int(packed.get("n_two_horiz", 0)),
                ft_iou=m["IoU"],
                ft_f1=m["F1"],
                ft_precision=m["Precision"],
                ft_recall=m["Recall"],
                ft_dtheta=ft_d,
                ft_ioumax=float(ious[kbest]),
                t_ioumax=float(times[kbest]),
                finder_dtheta=fd_d,
            )
        )
        print(
            f"  [{key}/{gt_name}] FT IoU={m['IoU']:.3f} Δθ={ft_d:.1f}  "
            f"IoUmax={ious[kbest]:.3f} @{times[kbest]:.1f}s  FinDer Δθ={fd_d:.1f}"
        )
    return rows


def run_event(key, model_pgv, device, dpgv_mean, dpgv_std, do_finder=False, write_live=False, combine=None):
    cfg = EVENTS[key]
    how = combine or COMBINE_BY_EVENT[key]
    packed = build_station_traces(cfg)
    packed["combine"] = how
    print(
        f"=== {key} combine={how} stations={len(packed['stations'])} frames={len(packed['times'])} "
        f"final_idx={packed['final_idx']} t={packed['times'][packed['final_idx']]:.3f} ==="
    )
    pga, pgv, _pga_rp = envelopes(packed, combine=how)
    pgv_fields = interpolate_env(pgv, packed)
    pga_fields = interpolate_env(pga, packed)
    committed, binary = run_ft(pgv_fields, packed["times"], packed, model_pgv, device, dpgv_mean, dpgv_std)
    finder_masks = None
    if do_finder and cfg.get("finder"):
        _results, finder_masks, _pk = run_finder(pga_fields, packed)
    gts = gt_masks(key, packed)
    subdir = "infer_chichi" if how == "geomean" else "infer_max"
    dest = cfg["src"] if write_live else (OUT / subdir / key)
    dest.mkdir(parents=True, exist_ok=True)
    if write_live:
        backup_event(cfg["src"])
    np.save(dest / "pgv_fields.npy", pgv_fields)
    np.save(dest / "pga_fields.npy", pga_fields)
    np.save(dest / "pred_prob.npy", committed.astype(np.float32))
    np.save(dest / "pred_binary.npy", binary)
    if finder_masks is not None:
        np.save(dest / "finder_pred_line_mask.npy", finder_masks)
        np.save(dest / "finder_times_mid.npy", packed["times"])
    print(f"  wrote {dest}")
    return score_rows(key, binary, finder_masks, packed, gts)


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", nargs="*", default=None)
    parser.add_argument(
        "--with-finder",
        action="store_true",
        help="Fit FinDer on the FaultTrans PGA field (do not use; native FinDer is finder_native.py).",
    )
    parser.add_argument("--skip-finder", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--write-live",
        action="store_true",
        help="Overwrite Test_on_Event live pred_*.npy (default writes under outputs/infer_max/).",
    )
    args = parser.parse_args()
    keys = list(args.events or MAX_EVENTS)
    bad = [k for k in keys if k in CHICHI_EVENTS]
    if bad:
        raise SystemExit(f"Chi-Chi belongs in infer_chichi.py, not infer.py: {bad}")
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dpgv_mean, dpgv_std = load_norm("pgv")
    model = load_model(CKPT_PGV, device)
    rows = []
    do_finder = bool(args.with_finder) and not args.skip_finder
    for key in keys:
        rows.extend(
            run_event(
                key,
                model,
                device,
                dpgv_mean,
                dpgv_std,
                do_finder=do_finder,
                write_live=args.write_live,
                combine="max",
            )
        )
        write_csv(OUT / "metrics_max.csv", rows)
        (OUT / "metrics_max.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("[done]", datetime.now().isoformat(), "n_rows", len(rows))
    for r in rows:
        print(
            f"{r['event']:12s} {r['gt']:8s}  IoU={r['ft_iou']:.3f}  "
            f"FTΔθ={r['ft_dtheta']:.1f}  FinDerΔθ={r['finder_dtheta']:.1f}"
        )


if __name__ == "__main__":
    main()
