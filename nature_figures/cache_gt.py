"""Cache binary GT masks for all paper events."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from catalog import EVENTS
from geo import parse_nied, parse_pku, parse_srcmod_mat, rasterize_subfaults
from style import CACHE, ensure_dirs


def _save(name, mask, slip=None):
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / f"{name}.npy", mask.astype(np.uint8))
    if slip is not None:
        np.save(CACHE / f"{name}_slip.npy", slip.astype(np.float32))
    print(f"  cached {name}  n={int(mask.sum())}")
    return mask


def load_cached(name):
    p = CACHE / f"{name}.npy"
    return np.load(p) if p.exists() else None


def build_all():
    ensure_dirs()
    out = {}

    # Kumamoto
    ev = EVENTS["kumamoto"]
    meta = np.load(ev["pgv_dir"] / "event_metadata.npz")
    lat, lon = float(meta["eq_lat"]), float(meta["eq_lon"])
    df = parse_nied(ev["nied_rp"])
    mask, slip = rasterize_subfaults(df, lat, lon)
    out["kumamoto_rp"] = _save("kumamoto_rp", mask, slip)
    df = parse_nied(ev["nied_cv"])
    mask, slip = rasterize_subfaults(df, lat, lon)
    out["kumamoto_cv"] = _save("kumamoto_cv", mask, slip)

    # Miyagi
    ev = EVENTS["Miyagi"]
    meta = np.load(ev["pgv_dir"] / "event_metadata.npz")
    lat, lon = float(meta["eq_lat"]), float(meta["eq_lon"])
    df = parse_nied(ev["nied_rp"])
    mask, slip = rasterize_subfaults(df, lat, lon)
    out["Miyagi_rp"] = _save("Miyagi_rp", mask, slip)

    # Tottori
    ev = EVENTS["Tottori"]
    meta = np.load(ev["pgv_dir"] / "event_metadata.npz")
    lat, lon = float(meta["eq_lat"]), float(meta["eq_lon"])
    df = parse_nied(ev["nied_rp"])
    mask, slip = rasterize_subfaults(df, lat, lon)
    out["Tottori_rp"] = _save("Tottori_rp", mask, slip)

    # Ridgecrest SRCMOD + PKU
    ev = EVENTS["ridgecrest"]
    meta = np.load(ev["pgv_dir"] / "event_metadata.npz")
    lat, lon = float(meta["eq_lat"]), float(meta["eq_lon"])
    for key, path in ev["srcmod"].items():
        df = parse_srcmod_mat(path)
        mask, slip = rasterize_subfaults(df, lat, lon)
        out[f"ridgecrest_{key}"] = _save(f"ridgecrest_{key}", mask, slip)
    df = parse_pku(ev["pku_txt"])
    mask, slip = rasterize_subfaults(df, lat, lon)
    out["ridgecrest_pku"] = _save("ridgecrest_pku", mask, slip)
    df.to_pickle(CACHE / "ridgecrest_pku_df.pkl")

    # Chi-Chi
    ev = EVENTS["chichi"]
    meta = np.load(ev["pgv_dir"] / "event_metadata.npz")
    lat, lon = float(meta["eq_lat"]), float(meta["eq_lon"])
    for key, path in ev["srcmod"].items():
        df = parse_srcmod_mat(path)
        mask, slip = rasterize_subfaults(df, lat, lon)
        out[f"chichi_{key}"] = _save(f"chichi_{key}", mask, slip)

    # Menyuan PKU
    ev = EVENTS["Menyuan"]
    meta = np.load(ev["pgv_dir"] / "event_metadata.npz")
    lat, lon = float(meta["eq_lat"]), float(meta["eq_lon"])
    df = parse_pku(ev["pku_txt"])
    mask, slip = rasterize_subfaults(df, lat, lon)
    out["Menyuan_pku"] = _save("Menyuan_pku", mask, slip)
    df.to_pickle(CACHE / "Menyuan_pku_df.pkl")
    return out


PREFERRED = {
    "kumamoto": "kumamoto_cv",
    "Miyagi": "Miyagi_rp",
    "Tottori": "Tottori_rp",
    "ridgecrest": "ridgecrest_XU",
    "chichi": "chichi_CHI",
    "Menyuan": "Menyuan_pku",
}


def preferred_mask(event_key):
    name = PREFERRED[event_key]
    p = CACHE / f"{name}.npy"
    if not p.exists():
        build_all()
    return np.load(p), name


if __name__ == "__main__":
    build_all()
