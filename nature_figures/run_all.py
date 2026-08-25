#!/usr/bin/env python
"""Build GT cache and render all campaign figures."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cache_gt import build_all
from figures import ALL
from style import CACHE, EXPORT, SOURCE, ensure_dirs


def main():
    ensure_dirs()
    legacy = EXPORT / "_combined_legacy"
    legacy.mkdir(parents=True, exist_ok=True)
    combined_stems = {
        "fig01_six_event_gallery",
        "fig02_kumamoto_realtime",
        "fig03_ridgecrest_models",
        "fig04_ridgecrest_timing",
        "fig05_chichi",
        "fig06_pga_vs_pgv",
        "fig07_failures",
        "fig08_finder",
        "fig09_synthetic",
        "fig10_gt_pipeline",
    }
    for p in EXPORT.iterdir():
        if p.is_file() and p.stem in combined_stems:
            p.replace(legacy / p.name)
            print("moved combined", p.name, "-> _combined_legacy")
    print("=== cache GT ===")
    if not (CACHE / "kumamoto_cv.npy").exists() or not (CACHE / "kumamoto_cv_slip.npy").exists():
        build_all()
    else:
        print("cache exists, skip")
    print("=== figures ===")
    failed = []
    for fn in ALL:
        print(f"-> {fn.__name__}")
        try:
            fn()
        except Exception:
            failed.append(fn.__name__)
            traceback.print_exc()
    print("png panels:", sorted(p.name for p in EXPORT.glob("*.png") if p.parent == EXPORT))
    print("source :", sorted(p.name for p in SOURCE.glob("*")))
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
    print("ALL OK")


if __name__ == "__main__":
    main()
