#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Finish unified eval on the copy box: restore, PGA FT, figures, FinDer maps, noise, archive."""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/autodl-tmp")
HERE = ROOT / "Test_on_Event" / "unified"
PY = "/root/miniconda3/bin/python"
LOG = HERE / "outputs" / "complete.log"


def run(name, argv, cwd=None):
    print(f"\n===== {name}  {datetime.now().isoformat()} =====", flush=True)
    print(" ".join(argv), flush=True)
    r = subprocess.run(argv, cwd=str(cwd) if cwd else None)
    if r.returncode != 0:
        raise SystemExit(f"FAILED {name} exit={r.returncode}")
    print(f"===== OK {name} =====", flush=True)


def main():
    HERE.joinpath("outputs").mkdir(parents=True, exist_ok=True)
    print("[complete_pipeline]", datetime.now().isoformat(), flush=True)

    restore = HERE / "restore_special.sh"
    if restore.exists():
        run("restore", ["bash", str(restore)])

    run("dump_status", [PY, "-u", str(HERE / "dump_status.py")], cwd=HERE)
    run("pga_ft", [PY, "-u", str(HERE / "pga_ft.py")], cwd=HERE)
    run("rerun_table", [PY, "-u", "rerun_table.py"], cwd=ROOT / "nature_figures")
    run("figures", [PY, "-u", "run_all.py"], cwd=ROOT / "nature_figures")
    run("finder_static", [PY, "-u", str(ROOT / "CmpFinDer" / "regen_finder_static_panel.py")])
    run("noise", [PY, "-u", str(ROOT / "envelope_noise_kumamoto" / "run_sweep.py")])
    run("archive", [PY, "-u", str(HERE / "archive_divergent.py")], cwd=HERE)
    run("dump_status_final", [PY, "-u", str(HERE / "dump_status.py")], cwd=HERE)
    print("[ALL DONE]", datetime.now().isoformat(), flush=True)


if __name__ == "__main__":
    main()
