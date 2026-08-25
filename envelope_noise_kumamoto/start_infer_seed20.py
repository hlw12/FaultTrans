#!/usr/bin/env python
import os
import subprocess
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
here = Path("/root/autodl-tmp/envelope_noise_kumamoto")
log = here / "outputs" / "infer_seed20.log"
fh = open(log, "w", encoding="utf-8")
proc = subprocess.Popen(
    ["/root/miniconda3/bin/python", "-u", str(here / "infer_seed20.py")],
    cwd=str(here),
    stdout=fh,
    stderr=subprocess.STDOUT,
    start_new_session=True,
    env=os.environ.copy(),
)
print("PID", proc.pid)
print("LOG", log)
