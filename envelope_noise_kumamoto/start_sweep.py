#!/usr/bin/env python
"""Start run_sweep.py in a new session so SSH can disconnect."""
import subprocess
from pathlib import Path

here = Path("/root/autodl-tmp/envelope_noise_kumamoto")
log = here / "outputs" / "sweep.log"
log.parent.mkdir(parents=True, exist_ok=True)
fh = open(log, "w", encoding="utf-8")
proc = subprocess.Popen(
    ["/root/miniconda3/bin/python", "-u", str(here / "run_sweep.py")],
    cwd=str(here),
    stdout=fh,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print("PID", proc.pid)
print("LOG", log)
