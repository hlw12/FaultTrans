#!/usr/bin/env python
import subprocess
from pathlib import Path

here = Path("/root/autodl-tmp/envelope_noise_kumamoto")
log = here / "outputs" / "plot_from_masks.log"
fh = open(log, "w", encoding="utf-8")
proc = subprocess.Popen(
    ["/root/miniconda3/bin/python", "-u", str(here / "plot_from_masks.py")],
    cwd=str(here),
    stdout=fh,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print("PID", proc.pid)
print("LOG", log)
