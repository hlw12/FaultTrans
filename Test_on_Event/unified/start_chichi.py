#!/usr/bin/env python
"""Detach Chi-Chi GMxy infer. Does not overwrite live notebook preds."""
import subprocess
from pathlib import Path

here = Path("/root/autodl-tmp/Test_on_Event/unified")
log = here / "outputs" / "infer_chichi.log"
log.parent.mkdir(parents=True, exist_ok=True)
fh = open(log, "w", encoding="utf-8")
proc = subprocess.Popen(
    ["/root/miniconda3/bin/python", "-u", str(here / "infer_chichi.py")],
    cwd=str(here),
    stdout=fh,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print("PID", proc.pid)
print("LOG", log)
