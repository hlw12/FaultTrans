import os
import subprocess
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

here = Path("/root/autodl-tmp/envelope_noise_kumamoto")
log = here / "outputs" / "plot_diagnostic.log"
log.parent.mkdir(parents=True, exist_ok=True)
fh = open(log, "w", encoding="utf-8")
proc = subprocess.Popen(
    ["/root/miniconda3/bin/python", "-u", str(here / "plot_diagnostic.py")],
    cwd=str(here),
    stdout=fh,
    stderr=subprocess.STDOUT,
    start_new_session=True,
    env=os.environ.copy(),
)
print("PID", proc.pid)
print("LOG", log)
