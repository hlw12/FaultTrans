#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/PGA_real_eval
export PATH=/root/miniconda3/bin:$PATH
python -u infer_pga.py
python -u compare_pga.py
echo "PGA real-event eval finished."
