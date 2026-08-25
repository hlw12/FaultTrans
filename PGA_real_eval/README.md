# PGA real-event evaluation

Independent copy of the PGV real-event pipeline (`Test_on_Event/real_event.ipynb` + `compare.ipynb`).

**Does not modify or delete any existing files under `Test_on_Event/`.**

## What changed vs PGV

1. Station field: 2 s running peak of band-passed `|acceleration|` (gal), not `|velocity|` (cm/s).
2. Normalization: `dpga_mean` / `dpga_std` from `events.h5`.
3. Checkpoint: `/root/autodl-tmp/checkpoints_pga/pga/best_iou_rule_hybrid.pth`.

Ridgecrest CESMD HDF5 stores velocity (`H1`/`H2`); PGA is obtained by differentiating that velocity, then applying the same 0.1–10 Hz filter.

## Run

```bash
cd /root/autodl-tmp/PGA_real_eval
python infer_pga.py
python compare_pga.py
```

Outputs: `outputs/<event>/`  Metrics: `metrics/*.csv`
