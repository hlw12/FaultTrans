# Real-event evaluation — data generation

Training (synthetic `events.h5`, checkpoints) is frozen. Do not re-run `main.py` for the paper tables.

## Canonical scripts (server)

Root: `/root/autodl-tmp/Test_on_Event/unified/`

| Product | Entry | Notes |
|---|---|---|
| FaultTrans PGV, max(\|EW\|,\|NS\|) | `infer.py` | Japan + Ridgecrest + Menyuan. Default writes `outputs/infer_max/`. Pass `--write-live` to overwrite `Test_on_Event` `pred_*.npy`. Refuses Chi-Chi. |
| FaultTrans PGV, Chi-Chi GMxy | `infer_chichi.py` | Independent running-peaks, then sqrt(IM_E * IM_N). Default writes `outputs/infer_chichi/`. Pass `--write-live` to overwrite live Chi-Chi preds. |
| FinDer Table 2 PGA + line masks | `finder_native.py` | **Two modes.** Japan: Gaussian, ellipse Δθ (13.5 / 12.0 / 35.7). USA/Ridgecrest: Delaunay, geographic-strike Δθ (15.5 / 13.9 / 9.7). Do not use `infer.py --with-finder`. |
| FaultTrans PGA control (fig06) | `pga_ft.py` | Writes `PGA_real_eval/outputs/` only. |
| Kumamoto envelope-noise sweep | `/root/autodl-tmp/envelope_noise_kumamoto/run_sweep.py` | Table 1 PGV (zeros dropped) + Table 2 FinDer (zeros kept). σ=0 must reprint FinDer 12.0°. |
| Restore Chi-Chi / Menyuan notebook preds | `restore_special.sh` | Copies `_backup_pre_unified/{Chichi,Menyuan}`. |
| Manuscript figures | `/root/autodl-tmp/nature_figures/run_all.py` | Reads live event preds + GT. |
| FinDer static maps | `/root/autodl-tmp/CmpFinDer/regen_finder_static_panel.py` | Kumamoto / Tottori panels. |

Algorithm (not an entry): `/root/autodl-tmp/CmpFinDer/finder.py`.

## Event data (do not delete)

- `/root/autodl-tmp/Test_on_Event/{kumamoto,Miyagi,Tottori,ridgecrest,chichi,menyuan}` — H5, GT, `event_metadata.npz`, `times_mid_valid.npy`, live `pred_*.npy`
- `/root/autodl-tmp/Test_on_Event/_backup_pre_unified/{Chichi,Menyuan}` — notebook FaultTrans artifacts
- Live `pred_*.npy` stay on the current artifacts until someone passes `--write-live`.
  Chi-Chi protocol is GMxy (`infer_chichi.py`); Menyuan protocol is max (`infer.py`).
  Notebook copies remain in `_backup_pre_unified/` and `original_notebooks/`.

## Original notebooks (do not delete, do not re-run onto live preds)

Paper Table 1 Chi-Chi / Menyuan were produced by these notebooks. Restored from
`D:\Mycode\FaultsMethod\autodl-tmp\` after the eval-box cleanup.

| Event | Notebook | Server path |
|---|---|---|
| Chi-Chi inference | `chichi_real_event.ipynb` | `/root/autodl-tmp/Test_on_Event/chichi/` |
| Chi-Chi compare | `compare_chichi.ipynb` | same |
| Chi-Chi picks | `phase_pick.ipynb` | same |
| Menyuan inference | `menyuan_inference.ipynb` | `/root/autodl-tmp/Test_on_Event/menyuan/` |
| Menyuan compare | `menyuan_compare.ipynb` | same |
| Menyuan debug | `menyuan_debug.ipynb` | same |
| Menyuan record section | `menyuan_record_section.ipynb` | same |

Second copy: `/root/autodl-tmp/Test_on_Event/_backup_pre_unified/{Chichi,Menyuan}/notebooks/`  
Local copy: `D:\article\Springer_Nature_Faults\figure_campaign\original_notebooks\`

## H5 conversion (local, not on the eval box)

- Chi-Chi: `D:\Mycode\Faults\utils\TW\CHNTWToHDF5.py` + `TWRecorder.py`
- Japan KiK-net/K-NET: `D:\Mycode\Faults\utils\KikNetToHDF5.py` (and `Recorder.py`)
- Ridgecrest CESMD: `D:\Mycode\Faults\utils\CESMDToHDF5.py`

## Training (frozen)

- `/root/autodl-tmp/main.py`, `fault_dataset.py`, `model.py`
- `/root/autodl-tmp/data/data/events.h5`
- PGV ckpt: `/root/autodl-tmp/checkpoints_pgv_aniso/pgv/best_iou_rule_hybrid.pth`
- PGA ckpt: `/root/autodl-tmp/checkpoints_pga/pga/best_iou_rule_hybrid.pth`

## Local copy of this folder

`D:\article\Springer_Nature_Faults\figure_campaign\eval_unified\`
