# Eval layout (post-reorg)

Training code, `data/`, `checkpoints_*`, and `logs/` were not modified.

## Live eval
- `Test_on_Event/unified/`  shared PGV/PGA/FinDer preprocessor and runners
- `Test_on_Event/{kumamoto,Miyagi,Tottori,ridgecrest,chichi,menyuan}`  H5, GT, metadata, predictions
- `Test_on_Event/_backup_pre_unified/{Chichi,Menyuan}`  original notebook artifacts (do not overwrite)
- `nature_figures/`  manuscript figures
- `CmpFinDer/finder.py`  FinDer implementation
- `PGA_real_eval/outputs/`  unified PGA FaultTrans (Japan + Ridgecrest); Chi-Chi/Menyuan remain notebook PGA

Do not rebuild fields from archived notebooks. They were deleted.
