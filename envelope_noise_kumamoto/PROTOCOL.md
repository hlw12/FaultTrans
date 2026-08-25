# Kumamoto envelope-noise control (FaultTrans vs FinDer)

Isolated workspace on the eval box:

`/root/autodl-tmp/envelope_noise_kumamoto/`

Does **not** write into `Test_on_Event/` live preds, `CmpFinDer/`, `PGA_real_eval/`, or the manuscript.

The unified `Test_on_Event/unified/noise_sweep.py` recipe is **deleted**. It interpolated FinDer PGA with the FaultTrans interpolator (zeros dropped) and gated FinDer at ~29°, which is not Table 2.

## Question

Fix Kumamoto geometry and PhaseNet P arrivals. Apply a **station-wise, time-constant lognormal gain** to each station, then ask how FaultTrans (ΔPGV segmentation) and FinDer (PGA line-source matching) change.

This is **not** waveform denoising, baseline-drift simulation, or pick jitter.

## Perturbation

For each station location \(k\):

\[
\varepsilon_k \sim \mathcal{N}(0,\sigma^2),\qquad
\mathrm{PGV}_k(t)\leftarrow \mathrm{PGV}_k(t)\,e^{\varepsilon_k},\qquad
\mathrm{PGA}_k(t)\leftarrow \mathrm{PGA}_k(t)\,e^{\varepsilon_k}.
\]

Same \(\varepsilon_k\) on both intensity measures. \(\sigma=0\) is the clean gate.

## Native fields (do not mix)

| Method | Envelope | Interpolator | \(\Delta\theta\) |
|---|---|---|---|
| FaultTrans | unified PGV, `max(\|EW\|,\|NS\|)` | Gaussian \(\sigma=0.15^\circ\), **zeros dropped** | ellipse vs Curved GT |
| FinDer | `finder_native` Japan PGA, max-\(\lvert a\rvert\) including UD | Gaussian \(\sigma=0.15^\circ\), **zeros kept** | ellipse of painted line vs Curved GT |

Last-frame clock: `times_mid_valid.npy`, idx 30, \(t\le t_\mathrm{end}^\mathrm{inf}\).

## Frozen

- Station set and coordinates inside the \(1.5^\circ\) grid
- PhaseNet CSV; no re-picking
- FaultTrans checkpoint `checkpoints_pgv_aniso/pgv/best_iou_rule_hybrid.pth`
- FinDer: `infer.run_finder` / `finder_native.py` Japan mode
- Training data and checkpoints

## Gate before the sweep

| Method | Quantity | Paper | Must reprint at \(\sigma=0\) |
|---|---|---|---|
| FaultTrans | IoU vs Curved | 0.564 | \(0.564\pm 0.02\) |
| FaultTrans | \(\Delta\theta_\mathrm{final}\) | 11.3° | \(11.3\pm 1.5^\circ\) |
| FinDer | \(\Delta\theta_\mathrm{final}\) | 12.0° | \(12.0\pm 1.0^\circ\) |

FinDer 12.0° is a hard stop. Do not substitute Delaunay, a 60-frame origin grid, geographic-strike \(\Delta\theta\), or the deleted unified PGA interpolator.

## Sweep result

Finished on the Table 2 FinDer field. Same \(\varepsilon_k\) on PGV and PGA envelopes. 10 seeds at each \(\sigma>0\). FinDer \(\sigma=0\) is the paper 12.0°.

| \(\sigma\) | n | FaultTrans IoU | FaultTrans \(\Delta\theta\) | FinDer \(\Delta\theta\) |
|---|---|---|---|---|
| 0.0 | 1 | 0.564 | 11.3° | **12.0°** |
| 0.2 | 10 | 0.567 ± 0.029 | 11.2 ± 1.6° | 17.9 ± 10.9° |
| 0.4 | 10 | 0.570 ± 0.050 | 11.2 ± 2.3° | 21.7 ± 12.0° |

## What this does not do

- No training-time noise
- No grid/pixel iid noise
- No pick perturbation
- No LaTeX edits
- No other events
