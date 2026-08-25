# CmpFinDer

A paper-level independent reimplementation of FinDer v2, used for
reproducible comparison against the current fault-rupture identification
model. This is not the unpublished official ETH/SED FinDer source code.

## Implemented

- Cua–Heaton (2009) rock-site PGA relations;
- Wells–Coppersmith (1994) rupture length–magnitude relations;
- the nine PGA binary thresholds from the paper;
- generation of binary finite line-source templates;
- translation matching via FFT correlation;
- hierarchical strike search and adjacent-length search;
- time-series processing in which threshold, magnitude, and rupture
  length are constrained to be non-decreasing;
- Delaunay interpolation of station PGA with a zero-valued boundary;
- a unified evaluation script on the project `events.h5` test set.

## Files

- `finder.py`: FinDer core algorithm.
- `benchmark_h5.py`: compute IoU, F1, precision, recall, magnitude
  error, strike error, and runtime on an existing HDF5 dataset.
- `test_finder.py`: unit tests for the formulae and synthetic templates.
- `requirements.txt`: minimal dependencies.

## Usage

```bash
cd /root/autodl-tmp/CmpFinDer
python3 -m unittest -v
python3 benchmark_h5.py --h5 ../data/data/events.h5 --max-events 10
```

Evaluation results are written to:

- `results/frame_results.csv`
- `results/summary.json`

Full test set:

```bash
python3 benchmark_h5.py --h5 ../data/data/events.h5 --max-events 0
```

## Inputs and units

The core interface `FinDer.fit(pga_image)` takes a 2-D cumulative PGA
field. The default is `pga_unit=auto`: values whose maximum is at most 5
are treated as `g`, otherwise as `cm/s²`. For production experiments,
set `--pga-unit g` or `--pga-unit cm_s2` according to the data-generation
pipeline.

The current data configuration uses `resolution_deg=0.01`. The evaluation
script reads this from HDF5 root attributes and converts it to about
1.11 km/pixel. The data are already a spatial PGA grid, so station
interpolation is not repeated.

## References

- Böse, M., Heaton, T. H., & Hauksson, E. (2012), *Real-time Finite
  Fault Rupture Detector (FinDer) for large earthquakes*, GJI 191.
- Böse, M. et al. (2018), *FinDer v.2: Improved real-time ground-motion
  predictions for M2–M9*, GJI 212.
