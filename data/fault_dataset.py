#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/3/17 01:58
# @Author  : 上头欢乐送、
# @File    : fault_dataset.py
# @Software: PyCharm
# 学习新思想，争做新青年

"""
fault_dataset.py

Each sample returns:
    inputs       : Tensor [3, T, H, W]   PGA / PGV / Intensity
    rup_grid     : Tensor [T, H, W]      label = rupture_grid(t - Δt)
    magnitude    : Tensor [T]
    times        : Tensor [T]
    epicenter    : Tensor [2]
    rup_progress : Tensor [T]            0→1, fraction of rupture complete
    rup_complete : Tensor [T]            0/1, whether rupture has finished
    valid_frame  : Tensor [T]            1 = unique label, 0 = ambiguous
    meta         : dict

Frame validity
--------------
valid_frame[k] = 1  if  t_k ≤ t_complete + Δt
               = 0  otherwise

After t_complete + Δt:
  - rupture_grid is frozen (final state)
  - field is still changing (far-field S-waves arriving)
  → multiple different inputs map to same label → ambiguous → weight = 0

Training loss weighting
-----------------------
Use compute_frame_weights() to get per-frame loss weights:

    weights = compute_frame_weights(batch['valid_frame'])
    loss    = (criterion(pred, batch['rup_grid']) * weights).mean()
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List


def compute_frame_weights(
    valid_frame:  torch.Tensor,
    w_valid:      float = 1.0,
    w_invalid:    float = 0.0,
) -> torch.Tensor:
    """
    Per-frame loss weights.

    Parameters
    ----------
    valid_frame : Tensor [..., T]  values 0.0 or 1.0
    w_valid     : weight for frames where label is unambiguous (default 1.0)
    w_invalid   : weight for ambiguous post-rupture frames (default 0.0)
                  Set > 0 if you want to partially supervise those frames.

    Returns
    -------
    weights : Tensor [..., T]
    """
    return torch.where(
        valid_frame > 0.5,
        torch.full_like(valid_frame, w_valid),
        torch.full_like(valid_frame, w_invalid),
    )


class EarthquakeFaultDataset(Dataset):
    """
    Lazy-loading HDF5 dataset.

    Parameters
    ----------
    h5_path   : path to events.h5
    event_ids : optional list of event group names (for split)
    normalize : normalise PGA/PGV/Intensity channels
    """

    CH_PGA       = 0
    CH_PGV       = 1
    CH_INTENSITY = 2

    def __init__(
        self,
        h5_path:   str,
        event_ids: Optional[List[str]] = None,
        normalize: bool = True,
    ):
        self.h5_path   = str(h5_path)
        self.normalize = normalize
        self._h5       = None

        with h5py.File(self.h5_path, 'r') as f:
            all_ids = sorted(k for k in f.keys() if k.startswith('event_'))
        self.event_ids = event_ids if event_ids is not None else all_ids
        self._stats: Optional[dict] = None
        if normalize:
            self._stats = self._compute_or_load_stats()

    def _compute_or_load_stats(self) -> dict:
        with h5py.File(self.h5_path, 'r+') as f:
            if 'normalization' in f:
                ng = f['normalization']
                return {
                    'pga': (float(ng.attrs['pga_mean']), float(ng.attrs['pga_std'])),
                    'pgv': (float(ng.attrs['pgv_mean']), float(ng.attrs['pgv_std'])),
                    'int': (float(ng.attrs['int_mean']), float(ng.attrs['int_std'])),
                }
            pga_vals, pgv_vals, int_vals = [], [], []
            sample_ids = self.event_ids[:min(200, len(self.event_ids))]
            for eid in sample_ids:
                grp = f[eid]['inputs']
                pga_vals.append(grp['pga_field'][:].ravel())
                pgv_vals.append(grp['pgv_field'][:].ravel())
                int_vals.append(grp['intensity_field'][:].ravel())

            def ms(arr_list):
                a = np.concatenate(arr_list)
                a = a[a > 0]
                return (float(a.mean()) if len(a) else 1.0,
                        float(a.std())  if len(a) else 1.0)

            pm, ps = ms(pga_vals)
            vm, vs = ms(pgv_vals)
            im, is_ = ms(int_vals)
            ng = f.require_group('normalization')
            ng.attrs['pga_mean'] = pm; ng.attrs['pga_std'] = ps
            ng.attrs['pgv_mean'] = vm; ng.attrs['pgv_std'] = vs
            ng.attrs['int_mean'] = im; ng.attrs['int_std'] = is_
        return {'pga': (pm, ps), 'pgv': (vm, vs), 'int': (im, is_)}

    def _norm(self, x: np.ndarray, key: str) -> np.ndarray:
        m, s = self._stats[key]
        return (x - m) / (s + 1e-8)

    @property
    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def __del__(self):
        if self._h5 is not None:
            try: self._h5.close()
            except Exception: pass

    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, idx: int):
        eid = self.event_ids[idx]
        grp = self._file[eid]
        inp = grp['inputs']
        lbl = grp['labels']

        pga   = inp['pga_field'][:]
        pgv   = inp['pgv_field'][:]
        ity   = inp['intensity_field'][:]
        times = inp['times'][:]

        rup = lbl['rupture_grid'][:]
        mag = lbl['magnitude'][:]

        # Rupture state
        rup_prog = lbl['rupture_progress'][:] if 'rupture_progress' in lbl \
                   else np.ones(rup.shape[0], dtype=np.float32)
        rup_comp = lbl['rupture_complete'][:] if 'rupture_complete' in lbl \
                   else np.zeros(rup.shape[0], dtype=np.float32)

        # valid_frame: 1 = label is unambiguous, 0 = exclude from training
        if 'valid_frames' in lbl:
            valid = lbl['valid_frames'][:]
        else:
            # Fallback for older files: all frames valid
            valid = np.ones(rup.shape[0], dtype=np.float32)

        epi  = grp['epicenter'][:]

        if self.normalize and self._stats:
            pga = self._norm(pga, 'pga')
            pgv = self._norm(pgv, 'pgv')
            ity = self._norm(ity, 'int')

        fields = np.stack([pga, pgv, ity], axis=0).astype(np.float32)
        meta   = dict(grp.attrs)
        meta['event_id'] = eid

        return {
            'inputs':       torch.from_numpy(fields),
            'rup_grid':     torch.from_numpy(rup.astype(np.float32)),
            'magnitude':    torch.from_numpy(mag.astype(np.float32)),
            'times':        torch.from_numpy(times.astype(np.float32)),
            'epicenter':    torch.from_numpy(epi.astype(np.float32)),
            'rup_progress': torch.from_numpy(rup_prog.astype(np.float32)),
            'rup_complete': torch.from_numpy(rup_comp.astype(np.float32)),
            'valid_frame':  torch.from_numpy(valid.astype(np.float32)),
            'meta':         meta,
        }

    @classmethod
    def train_val_test_split(
        cls,
        h5_path:   str,
        ratios:    Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed:      int  = 42,
        normalize: bool = True,
    ) -> Tuple['EarthquakeFaultDataset',
               'EarthquakeFaultDataset',
               'EarthquakeFaultDataset']:
        with h5py.File(h5_path, 'r') as f:
            all_ids = sorted(k for k in f.keys() if k.startswith('event_'))
        rng = np.random.default_rng(seed)
        ids = rng.permutation(all_ids).tolist()
        n   = len(ids)
        n1  = int(n * ratios[0])
        n2  = int(n * ratios[1])
        train_ds = cls(h5_path, event_ids=ids[:n1],      normalize=normalize)
        val_ds   = cls(h5_path, event_ids=ids[n1:n1+n2], normalize=normalize)
        test_ds  = cls(h5_path, event_ids=ids[n1+n2:],   normalize=normalize)
        if normalize:
            val_ds._stats  = train_ds._stats
            test_ds._stats = train_ds._stats
        return train_ds, val_ds, test_ds


def collate_fn(batch):
    return torch.utils.data.dataloader.default_collate(
        [{k: v for k, v in b.items() if k != 'meta'} for b in batch])


if __name__ == '__main__':
    import sys
    h5_path = sys.argv[1] if len(sys.argv) > 1 else 'data/events.h5'

    train_ds, val_ds, test_ds = EarthquakeFaultDataset.train_val_test_split(h5_path)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                        num_workers=0, collate_fn=collate_fn)
    batch  = next(iter(loader))
    print(f"inputs      : {batch['inputs'].shape}")       # [4, 3, T, H, W]
    print(f"rup_grid    : {batch['rup_grid'].shape}")     # [4, T, H, W]
    print(f"valid_frame : {batch['valid_frame'].shape}")  # [4, T]
    print(f"valid ratio : {batch['valid_frame'].mean():.2f}")

    # Example weighted loss
    weights = compute_frame_weights(batch['valid_frame'])
    print(f"weight sample: {weights[0]}")   # [T]: 1.0 for valid, 0.0 for invalid