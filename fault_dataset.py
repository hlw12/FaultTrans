#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple


GRID_SIZE = 150


def compute_frame_weights(
    valid_frame: torch.Tensor,
    w_valid: float = 1.0,
    w_invalid: float = 0.0,
) -> torch.Tensor:
    return torch.where(
        valid_frame > 0.5,
        torch.full_like(valid_frame, w_valid),
        torch.full_like(valid_frame, w_invalid),
    )


class EarthquakeFaultDataset(Dataset):
    CHANNELS = ('pga', 'pgv', 'int')
    _CH_FIELD = {'pga': 'pga_field', 'pgv': 'pgv_field', 'int': 'intensity_field'}
    _CH_STAT = {'pga': 'dpga', 'pgv': 'dpgv', 'int': 'dint'}

    def __init__(
        self,
        h5_path: str,
        event_ids: Optional[List[str]] = None,
        normalize: bool = True,
        only_valid: bool = True,
        channel: str = 'pgv',
        sparse_sim: bool = False,
        n_stations: int = 80,
    ):
        assert channel in self.CHANNELS
        self.h5_path = str(h5_path)
        self.normalize = normalize
        self.only_valid = only_valid
        self.channel = channel
        self.sparse_sim = sparse_sim
        self.n_stations = n_stations
        self._h5 = None

        with h5py.File(self.h5_path, 'r') as f:
            all_ids = sorted(k for k in f.keys() if k.startswith('event_'))
        self.event_ids = event_ids if event_ids is not None else all_ids

        self._stats: Optional[dict] = None
        if normalize:
            self._stats = self._compute_or_load_stats()

    def _sparse_resample(self, field: np.ndarray, sigma: float = 0.05) -> np.ndarray:
        t_steps, height, width = field.shape
        rng = np.random.default_rng()
        idx_h = rng.integers(0, height, self.n_stations)
        idx_w = rng.integers(0, width, self.n_stations)
        xs = idx_w / width
        ys = idx_h / height
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 1, width),
            np.linspace(0, 1, height),
        )
        result = np.zeros_like(field)

        for t in range(t_steps):
            vals = field[t, idx_h, idx_w]
            new_field = np.zeros((height, width), dtype=np.float32)
            weight_sum = np.zeros((height, width), dtype=np.float32)

            for x, y, val in zip(xs, ys, vals):
                if val <= 0:
                    continue
                dist2 = (grid_x - x) ** 2 + (grid_y - y) ** 2
                weight = np.exp(-dist2 / (2 * sigma ** 2))
                new_field += weight * val
                weight_sum += weight

            mask = weight_sum > 1e-10
            new_field[mask] /= weight_sum[mask]
            result[t] = new_field

        return result

    def _compute_or_load_stats(self) -> dict:
        with h5py.File(self.h5_path, 'r') as f:
            if 'normalization_diff' in f:
                ng = f['normalization_diff']
                return {
                    'dpga': (float(ng.attrs['dpga_mean']), float(ng.attrs['dpga_std'])),
                    'dpgv': (float(ng.attrs['dpgv_mean']), float(ng.attrs['dpgv_std'])),
                    'dint': (float(ng.attrs['dint_mean']), float(ng.attrs['dint_std'])),
                }

        dpga_vals, dpgv_vals, dint_vals = [], [], []
        with h5py.File(self.h5_path, 'r') as f:
            sample_ids = self.event_ids[:min(200, len(self.event_ids))]
            for eid in sample_ids:
                inp = f[eid]['inputs']
                dpga_vals.append(np.diff(inp['pga_field'][:], axis=0).ravel())
                dpgv_vals.append(np.diff(inp['pgv_field'][:], axis=0).ravel())
                dint_vals.append(np.diff(inp['intensity_field'][:], axis=0).ravel())

        def mean_std(arrs):
            arr = np.concatenate(arrs)
            return float(arr.mean()), max(float(arr.std()), 1e-8)

        pm, ps = mean_std(dpga_vals)
        vm, vs = mean_std(dpgv_vals)
        im, is_ = mean_std(dint_vals)

        with h5py.File(self.h5_path, 'r+') as f:
            ng = f.require_group('normalization_diff')
            ng.attrs['dpga_mean'] = pm
            ng.attrs['dpga_std'] = ps
            ng.attrs['dpgv_mean'] = vm
            ng.attrs['dpgv_std'] = vs
            ng.attrs['dint_mean'] = im
            ng.attrs['dint_std'] = is_

        return {'dpga': (pm, ps), 'dpgv': (vm, vs), 'dint': (im, is_)}

    def _norm(self, x: np.ndarray) -> np.ndarray:
        key = self._CH_STAT[self.channel]
        mean, std = self._stats[key]
        return (x - mean) / std

    @property
    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass

    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, idx: int):
        eid = self.event_ids[idx]
        grp = self._file[eid]
        inp = grp['inputs']
        lbl = grp['labels']

        field = inp[self._CH_FIELD[self.channel]][:]
        times = inp['times'][:]

        if self.sparse_sim:
            field = self._sparse_resample(field)

        rup = lbl['rupture_grid'][:]
        mag = lbl['magnitude'][:]
        valid = lbl['valid_frames'][:] if 'valid_frames' in lbl else np.ones(len(times), dtype=np.float32)
        epicenter = grp['epicenter'][:]
        strike = np.float32(grp.attrs.get('strike_rad', grp.attrs.get('strike', 0.0)))

        dfield = np.diff(field, axis=0).astype(np.float32)
        rup_diff = rup[1:].astype(np.float32)
        mag_diff = mag[1:].astype(np.float32)
        valid_diff = (valid[:-1] * valid[1:]).astype(np.float32)
        times_diff = ((times[:-1] + times[1:]) / 2).astype(np.float32)

        if self.only_valid:
            valid_idx = np.where(valid_diff > 0.5)[0]
            if len(valid_idx) == 0:
                valid_idx = np.arange(len(valid_diff))
            dfield = dfield[valid_idx]
            rup_diff = rup_diff[valid_idx]
            mag_diff = mag_diff[valid_idx]
            valid_diff = valid_diff[valid_idx]
            times_diff = times_diff[valid_idx]

        if self.normalize and self._stats:
            dfield = self._norm(dfield)

        inputs = dfield[np.newaxis].astype(np.float32)

        meta = dict(grp.attrs)
        meta['event_id'] = eid
        meta['channel'] = self.channel

        return {
            'inputs': torch.from_numpy(inputs),
            'rup_grid': torch.from_numpy(rup_diff),
            'magnitude': torch.from_numpy(mag_diff),
            'times': torch.from_numpy(times_diff),
            'epicenter': torch.from_numpy(epicenter.astype(np.float32)),
            'valid_frame': torch.from_numpy(valid_diff),
            'strike': torch.tensor(strike, dtype=torch.float32),
            'meta': meta,
        }

    @classmethod
    def train_val_test_split(
        cls,
        h5_path: str,
        ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        normalize: bool = True,
        only_valid: bool = True,
        channel: str = 'pgv',
        sparse_sim: bool = False,
        n_stations: int = 80,
    ):
        with h5py.File(h5_path, 'r') as f:
            all_ids = sorted(k for k in f.keys() if k.startswith('event_'))

        rng = np.random.default_rng(seed)
        ids = rng.permutation(all_ids).tolist()
        n = len(ids)
        n1 = int(n * ratios[0])
        n2 = int(n * ratios[1])

        train_ds = cls(
            h5_path,
            ids[:n1],
            normalize,
            only_valid,
            channel,
            sparse_sim=sparse_sim,
            n_stations=n_stations,
        )
        val_ds = cls(
            h5_path,
            ids[n1:n1 + n2],
            normalize,
            only_valid,
            channel,
            sparse_sim=False,
        )
        test_ds = cls(
            h5_path,
            ids[n1 + n2:],
            normalize,
            only_valid,
            channel,
            sparse_sim=False,
        )

        if normalize:
            val_ds._stats = train_ds._stats
            test_ds._stats = train_ds._stats

        return train_ds, val_ds, test_ds


def collate_fn(batch):
    batch_size = len(batch)
    lengths = [b['inputs'].shape[1] for b in batch]
    t_max = max(lengths)
    height, width = batch[0]['inputs'].shape[2], batch[0]['inputs'].shape[3]

    out = {
        'inputs': torch.zeros(batch_size, 1, t_max, height, width),
        'rup_grid': torch.zeros(batch_size, t_max, height, width),
        'magnitude': torch.zeros(batch_size, t_max),
        'times': torch.zeros(batch_size, t_max),
        'valid_frame': torch.zeros(batch_size, t_max),
        'seq_mask': torch.zeros(batch_size, t_max, dtype=torch.bool),
        'strike': torch.zeros(batch_size, dtype=torch.float32),
        'epicenter': torch.zeros(batch_size, 2, dtype=torch.float32),
        'meta': [b['meta'] for b in batch],
    }

    for i, sample in enumerate(batch):
        t = lengths[i]
        out['inputs'][i, :, :t] = sample['inputs']
        out['rup_grid'][i, :t] = sample['rup_grid']
        out['magnitude'][i, :t] = sample['magnitude']
        out['times'][i, :t] = sample['times']
        out['valid_frame'][i, :t] = sample['valid_frame']
        out['seq_mask'][i, :t] = True
        out['strike'][i] = sample['strike']
        out['epicenter'][i] = sample['epicenter']

    return out
