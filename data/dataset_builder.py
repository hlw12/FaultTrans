#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/3/17 01:58
# @Author  : 上头欢乐送、
# @File    : dataset_builder.py
# @Software: PyCharm
# 学习新思想，争做新青年

import sys
import time
import argparse
import logging
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import h5py
import yaml

from Simulator import ShallowEarthquakeSimulator, _GMPE_POOL, SimulationConfig

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(),
              logging.FileHandler('dataset_build.log')])
log = logging.getLogger(__name__)


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def sample_rake(rake_cfg: dict, rng) -> float:
    """
    支持两种 rake 采样格式：

    旧格式（离散）：
        rake_deg:
          choices: [0.0, 90.0, -90.0, 45.0, -45.0]
          weights: [0.35, 0.25, 0.20, 0.10, 0.10]

    新格式（按断层类型区间均匀采样）：
        rake_deg:
          distribution: uniform_per_type
          strike_slip:
            range: [-30.0, 30.0]
            weight: 0.35
          reverse:
            range: [60.0, 120.0]
            weight: 0.25
          normal:
            range: [-120.0, -60.0]
            weight: 0.20
          oblique_rs:
            range: [30.0, 60.0]
            weight: 0.10
          oblique_ns:
            range: [-60.0, -30.0]
            weight: 0.10

    新格式采样流程：
      1. 按各类型 weight 随机选取断层类型
      2. 在该类型的 range 内均匀采样 rake 角
    """
    if rake_cfg.get('distribution') == 'uniform_per_type':
        # 提取各断层类型条目（排除 'distribution' 键本身）
        type_keys = [k for k in rake_cfg if k != 'distribution']
        weights   = np.array([rake_cfg[t]['weight'] for t in type_keys],
                             dtype=float)
        weights  /= weights.sum()   # 归一化，防止配置文件浮点误差

        chosen = rng.choice(len(type_keys), p=weights)
        lo, hi = rake_cfg[type_keys[chosen]]['range']
        return float(rng.uniform(lo, hi))
    else:
        # 旧格式：离散采样
        return float(rng.choice(rake_cfg['choices'], p=rake_cfg['weights']))


def sample_config(scfg: dict, gmpe_names: list) -> SimulationConfig:
    rng = np.random
    mf  = rng.uniform(scfg['magnitude']['min'], scfg['magnitude']['max'])
    offset = rng.uniform(scfg['initial_magnitude_offset']['min'],
                          scfg['initial_magnitude_offset']['max'])
    m0    = max(3.5, mf - offset)
    depth = rng.uniform(scfg['depth_km']['min'], scfg['depth_km']['max'])
    dip   = rng.uniform(scfg['dip_deg']['min'],  scfg['dip_deg']['max'])

    rake_cfg = scfg['rake_deg']
    rake     = sample_rake(rake_cfg, rng)

    ft_cfg   = scfg['fault_type']
    ft_names = list(ft_cfg.keys())
    ft_probs = [ft_cfg[k] for k in ft_names]
    fault_type = str(rng.choice(ft_names, p=ft_probs))

    n_segs   = int(rng.randint(scfg['n_fault_segments']['min'],
                                scfg['n_fault_segments']['max'] + 1))
    max_dk   = rng.uniform(scfg['max_strike_change_deg']['min'],
                            scfg['max_strike_change_deg']['max'])
    bil_ratio = rng.uniform(scfg['bilateral_ratio']['min'],
                             scfg['bilateral_ratio']['max'])

    ev_cfg   = scfg['magnitude_evolution_type']
    ev_names = list(ev_cfg.keys())
    ev_probs = [ev_cfg[k] for k in ev_names]
    ev_type  = str(rng.choice(ev_names, p=ev_probs))

    v_rup    = rng.uniform(scfg['rupture_velocity_km_s']['min'],
                            scfg['rupture_velocity_km_s']['max'])
    vs30_mean = rng.uniform(scfg['vs30_mean']['min'], scfg['vs30_mean']['max'])
    vs30_std  = vs30_mean * scfg['vs30_std_fraction']
    gmpe_name = str(rng.choice(gmpe_names))

    return SimulationConfig(
        magnitude                = float(mf),
        initial_magnitude        = float(m0),
        magnitude_evolution_type = ev_type,
        depth                    = float(depth),
        strike                   = None,
        dip                      = float(dip),
        rake                     = float(rake),
        rupture_velocity         = float(v_rup),
        fault_type               = fault_type,
        n_fault_segments         = n_segs,
        max_strike_change_deg    = float(max_dk),
        bilateral_ratio          = float(bil_ratio),
        total_duration           = float(scfg['total_duration_s']),
        n_time_steps             = int(scfg['n_time_steps']),
        grid_size                = int(scfg['grid_size']),
        resolution               = float(scfg['resolution_deg']),
        n_stations               = int(scfg['n_stations']),
        vs30_mean                = float(vs30_mean),
        vs30_std                 = float(vs30_std),
        s_wave_velocity          = float(scfg['s_wave_velocity_km_s']),
        gmpe_name                = gmpe_name,
    )


def _run_one_event(args):
    event_idx, scfg, gmpe_names, seed = args
    np.random.seed(seed)
    try:
        cfg = sample_config(scfg, gmpe_names)
        sim = ShallowEarthquakeSimulator(cfg)
        seq = sim.generate_synthetic_earthquake()

        for t_idx in [0, 5, 10, 19]:
            grid = seq[t_idx]['rupture_grid']
            print(f"t={t_idx}  nonzero={grid.mean():.4f}  shape={grid.shape}")

        T = len(seq)

        pga_fields  = np.stack([s['pga_field']       for s in seq])
        pgv_fields  = np.stack([s['pgv_field']       for s in seq])
        int_fields  = np.stack([s['intensity_field'] for s in seq])
        rup_grids    = np.stack([s['rupture_grid'] for s in seq])
        magnitudes   = np.array([s['magnitude']       for s in seq], dtype=np.float32)
        times        = np.array([s['time']            for s in seq], dtype=np.float32)
        rup_progress = np.array([s['rupture_progress'] for s in seq], dtype=np.float32)
        rup_complete = np.array([s['rupture_complete'] for s in seq], dtype=np.float32)
        valid_frames = np.array([s['valid_frame'] for s in seq], dtype=np.float32)



        center = seq[0]['metadata']['rupture_center']

        meta = {
            'event_idx':              event_idx,
            'gmpe':                   cfg.gmpe_name,
            'fault_type':             cfg.fault_type,
            'magnitude_final':        cfg.magnitude,
            'magnitude_initial':      cfg.initial_magnitude,
            'magnitude_evolution':    cfg.magnitude_evolution_type,
            'depth_km':               cfg.depth,
            'dip_deg':                cfg.dip,
            'rake_deg':               cfg.rake,
            'strike_rad':             sim._strike,
            'rupture_velocity_km_s':  cfg.rupture_velocity,
            'n_fault_segments':       cfg.n_fault_segments,
            'max_strike_change_deg':  cfg.max_strike_change_deg,
            'bilateral_ratio':        cfg.bilateral_ratio,
            'vs30_mean':              cfg.vs30_mean,
            'vs30_std':               cfg.vs30_std,
            'epicenter_x':            center[0],
            'epicenter_y':            center[1],
            'n_time_steps':           T,
            't_complete_s':           float(seq[0]['metadata']['t_complete']),
            'delta_t_s':              float(seq[0]['metadata']['delta_t']),
        }

        return event_idx, {
            'pga_fields':    pga_fields.astype(np.float32),
            'pgv_fields':    pgv_fields.astype(np.float32),
            'int_fields':    int_fields.astype(np.float32),
            'rup_grids':     rup_grids.astype(np.float32),
            'magnitudes':    magnitudes,
            'times':         times,
            'rup_progress':  rup_progress,
            'rup_complete':  rup_complete,
            'valid_frames':  valid_frames,
            'epicenter':       np.array(center, dtype=np.float32),
            'meta':          meta,
        }

    except Exception:
        log.error(f"Event {event_idx} failed:\n{traceback.format_exc()}")
        return event_idx, None


def write_event_to_hdf5(h5: h5py.File, event_id: str, data: dict):
    grp = h5.require_group(event_id)

    inp = grp.require_group('inputs')
    inp.create_dataset('pga_field',       data=data['pga_fields'],
                       compression='gzip', compression_opts=4)
    inp.create_dataset('pgv_field',       data=data['pgv_fields'],
                       compression='gzip', compression_opts=4)
    inp.create_dataset('intensity_field', data=data['int_fields'],
                       compression='gzip', compression_opts=4)
    inp.create_dataset('times',           data=data['times'])

    lbl = grp.require_group('labels')
    lbl.create_dataset('rupture_grid', data=data['rup_grids'],
                       compression='gzip', compression_opts=4)

    lbl.create_dataset('magnitude',    data=data['magnitudes'])

    # rupture state + label alignment datasets
    lbl.create_dataset('rupture_progress', data=data['rup_progress'])
    lbl.create_dataset('rupture_complete', data=data['rup_complete'])
    lbl.create_dataset('valid_frames',     data=data['valid_frames'])


    grp.create_dataset('epicenter', data=data['epicenter'])
    for k, v in data['meta'].items():
        try:
            grp.attrs[k] = v
        except Exception:
            grp.attrs[k] = str(v)


def build_dataset(
    sampling_config_path: str,
    gmpe_config_path:     str,
    output_h5_path:       str,
    n_events:             int,
    workers:              int = 1,
    base_seed:            int = 42,
):
    Path(output_h5_path).parent.mkdir(parents=True, exist_ok=True)

    scfg       = load_yaml(sampling_config_path)
    _gcfg      = load_yaml(gmpe_config_path)
    gmpe_names = list(_GMPE_POOL.keys())

    log.info(f"Building dataset: {n_events} events | "
             f"GMPEs: {gmpe_names} | workers: {workers}")
    log.info(f"Output: {output_h5_path}")

    seeds    = np.random.SeedSequence(base_seed).generate_state(n_events)
    job_args = [(i, scfg, gmpe_names, int(seeds[i])) for i in range(n_events)]

    metadata_rows = []
    success = 0; fail = 0
    t0 = time.time()

    with h5py.File(output_h5_path, 'w') as h5:
        h5.attrs['n_events_target'] = n_events
        h5.attrs['gmpe_pool']       = str(gmpe_names)
        h5.attrs['grid_size']       = scfg['grid_size']
        h5.attrs['resolution_deg']  = scfg['resolution_deg']
        h5.attrs['n_time_steps']    = scfg['n_time_steps']

        if workers <= 1:
            for args in job_args:
                idx, result = _run_one_event(args)
                event_id    = f"event_{idx:05d}"
                if result is not None:
                    write_event_to_hdf5(h5, event_id, result)
                    metadata_rows.append(result['meta'])
                    success += 1
                    if success % 50 == 0:
                        elapsed = time.time() - t0
                        log.info(f"  {success}/{n_events} done "
                                 f"({elapsed:.0f}s, {elapsed/success:.1f}s/event)")
                else:
                    fail += 1
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_run_one_event, a): a[0] for a in job_args}
                for fut in as_completed(futures):
                    idx, result = fut.result()
                    event_id    = f"event_{idx:05d}"
                    if result is not None:
                        write_event_to_hdf5(h5, event_id, result)
                        metadata_rows.append(result['meta'])
                        success += 1
                        if success % 50 == 0:
                            elapsed = time.time() - t0
                            log.info(f"  {success}/{n_events} done ({elapsed:.0f}s)")
                    else:
                        fail += 1

        h5.attrs['n_events_success'] = success
        h5.attrs['n_events_failed']  = fail

    meta_path = str(Path(output_h5_path).with_suffix('')) + '_metadata.csv'
    pd.DataFrame(metadata_rows).to_csv(meta_path, index=False)

    elapsed = time.time() - t0
    log.info(f"\n{'='*60}")
    log.info(f"Done. Success: {success}  Failed: {fail}")
    log.info(f"Total time: {elapsed:.1f}s  ({elapsed/max(success,1):.1f}s/event)")
    log.info(f"HDF5 : {output_h5_path}")
    log.info(f"CSV  : {meta_path}")
    log.info(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(description='Earthquake dataset builder')
    parser.add_argument('--config',      default='config/sampling.yml')
    parser.add_argument('--gmpe_config', default='config/gmpe.yml')
    parser.add_argument('--output',      default='data/events.h5')
    parser.add_argument('--n_events',    type=int, default=5000)
    parser.add_argument('--workers',     type=int, default=1)
    parser.add_argument('--seed',        type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build_dataset(
        sampling_config_path = args.config,
        gmpe_config_path     = args.gmpe_config,
        output_h5_path       = args.output,
        n_events             = args.n_events,
        workers              = args.workers,
        base_seed            = args.seed,
    )