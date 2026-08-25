#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/3/18 19:35
# @Author  : 上头欢乐送、
# @File    : check.py
# @Software: PyCharm
# 学习新思想，争做新青年

"""
check.py — Visualise each event as a GIF.

Frame validity
--------------
valid_frame = 1  (t ≤ t_complete + Δt):
    field(t) still carries new information about the rupture.
    label rupture_grid(t - Δt) is unique for each field.
    → included in training (loss_weight = 1.0)

valid_frame = 0  (t > t_complete + Δt):
    rupture_grid is frozen (final state).
    field is still changing (far-field S-waves) but label is not.
    → excluded from training (loss_weight = 0.0)

Visual cues:
    Green border on rupture panel = valid frame
    Red border                    = invalid (ambiguous) frame
    Red shading in time series    = invalid region
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tqdm import tqdm
import argparse


def make_event_gif(h5_path, event_id, save_dir):
    with h5py.File(h5_path, 'r') as f:
        grp   = f[event_id]
        pga   = grp['inputs']['pga_field'][:]
        pgv   = grp['inputs']['pgv_field'][:]
        rup   = grp['labels']['rupture_grid'][:]
        times = grp['inputs']['times'][:]
        epi   = grp['epicenter'][:]
        attrs = dict(grp.attrs)
        attrs.update({k: v for k, v in f.attrs.items() if k not in attrs})

        lbl = grp['labels']
        valid_frame  = lbl['valid_frames'][:]     if 'valid_frames'     in lbl \
                       else np.ones(rup.shape[0], dtype=np.float32)
        rup_progress = lbl['rupture_progress'][:] if 'rupture_progress' in lbl \
                       else np.linspace(0, 1, rup.shape[0], dtype=np.float32)
        rup_complete = lbl['rupture_complete'][:] if 'rupture_complete' in lbl \
                       else np.zeros(rup.shape[0], dtype=np.float32)

    T  = pga.shape[0]
    gs = pga.shape[1]

    max_pga_obs = [pga[k].max() for k in range(T)]
    max_pgv_obs = [pgv[k].max() for k in range(T)]
    g_pga = float(max(max_pga_obs)) if max(max_pga_obs) > 0 else 1.0
    g_pgv = float(max(max_pgv_obs)) if max(max_pgv_obs) > 0 else 1.0
    g_rup = float(rup.max())        if rup.max()        > 0 else 1.0

    # Key time boundaries
    t_complete = float(attrs.get('t_complete_s', times[-1]))
    delta_t    = float(attrs.get('delta_t_s',    0.0))
    t_cutoff   = t_complete + delta_t   # last valid frame boundary

    fault_type = attrs.get('fault_type', 'N/A')
    mag_f      = attrs.get('magnitude_final', 0.0)
    gmpe       = attrs.get('gmpe', 'N/A')
    cx, cy     = float(epi[0]), float(epi[1])

    # ── Layout ────────────────────────────────────────────────────────────────
    # Row 0: PGA field | rupture label | info
    # Row 1: PGV field | progress bar  | note
    # Row 2: time series (full width)
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#111111')
    gspec = GridSpec(3, 3, figure=fig,
                     height_ratios=[5, 1.8, 4],
                     hspace=0.40, wspace=0.30)

    ax_pga  = fig.add_subplot(gspec[0, 0])
    ax_rup  = fig.add_subplot(gspec[0, 1])
    ax_info = fig.add_subplot(gspec[0, 2])
    ax_pgv  = fig.add_subplot(gspec[1, 0])
    ax_prog = fig.add_subplot(gspec[1, 1])
    ax_note = fig.add_subplot(gspec[1, 2])
    ax_ts   = fig.add_subplot(gspec[2, :])

    for ax in [ax_pga, ax_rup, ax_info, ax_pgv, ax_prog, ax_note, ax_ts]:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white', labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')

    # ── Image handles ─────────────────────────────────────────────────────────
    dummy   = np.zeros((gs, gs))
    im_pga  = ax_pga.imshow(dummy, cmap='plasma',  origin='lower', vmin=0, vmax=g_pga)
    im_pgv  = ax_pgv.imshow(dummy, cmap='viridis', origin='lower', vmin=0, vmax=g_pgv)
    im_rup  = ax_rup.imshow(dummy, cmap='Reds',    origin='lower', vmin=0, vmax=g_rup)

    for ax in [ax_pga, ax_rup, ax_pgv]:
        ax.scatter([cx], [cy], c='cyan', marker='*', s=80, zorder=10)
        ax.axis('off')

    ax_pga.set_title('PGA Field  [input]',
                      color='#aaaaff', fontsize=9, fontweight='bold')
    ax_pgv.set_title('PGV Field  [input]',
                      color='#aaffaa', fontsize=8, fontweight='bold')
    ax_rup.set_title('Rupture Grid(t−Δt)  [label]',
                      color='#ff6666', fontsize=9, fontweight='bold')

    plt.colorbar(im_pga, ax=ax_pga, fraction=0.046).ax.tick_params(colors='white', labelsize=6)
    plt.colorbar(im_rup, ax=ax_rup, fraction=0.046).ax.tick_params(colors='white', labelsize=6)

    # Border around rupture panel — changes colour with validity
    rup_border = plt.Rectangle((0, 0), 1, 1, fill=False, lw=3,
                                 edgecolor='#00cc44',
                                 transform=ax_rup.transAxes, zorder=20,
                                 clip_on=False)
    ax_rup.add_patch(rup_border)

    # ── Info panel ────────────────────────────────────────────────────────────
    ax_info.axis('off')
    info_txt = ax_info.text(
        0.05, 0.95, '', transform=ax_info.transAxes,
        color='white', fontsize=8, va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#222', alpha=0.8))
    state_badge = ax_info.text(
        0.5, 0.07, '', transform=ax_info.transAxes,
        color='black', fontsize=10, fontweight='bold', ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#00cc44', alpha=0.9))

    # ── Progress / weight panel ───────────────────────────────────────────────
    ax_prog.axis('off')
    ax_prog.barh(0.65, 1.0, height=0.3, left=0,
                  color='#333333', transform=ax_prog.transAxes)
    prog_bar = ax_prog.barh(0.65, 0.0, height=0.3, left=0,
                             color='#00cc44', transform=ax_prog.transAxes)[0]
    prog_txt   = ax_prog.text(0.5, 0.32, '', transform=ax_prog.transAxes,
                               color='white', fontsize=8, ha='center', family='monospace')
    weight_txt = ax_prog.text(0.5, 0.88, '', transform=ax_prog.transAxes,
                               color='white', fontsize=10, fontweight='bold', ha='center',
                               bbox=dict(boxstyle='round', facecolor='#333', alpha=0.9))

    # ── Explanation note ──────────────────────────────────────────────────────
    ax_note.axis('off')
    ax_note.text(0.5, 0.5,
        f'Label = rupture_grid(t − Δt)\n'
        f'Δt = {delta_t:.1f} s  (S-wave delay)\n\n'
        f'Valid: Δt ≤ t ≤ t_complete+Δt\n'
        f'  [{delta_t:.1f}s, {t_cutoff:.1f}s]\n\n'
        f'Invalid (start): field empty\n'
        f'Invalid (end):   label frozen\n'
        f'→ loss_weight = 0 for both',
        transform=ax_note.transAxes,
        color='#cccccc', fontsize=8, ha='center', va='center',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.9))

    # ── Time series ───────────────────────────────────────────────────────────
    ln_pga, = ax_ts.plot([], [], 'r-', lw=1.5, label='PGA max')
    ln_pgv, = ax_ts.plot([], [], 'b-', lw=1.5, label='PGV max')
    ax_ts.set_xlim(0, times[-1])
    ax_ts.set_ylim(0, max(g_pga * 1.1, 1.0))
    ax_ts.set_xlabel('Time (s)', color='white', fontsize=8)
    ax_ts.set_ylabel('Value',    color='white', fontsize=8)
    ax_ts.set_title('Ground Motion Time Series', color='white',
                     fontsize=9, fontweight='bold')
    ax_ts.grid(True, alpha=0.2)

    # Shaded regions: invalid at BOTH ends
    ax_ts.axvspan(0, delta_t, alpha=0.15, color='#ff4444',
                   label=f'invalid: field empty (t < Δt={delta_t:.1f}s)')
    ax_ts.axvspan(t_cutoff, times[-1], alpha=0.15, color='#ff4444',
                   label=f'invalid: label frozen (t > {t_cutoff:.1f}s)')
    ax_ts.axvline(x=delta_t,    color='#ff4444', lw=1.5, linestyle='--', alpha=0.9)
    ax_ts.axvline(x=t_complete, color='white',   lw=1.0, linestyle=':',
                   alpha=0.6, label=f't_complete={t_complete:.1f}s')
    ax_ts.axvline(x=t_cutoff,   color='#ff4444', lw=1.5, linestyle='--',
                   alpha=0.9, label=f't_cutoff={t_cutoff:.1f}s')

    vline = ax_ts.axvline(x=0, color='yellow', lw=1.5, alpha=0.9, label='t (current)')
    ax_ts.legend(fontsize=7, facecolor='#222', labelcolor='white')

    time_txt = fig.text(0.5, 0.002, '', ha='center', color='#aaa', fontsize=8)
    fig.suptitle(
        f'{event_id}  |  {fault_type}  |  M{mag_f:.2f}  |  {gmpe}  |  '
        f't_complete={t_complete:.1f}s  Δt={delta_t:.1f}s  t_cutoff={t_cutoff:.1f}s',
        color='white', fontsize=9, fontweight='bold', y=0.999)

    # ── Update ────────────────────────────────────────────────────────────────
    def update(k):
        t     = times[k]
        valid = float(valid_frame[k]) > 0.5
        prog  = float(rup_progress[k])
        comp  = float(rup_complete[k]) > 0.5

        im_pga.set_data(pga[k])
        im_pgv.set_data(pgv[k])
        im_rup.set_data(rup[k])

        ln_pga.set_data(times[:k+1], max_pga_obs[:k+1])
        ln_pgv.set_data(times[:k+1], max_pgv_obs[:k+1])
        vline.set_xdata([t, t])

        # Border colour: green = valid, red = invalid
        rup_border.set_edgecolor('#00cc44' if valid else '#ff4444')

        # Progress bar
        prog_bar.set_width(prog)
        prog_bar.set_facecolor('#ff4444' if comp else '#00cc44')
        prog_txt.set_text(f't = {t:.1f}s  |  progress = {prog*100:.0f}%')

        # Loss weight badge
        w = 1.0 if valid else 0.0
        weight_txt.set_text(f'loss_weight = {w:.1f}')
        weight_txt.get_bbox_patch().set_facecolor(
            '#004400' if valid else '#440000')

        # State badge
        if not valid:
            state_badge.set_text('INVALID (ambiguous)')
            state_badge.get_bbox_patch().set_facecolor('#ff4444')
            state_badge.set_color('white')
        elif comp:
            state_badge.set_text('POST-RUPTURE (valid)')
            state_badge.get_bbox_patch().set_facecolor('#ff8800')
            state_badge.set_color('white')
        else:
            state_badge.set_text('RUPTURING (valid)')
            state_badge.get_bbox_patch().set_facecolor('#00cc44')
            state_badge.set_color('black')

        info_txt.set_text(
            f"t        = {t:.1f} s\n"
            f"t_cutoff = {t_cutoff:.1f} s\n"
            f"frame    : {k+1}/{T}\n\n"
            f"PGA max  : {max_pga_obs[k]:.3f}\n"
            f"PGV max  : {max_pgv_obs[k]:.3f}\n"
            f"Rup nz   : {rup[k].mean():.5f}\n\n"
            f"depth    : {attrs.get('depth_km',  0):.1f} km\n"
            f"dip      : {attrs.get('dip_deg',   0):.1f} deg\n"
            f"rake     : {attrs.get('rake_deg',  0):.1f} deg\n"
            f"v_rup    : {attrs.get('rupture_velocity_km_s', 0):.1f} km/s"
        )
        time_txt.set_text(
            f't={t:.1f}s  |  '
            f'{"INVALID" if not valid else "POST-RUPTURE" if comp else "RUPTURING"}  |  '
            f'loss_weight={w:.1f}  |  frame {k+1}/{T}')

        return (im_pga, im_pgv, im_rup,
                ln_pga, ln_pgv, vline, rup_border,
                info_txt, state_badge,
                prog_bar, prog_txt, weight_txt, time_txt)

    ani = animation.FuncAnimation(fig, update, frames=T, interval=300, blit=True)
    out = Path(save_dir) / f'{event_id}.gif'
    ani.save(str(out), writer='pillow', fps=4, dpi=90)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5',      default='data/events.h5')
    p.add_argument('--out',     default='gifs')
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5, 'r') as f:
        event_ids = sorted(k for k in f.keys() if k.startswith('event_'))

    if args.workers <= 1:
        for eid in tqdm(event_ids, desc='Rendering GIFs', unit='event'):
            try:
                make_event_gif(args.h5, eid, args.out)
            except Exception as e:
                tqdm.write(f'Failed {eid}: {e}')
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        jobs = [(args.h5, eid, args.out) for eid in event_ids]
        def _worker(job):
            make_event_gif(*job)
            return job[1]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_worker, j): j[1] for j in jobs}
            pbar = tqdm(total=len(futs), desc='Rendering GIFs', unit='event')
            for fut in as_completed(futs):
                try: fut.result()
                except Exception as e: tqdm.write(f'Failed {futs[fut]}: {e}')
                pbar.update(1)
            pbar.close()


if __name__ == '__main__':
    main()