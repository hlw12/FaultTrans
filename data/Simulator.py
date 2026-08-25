#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/3/17 01:57
# @Author  : 上头欢乐送、
# @File    : Simulator.py
# @Software: PyCharm
# 学习新思想，争做新青年

"""
simulator.py
Core earthquake simulator:
  - Multi-GMPE support (ASK14 / BSSA14 / CB14 / CY14)
  - Diverse fault geometry (linear / polyline / bilateral / bilateral_polyline)
  - Per-frame GMPE(mag_t, rrup_t) + history/max accumulation

Label alignment
---------------
The PGA/PGV field at time t reflects rupture that occurred at t - Δt,
where Δt is the mean S-wave travel time from the rupture plane to the
station ensemble.

Δt is derived analytically from the station generation distribution:
    along-strike : a ~ Normal(0, gs*0.3)       → E[a²] = (gs*0.3)²
    perpendicular: p ~ Uniform(-gs*0.4, gs*0.4) → E[p²] = (gs*0.4)²/3
    mean planar distance (grids) = sqrt(E[a²] + E[p²])
    mean planar distance (km)    = above * resolution * 111
    Δt = sqrt(d_planar² + depth²) / v_s

The stored rupture_grid is therefore rupture_grid(t - Δt), not
rupture_grid(t).  This ensures that the label and the field describe
the same physical instant.

Additional fields per time step
--------------------------------
rupture_complete : 0.0 while propagating, 1.0 after rupture finishes
rupture_progress : fraction of final rupture length reached (0→1)
delta_t_s        : the Δt used (stored in metadata for the viewer)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

_OQ_AVAILABLE = False
try:
    from openquake.hazardlib import imt, const
    from openquake.hazardlib.contexts import RuptureContext
    from openquake.hazardlib.gsim.abrahamson_2014 import AbrahamsonEtAl2014
    from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
    from openquake.hazardlib.gsim.campbell_bozorgnia_2014 import CampbellBozorgnia2014
    from openquake.hazardlib.gsim.chiou_youngs_2014 import ChiouYoungs2014
    _OQ_AVAILABLE = True
except Exception as e:
    print(e)


@dataclass
class SimulationConfig:
    magnitude:                float = 7.2
    initial_magnitude:        float = 4.8
    magnitude_evolution_type: str   = 'sqrt'
    depth:                    float = 12.0
    strike:          Optional[float] = None
    dip:                      float = 85.0
    rake:                     float = 0.0
    rupture_velocity:         float = 3.0
    fault_type:               str   = 'bilateral_polyline'
    n_fault_segments:         int   = 4
    max_strike_change_deg:    float = 25.0
    bilateral_ratio:          float = 0.70
    total_duration:           float = 40.0
    n_time_steps:             int   = 20
    grid_size:                int   = 150
    resolution:               float = 0.01
    n_stations:               int   = 50
    vs30_mean:                float = 500.0
    vs30_std:                 float = 150.0
    s_wave_velocity:          float = 3.5
    gmpe_name:       Optional[str]  = None


# ─────────────────────────── GMPE layer ──────────────────────────────────────

class _AnalyticalGMPE:
    """
    Base class for approximate analytical GMPEs.

    PGA and PGV are computed from **independent** regression equations.
    Key differences between the two IMs:
      - Magnitude scaling : PGV coefficient ~1.05–1.12  vs PGA ~0.97–0.98
        (PGV is more sensitive to Mw because it is dominated by ~1 Hz energy)
      - Distance attenuation: PGV coefficient ~0.95–1.02 vs PGA ~1.06–1.09
        (low-frequency waves spread more slowly in amplitude)
      - Site amplification : PGV c ~0.45–0.55          vs PGA c ~0.28–0.35
        (resonance at 1 Hz is stronger for typical soft-soil conditions)
    """
    name = 'base';  sigma_pga = 0.65;  sigma_pgv = 0.68

    def compute(self, magnitude, rrup, vs30,
                rake=0.0, dip=90.0, ztor=0.0, width=15.0) -> Dict:
        lnP  = self._ln_pga(magnitude, rrup, vs30, rake, dip, ztor, width)
        lnV  = self._ln_pgv(magnitude, rrup, vs30, rake, dip, ztor, width)
        pga  = np.exp(lnP)
        pgv  = np.exp(lnV)
        sa03 = pga * self._sa03_factor(magnitude, rrup)
        return {'PGA':    (pga,  self.sigma_pga),
                'PGV':    (pgv,  self.sigma_pgv),
                'SA_0.3': (sa03, self.sigma_pga)}

    def _ln_pga(self, M, R, vs30, rake, dip, ztor, width):
        raise NotImplementedError

    def _ln_pgv(self, M, R, vs30, rake, dip, ztor, width):
        raise NotImplementedError

    @staticmethod
    def _sa03_factor(M, R):
        return np.where(R < 50, 1.6 + 0.05*M, 1.2 + 0.03*M)

    @staticmethod
    def _site_term(vs30, c=0.3, vref=760.0):
        return np.where(vs30 < vref,
                        c * np.log(vref / np.maximum(vs30, 1.0)), 0.0)


class _ASK14_Approx(_AnalyticalGMPE):
    """
    Abrahamson, Silva & Kamai (2014) — approximate form.

    PGV equation retains the hanging-wall rake correction (+0.18 ln units
    for reverse; ASK14 Table 3 shows PGV is more sensitive to faulting
    style than PGA because of its dominant period ~1 s).
    Pseudo-depth h is computed the same way as PGA (ztor + half-width
    projected to depth), consistent with the finite-fault geometry.
    """
    name = 'ASK14'

    def _ln_pga(self, M, R, vs30, rake, dip, ztor, width):
        h  = max(ztor + width*np.sin(np.radians(dip))/2, 1.0)
        Rh = np.sqrt(R**2 + h**2)
        ln = (-4.416 + 0.984*M - 1.090*np.log(np.maximum(Rh, 0.1))
              + self._site_term(vs30, c=0.35))
        ln += np.where((rake > 30) & (rake < 150), 0.12, 0.0)
        return ln

    def _ln_pgv(self, M, R, vs30, rake, dip, ztor, width):
        # Independent PGV regression for ASK14 approximate form.
        # Higher Mw scaling (1.10 vs 0.984), slower attenuation (1.00 vs 1.09),
        # stronger site term (c=0.50 vs 0.35).
        h  = max(ztor + width*np.sin(np.radians(dip))/2, 1.0)
        Rh = np.sqrt(R**2 + h**2)
        ln = (-1.975 + 1.100*M - 1.000*np.log(np.maximum(Rh, 0.1))
              + self._site_term(vs30, c=0.50))
        # Reverse-faulting amplification is larger for PGV (~0.18 vs 0.12)
        ln += np.where((rake > 30) & (rake < 150), 0.18, 0.0)
        return ln


class _BSSA14_Approx(_AnalyticalGMPE):
    """
    Boore, Stewart, Seyhan & Atkinson (2014) — approximate form.

    BSSA14 uses Rjb (not Rrup); the pseudo-depth 4.5 km is retained
    for both PGA and PGV.  The PGV site term uses c=0.48, reflecting
    BSSA14's Vs30-scaling being relatively strong at 1 Hz.
    """
    name = 'BSSA14'

    def _ln_pga(self, M, R, vs30, rake, dip, ztor, width):
        Rh = np.sqrt(R**2 + 4.5**2)
        return (-3.595 + 0.970*M - 1.062*np.log(np.maximum(Rh, 0.1))
                + self._site_term(vs30, c=0.30))

    def _ln_pgv(self, M, R, vs30, rake, dip, ztor, width):
        # BSSA14 PGV: higher Mw scaling (1.06 vs 0.970), slower attenuation
        # (1.00 vs 1.062), stronger site amplification (c=0.48 vs 0.30).
        Rh = np.sqrt(R**2 + 4.5**2)
        return (-1.386 + 1.060*M - 1.000*np.log(np.maximum(Rh, 0.1))
                + self._site_term(vs30, c=0.48))


class _CB14_Approx(_AnalyticalGMPE):
    """
    Campbell & Bozorgnia (2014) — approximate form.

    CB14 applies a sediment-depth correction via ztor for PGA.  The same
    correction is retained for PGV (slightly reduced coefficient -0.006
    vs -0.007 because PGV is less sensitive to near-surface geometry).
    """
    name = 'CB14'

    def _ln_pga(self, M, R, vs30, rake, dip, ztor, width):
        h  = max(ztor, 1.0)
        Rh = np.sqrt(R**2 + h**2)
        return (-4.416 + 0.984*M - 1.091*np.log(np.maximum(Rh, 0.1))
                - 0.007*np.minimum(ztor, 20.0)
                + self._site_term(vs30, c=0.32))

    def _ln_pgv(self, M, R, vs30, rake, dip, ztor, width):
        # CB14 PGV: stronger Mw scaling (1.08 vs 0.984), slower attenuation
        # (0.975 vs 1.091), stronger site term (c=0.46 vs 0.32).
        # ztor correction retained but slightly reduced.
        h  = max(ztor, 1.0)
        Rh = np.sqrt(R**2 + h**2)
        return (-1.862 + 1.080*M - 0.975*np.log(np.maximum(Rh, 0.1))
                - 0.006*np.minimum(ztor, 20.0)
                + self._site_term(vs30, c=0.46))


class _CY14_Approx(_AnalyticalGMPE):
    """
    Chiou & Youngs (2014) — approximate form.

    CY14 includes a direct Vs30 log-ratio term (0.014 * ln(Vs30/1130))
    in addition to the site_term.  For PGV the coefficient is enlarged
    to 0.025, reflecting greater sensitivity of 1-Hz motion to velocity
    contrasts.  The low-dip hanging-wall correction (+0.05 for PGA) is
    increased to +0.08 for PGV.
    """
    name = 'CY14'

    def _ln_pga(self, M, R, vs30, rake, dip, ztor, width):
        Rh = np.sqrt(R**2 + 6.0**2)
        ln = (-4.416 + 0.984*M - 1.080*np.log(np.maximum(Rh, 0.1))
              + 0.014*np.log(np.minimum(vs30, 1130)/1130)
              + self._site_term(vs30, c=0.28))
        ln += np.where(dip < 70, 0.05, 0.0)
        return ln

    def _ln_pgv(self, M, R, vs30, rake, dip, ztor, width):
        Rh = np.sqrt(R**2 + 6.0**2)
        ln = (-1.446 + 1.120*M - 0.950*np.log(np.maximum(Rh, 0.1))
              + 0.025*np.log(np.minimum(vs30, 1130)/1130)
              + self._site_term(vs30, c=0.52))
        ln += np.where(dip < 70, 0.08, 0.0)
        return ln


_ANALYTICAL_POOL: Dict[str, _AnalyticalGMPE] = {
    'ASK14': _ASK14_Approx(), 'BSSA14': _BSSA14_Approx(),
    'CB14':  _CB14_Approx(),  'CY14':   _CY14_Approx(),
}
_OQ_POOL: Dict[str, type] = {}
if _OQ_AVAILABLE:
    for _name, _cls in [('ASK14',  AbrahamsonEtAl2014),
                        ('BSSA14', BooreEtAl2014),
                        ('CB14',   CampbellBozorgnia2014),
                        ('CY14',   ChiouYoungs2014)]:
        if _cls: _OQ_POOL[_name] = _cls

_GMPE_POOL: Dict[str, object] = _OQ_POOL if _OQ_POOL else _ANALYTICAL_POOL


class GMPECalculator:
    def __init__(self, gmpe_name: Optional[str] = None):
        if gmpe_name is None:
            gmpe_name = str(np.random.choice(list(_GMPE_POOL.keys())))
        self.name = gmpe_name
        if _OQ_AVAILABLE and gmpe_name in _OQ_POOL:
            self._oq_gmpe = _OQ_POOL[gmpe_name](); self._analytical = None
        else:
            self._oq_gmpe = None; self._analytical = _ANALYTICAL_POOL[gmpe_name]

    def compute(self, magnitude, rrup, vs30,
                rake=0.0, dip=90.0, ztor=0.0, width=15.0) -> Dict:
        if self._analytical is not None:
            return self._analytical.compute(magnitude, rrup, vs30,
                                            rake, dip, ztor, width)
        return self._compute_oq(magnitude, rrup, vs30, rake, dip, ztor, width)

    def _compute_oq(self, magnitude, rrup, vs30, rake, dip, ztor, width):
        """
        构建 OpenQuake RuptureContext，正确区分三种距离度量：

        _rrup_from_segments 在 2D 水平面内计算最短距离，物理上对应：
            rjb（Joyner-Boore 距离）= 场点到断层面地表投影的水平最短距离

        rrup（3D 最短距离）和 rhypo（震源距）需从 rjb 和深度参数推导：

            rrup  = sqrt(rjb² + ztor²)         （断层顶端埋深作为有效深度）
            rhypo = sqrt(rjb² + hypo_depth²)   （震源深度）

        几何示意（剖面图）：
                 站点
                  |
            rjb ──┤──────── 断层地表投影
                  |  ↗ rrup
             ztor |↗
                  ●──── 断层面顶端
                   ↘
                    断层面（倾角 dip）

        注：rx（断层法向水平分量）维持为 0，保守估计悬壁效应。
            对于倾斜断层近场，此近似会低估悬壁侧的地震动放大。
        """
        n          = len(rrup)
        ztor_f     = float(ztor)
        hypo_depth = float(ztor + width * np.sin(np.radians(dip)) / 2)

        # rjb：_rrup_from_segments 计算的水平最短距离
        rjb = rrup.astype(float)

        # rrup_3d：考虑断层顶端埋深的 3D 最短距离
        # 下界为 ztor（站点在断层正上方时的极限值）
        rrup_3d = np.maximum(np.sqrt(rjb**2 + ztor_f**2), ztor_f)

        # rhypo：震源距
        rhypo = np.sqrt(rjb**2 + hypo_depth**2)

        ctx = RuptureContext()
        ctx.mag        = float(magnitude)
        ctx.rake       = float(rake)
        ctx.dip        = float(dip)
        ctx.ztor       = ztor_f
        ctx.width      = float(width)
        ctx.hypo_depth = hypo_depth
        ctx.rrup       = rrup_3d
        ctx.rjb        = rjb
        ctx.rhypo      = rhypo
        ctx.rx         = np.zeros(n)
        ctx.ry0        = np.zeros(n)
        ctx.vs30       = vs30.astype(float)
        ctx.z1pt0      = np.full(n, 50.0)
        ctx.z2pt5      = np.full(n, 1.0)
        ctx.sids       = np.arange(n)
        ctx.vs30measured = np.zeros(n, dtype=bool)
        try:
            def _g(im_type):
                ln, s = self._oq_gmpe.get_mean_and_stddevs(
                    ctx, ctx, ctx, im_type, [const.StdDev.TOTAL])
                return np.exp(ln), s[0]
            return {'PGA': _g(imt.PGA()), 'PGV': _g(imt.PGV()),
                    'SA_0.3': _g(imt.SA(0.3))}
        except Exception:
            return _ANALYTICAL_POOL[self.name].compute(
                magnitude, rrup, vs30, rake, dip, ztor, width)

    @staticmethod
    def pga_to_intensity(pga_g: float) -> float:
        """
        PGA → MMI，依据 Wald et al. (1999) 双折线公式。
        PGA 单位：g；内部转换为 gal（cm/s²）后代入。

        两段回归（Wald 1999, Earthquake Spectra 15(3):557-564）：
          高烈度段（MMI ≥ V，PGA ≥ ~66 gal）：
              MMI = 3.66 · log10(PGA_gal) − 1.66
          低烈度段（MMI < V，PGA < ~66 gal）：
              MMI = 2.20 · log10(PGA_gal) + 1.00
        两段交汇点：PGA ≈ 66 gal（约 0.067 g），对应 MMI ≈ V。
        """
        gal = pga_g * 980.0
        if gal < 1.0:
            return 1.0
        log_gal  = np.log10(gal)
        imm_hi   = 3.66 * log_gal - 1.66   # 高烈度段
        imm_lo   = 2.20 * log_gal + 1.00   # 低烈度段
        # 交汇点约 66.4 gal（两式联立解得）
        imm      = imm_lo if gal < 66.4 else imm_hi
        return float(np.clip(imm, 1.0, 12.0))

    @staticmethod
    def pgv_to_intensity(pgv_cms: float) -> float:
        """
        PGV → MMI，依据 Wald et al. (1999)。
        PGV 单位：cm/s。

        回归式（Wald 1999，适用范围 V ≤ MMI ≤ IX）：
            MMI = 3.47 · log10(PGV) + 2.35
        """
        if pgv_cms < 0.1:
            return 1.0
        return float(np.clip(3.47*np.log10(pgv_cms) + 2.35, 1.0, 12.0))

    @staticmethod
    def combined_intensity(pga_g: float, pgv_cms: float) -> float:
        """
        ShakeMap 风格的综合烈度，依据 Wald et al. (1999) 推荐策略：

          MMI < VI   → 仅使用 PGA 烈度
                        （低烈度由高频加速度主导，PGA 与人体感知相关性更强）
          VI ≤ MMI ≤ VII → 线性过渡混合
                        （过渡区，两者均有贡献）
          MMI > VII  → 仅使用 PGV 烈度
                        （高烈度由结构变形主导，PGV 与建筑破坏相关性更强）

        过渡区权重：w_pgv = (MMI_pga − 6) / 1，在 [VI, VII] 上从 0 线性增至 1。

        参考：Wald D.J. et al. (1999), Earthquake Spectra, 15(3), 557–564.
        """
        imm_pga = GMPECalculator.pga_to_intensity(pga_g)
        imm_pgv = GMPECalculator.pgv_to_intensity(pgv_cms)

        if imm_pga < 6.0:
            # 低烈度段：PGA 主导
            return imm_pga
        elif imm_pga > 7.0:
            # 高烈度段：PGV 主导
            return imm_pgv
        else:
            # 过渡段 [VI, VII]：线性混合
            w_pgv = (imm_pga - 6.0) / 1.0   # 0 → 1
            return float((1.0 - w_pgv) * imm_pga + w_pgv * imm_pgv)


# ─────────────────────────── Magnitude evolution ─────────────────────────────

class MagnitudeEvolution:
    @staticmethod
    def get(t: float, cfg: SimulationConfig) -> float:
        m0, mf, T = cfg.initial_magnitude, cfg.magnitude, cfg.total_duration
        if t <= 0: return m0
        if t >= T: return mf
        return m0 + (mf - m0) * MagnitudeEvolution._norm_curve(t / T, cfg)

    @staticmethod
    def _norm_curve(x: float, cfg: SimulationConfig) -> float:
        ev = cfg.magnitude_evolution_type
        if ev == 'sqrt':
            return float(np.sqrt(x))
        elif ev == 'linear':
            return float(x)
        elif ev == 'exponential':
            raw = 1.0 - np.exp(-3.0 * x)
            return float(raw / (1.0 - np.exp(-3.0)))
        elif ev == 'logistic':
            k = 10.0; mid = 0.5
            def sig(v): return 1.0 / (1.0 + np.exp(-k * (v - mid)))
            s0, s1, sv = sig(0.0), sig(1.0), sig(x)
            return float((sv - s0) / (s1 - s0))
        else:
            raise ValueError(f"Unknown evolution: {ev}")


# ─────────────────────────── Main simulator ──────────────────────────────────

class ShallowEarthquakeSimulator:

    def __init__(self, cfg: SimulationConfig):
        self.cfg    = cfg
        self.gmpe   = GMPECalculator(cfg.gmpe_name)
        self.res_km = cfg.resolution * 111.0
        self.stations                   = None
        self.station_vs30               = None
        self.grid_vs30                  = None
        self.station_wave_arrival_times = None
        self.station_trigger_times      = None
        self.forward_segments           = []
        self.backward_segments          = []
        self.magnitude_history          = []
        self._strike                    = None

    def _compute_delta_t(self) -> float:
        """
        Estimate the S-wave propagation delay Dt such that:
            field(t) is dominated by rupture at (t - Dt).

        The PGA field is dominated by the NEAREST stations (smallest rrup
        -> largest GMPE amplitude -> largest Gaussian weight). Using the
        mean station distance gave Dt ~18s, far too large.

        We use the near-field representative distance instead:
          - near-field planar distance ~= 0.4 * sigma_a  (~10th-pct magnitude)
          - lower-bounded by depth / v_s (station directly above hypocenter)
        """
        cfg = self.cfg
        gs  = cfg.grid_size

        sigma_a      = gs * 0.3
        near_d_grids = 0.4 * sigma_a
        near_d_km    = near_d_grids * cfg.resolution * 111.0
        near_hypo_km = np.sqrt(near_d_km**2 + cfg.depth**2)
        delta_t_near = near_hypo_km / cfg.s_wave_velocity

        delta_t_min  = cfg.depth / cfg.s_wave_velocity

        return float(max(delta_t_near, delta_t_min))

    def _generate_stations(self, strike, center):
        gs, n = self.cfg.grid_size, self.cfg.n_stations
        sta   = []
        for _ in range(n):
            a = np.random.normal(0, gs*0.3)
            p = np.random.uniform(-gs*0.4, gs*0.4)
            x = center[0] + a*np.cos(strike) - p*np.sin(strike)
            y = center[1] + a*np.sin(strike) + p*np.cos(strike)
            sta.append([float(np.clip(x, 2, gs-2)),
                        float(np.clip(y, 2, gs-2))])
        return np.array(sta)

    def _generate_vs30_field(self):
        from scipy.interpolate import RectBivariateSpline
        gs  = self.cfg.grid_size; cs = 10
        sig = self.cfg.vs30_std / self.cfg.vs30_mean
        c   = np.clip(np.random.lognormal(np.log(self.cfg.vs30_mean),
                                           sig, size=(cs, cs)), 150, 1000)
        xc   = np.linspace(0, gs-1, cs)
        fine = RectBivariateSpline(xc, xc, c)(np.arange(gs), np.arange(gs))
        noise = np.random.normal(0, self.cfg.vs30_std*0.1, fine.shape)
        return np.clip(fine * np.exp(noise / self.cfg.vs30_mean), 150, 1000)

    def _arrival_times_from_center(self, stations, center):
        times = []
        for s in stations:
            d_km  = np.linalg.norm(s - center) * self.res_km
            hypo  = np.sqrt(d_km**2 + self.cfg.depth**2)
            times.append(hypo / self.cfg.s_wave_velocity)
        return np.array(times)

    def _rupture_dimensions(self, magnitude):
        """
        按 Wells & Coppersmith (1994) Table 2A 的分断层类型回归式
        分别计算地下破裂长度（RLD）和断层宽度（RW）。

        断层类型由滑动角（rake）判定，遵循标准分类：
          走滑（SS）：|rake| ≤ 45° 或 |rake| ≥ 135°
          逆断层（R）：45° < rake ≤ 135°
          正断层（N）：-135° ≤ rake < -45°
          斜滑过渡区（±45°边界）：使用全类型（All）系数

        地下破裂长度 RLD，回归式：log10(RLD) = a + b * M
          SS :  a = -2.57, b = 0.62, σ = 0.15, r = 0.96
          R  :  a = -2.42, b = 0.58, σ = 0.16, r = 0.93
          N  :  a = -1.88, b = 0.50, σ = 0.17, r = 0.88
          All:  a = -2.44, b = 0.59, σ = 0.16, r = 0.94

        断层宽度 RW，回归式：log10(RW) = a + b * M
          SS :  a = -0.76, b = 0.27, σ = 0.14, r = 0.84
          R  :  a = -1.61, b = 0.41, σ = 0.15, r = 0.90
          N  :  a = -1.14, b = 0.35, σ = 0.12, r = 0.86
          All:  a = -1.01, b = 0.32, σ = 0.15, r = 0.84

        宽度上限 20 km：反映地壳发震层厚度对断层宽度的物理约束。

        参考：Wells, D.L. & Coppersmith, K.J. (1994).
              Bull. Seismol. Soc. Am., 84(4), 974–1002, Table 2A.
        """
        rake = self.cfg.rake
        abs_rake = abs(rake)

        # ── 断层类型判定 ──────────────────────────────────────────────
        # 严格使用开区间，±45° 边界归入斜滑（All），与 Wells & Coppersmith
        # 原文分类方案一致（HZ:VT=1:1 为走滑与倾滑的过渡区）
        if abs_rake < 45.0 or abs_rake > 135.0:
            # 走滑（Strike Slip）：|rake| < 45° 或 |rake| > 135°
            a_len, b_len = -2.57, 0.62
            a_wid, b_wid = -0.76, 0.27
        elif 45.0 < rake < 135.0:
            # 逆断层（Reverse）：45° < rake < 135°
            a_len, b_len = -2.42, 0.58
            a_wid, b_wid = -1.61, 0.41
        elif -135.0 < rake < -45.0:
            # 正断层（Normal）：-135° < rake < -45°
            a_len, b_len = -1.88, 0.50
            a_wid, b_wid = -1.14, 0.35
        else:
            # 斜滑过渡（rake ≈ ±45° 或 ≈ ±135°）：使用全类型（All）回归
            a_len, b_len = -2.44, 0.59
            a_wid, b_wid = -1.01, 0.32

        length_km = 10 ** (a_len + b_len * magnitude)
        width_km  = min(10 ** (a_wid + b_wid * magnitude), 20.0)
        return length_km, width_km

    def _build_segments(self, start_pos, base_strike, length_km, n_segs):
        seg_len_grids = (length_km / n_segs) / self.res_km
        max_dk        = np.radians(self.cfg.max_strike_change_deg)
        segs          = []
        pos           = start_pos.astype(float).copy()
        cur_strike    = float(base_strike)
        cum           = 0.0
        for i in range(n_segs):
            if i > 0:
                prev_dk = cur_strike - segs[-1]['strike']
                lo = max(-max_dk, -max_dk - prev_dk)
                hi = min( max_dk,  max_dk - prev_dk)
                cur_strike += float(np.random.uniform(lo, hi))
            end = pos + seg_len_grids * np.array([np.cos(cur_strike),
                                                   np.sin(cur_strike)])
            segs.append({'start': pos.copy(), 'end': end.copy(),
                         'strike': cur_strike,
                         'length_grids': seg_len_grids,
                         'cum_start': cum, 'cum_end': cum + seg_len_grids})
            cum += seg_len_grids; pos = end.copy()
        return segs

    def generate_fault_geometry(self, initial_strike, total_length_km):
        bilateral = 'bilateral' in self.cfg.fault_type
        polyline  = 'polyline'  in self.cfg.fault_type
        n_segs    = self.cfg.n_fault_segments if polyline else 1
        center    = np.array([self.cfg.grid_size/2.0, self.cfg.grid_size/2.0])
        if bilateral:
            ratio = self.cfg.bilateral_ratio
            if np.random.rand() > 0.5: ratio = 1.0 - ratio
            n_fwd = max(1, n_segs // 2); n_bwd = max(1, n_segs - n_fwd)
            fwd   = self._build_segments(center, initial_strike,
                                         total_length_km*ratio, n_fwd)
            bwd   = self._build_segments(center, initial_strike + np.pi,
                                         total_length_km*(1-ratio), n_bwd)
            return fwd, bwd
        return self._build_segments(center, initial_strike,
                                    total_length_km, n_segs), []

    def _rrup_from_segments(self, stations, fwd, bwd,
                             front_fwd, front_bwd, width_grids):
        rrup = np.full(len(stations), np.inf)
        for segs, front in [(fwd, front_fwd), (bwd, front_bwd)]:
            if not segs or front <= 0: continue
            for seg in segs:
                if seg['cum_start'] >= front: break
                frac = min(1.0, (front - seg['cum_start']) / seg['length_grids'])
                aend = seg['start'] + frac*(seg['end'] - seg['start'])
                vec  = aend - seg['start']
                slen = np.linalg.norm(vec)
                if slen < 1e-10: continue
                sd = vec / slen
                for i, sta in enumerate(stations):
                    dp    = sta - seg['start']
                    along = float(np.dot(dp, sd))
                    perp  = abs(float(dp[0]*(-sd[1]) + dp[1]*sd[0]))
                    da    = max(0.0, max(along - slen, -along))
                    dp_   = max(0.0, perp - width_grids/2)
                    rrup[i] = min(rrup[i], np.sqrt(da**2 + dp_**2))
        rrup[rrup == np.inf] = 1.0
        return rrup * self.res_km   # → km

    def _rupture_grid_from_segments(self, fwd, bwd,
                                     front_fwd, front_bwd, width_grids):
        gs   = self.cfg.grid_size
        Y, X = np.meshgrid(np.arange(gs), np.arange(gs), indexing='ij')
        pts  = np.stack([X.ravel(), Y.ravel()], axis=1).astype(float)
        mask = np.zeros(len(pts), dtype=bool)
        for segs, front in [(fwd, front_fwd), (bwd, front_bwd)]:
            if not segs or front <= 0: continue
            for seg in segs:
                if seg['cum_start'] >= front: break
                frac = min(1.0, (front - seg['cum_start']) / seg['length_grids'])
                aend = seg['start'] + frac*(seg['end'] - seg['start'])
                vec  = aend - seg['start']
                slen = np.linalg.norm(vec)
                if slen < 1e-10: continue
                sd    = vec / slen
                dp    = pts - seg['start']
                along = dp @ sd
                perp  = np.abs(dp[:,0]*(-sd[1]) + dp[:,1]*sd[0])
                mask |= (along >= 0) & (along <= slen) & (perp <= width_grids/2)
        return mask.reshape(gs, gs).astype(np.float32)

    def _gauss_interp(self, positions, values, min_value=0.0,
                      sigma=15.0, cutoff=2.5) -> np.ndarray:
        valid = values > min_value
        if np.sum(valid) < 3:
            return np.zeros((self.cfg.grid_size, self.cfg.grid_size), np.float32)
        vp, vv = positions[valid], values[valid]
        gs     = self.cfg.grid_size
        field  = np.zeros((gs, gs), np.float32)
        for i in range(gs):
            for j in range(gs):
                pt     = np.array([j, i], dtype=float)
                d2     = np.sum((vp - pt)**2, axis=1)
                within = d2 < (cutoff * sigma)**2
                if not np.any(within): continue
                w = np.exp(-d2[within] / (2 * sigma**2))
                w /= w.sum()
                field[i, j] = float(np.dot(w, vv[within]))
        return np.maximum(field, 0)

    def generate_synthetic_earthquake(self) -> List[Dict]:
        cfg       = self.cfg
        bilateral = 'bilateral' in cfg.fault_type

        final_len_km, final_wid_km = self._rupture_dimensions(cfg.magnitude)
        final_len_grids = final_len_km / self.res_km
        final_wid_grids = final_wid_km / self.res_km
        t_complete = final_len_km / cfg.rupture_velocity

        # S-wave propagation delay: field(t) reflects rupture at (t - delta_t)
        delta_t = self._compute_delta_t()

        # Pre-compute rupture grids at every time step for label look-up
        # (needed before Phase 2 so label_idx can index into this list)
        times_pre = np.linspace(0, cfg.total_duration, cfg.n_time_steps)
        all_rupture_grids = []
        # geometry not yet built — will be populated after segments are created

        strike = (cfg.strike if cfg.strike is not None
                  else float(np.random.uniform(0, 2*np.pi)))
        self._strike = strike

        center = np.array([cfg.grid_size/2.0, cfg.grid_size/2.0])
        self.forward_segments, self.backward_segments = \
            self.generate_fault_geometry(strike, final_len_km)

        # Now segments exist: pre-compute rupture grids
        bilateral_pre = 'bilateral' in cfg.fault_type
        for tau_pre in times_pre:
            fp = min(cfg.rupture_velocity * tau_pre / self.res_km, final_len_grids)
            fb = fp if bilateral_pre else 0.0
            all_rupture_grids.append(self._rupture_grid_from_segments(
                self.forward_segments, self.backward_segments,
                fp, fb, final_wid_grids))

        self.stations     = self._generate_stations(strike, center)
        self.station_vs30 = np.array([
            float(np.clip(np.random.lognormal(
                np.log(cfg.vs30_mean),
                cfg.vs30_std/cfg.vs30_mean), 150, 1000))
            for _ in range(cfg.n_stations)])
        self.grid_vs30  = self._generate_vs30_field()
        self.station_wave_arrival_times = \
            self._arrival_times_from_center(self.stations, center)
        self.station_trigger_times = np.full(cfg.n_stations, np.inf)

        times = np.linspace(0, cfg.total_duration, cfg.n_time_steps)

        # ── Phase 1: history loop ─────────────────────────────────────────────
        history = []
        for tau in times:
            mag_tau = MagnitudeEvolution.get(tau, cfg)

            cur_len_km, cur_wid_km = self._rupture_dimensions(mag_tau)
            cur_wid_km    = min(cur_wid_km, final_wid_km)
            cur_wid_grids = cur_wid_km / self.res_km
            ztor_tau      = max(0.0, cfg.depth
                                - cur_wid_km*np.sin(np.radians(cfg.dip))/2)

            front_fwd = min(cfg.rupture_velocity * tau / self.res_km,
                            final_len_grids)
            front_bwd = front_fwd if bilateral else 0.0

            rrup_sta = self._rrup_from_segments(
                self.stations,
                self.forward_segments, self.backward_segments,
                front_fwd, front_bwd, cur_wid_grids)

            gm = self.gmpe.compute(
                mag_tau, rrup_sta, self.station_vs30,
                cfg.rake, cfg.dip, ztor_tau, cur_wid_km)

            min_rrup_epi = max(cfg.depth, cur_wid_km / 2)
            gm_epi       = self.gmpe.compute(
                mag_tau,
                np.array([min_rrup_epi]),
                np.array([cfg.vs30_mean]),
                cfg.rake, cfg.dip, ztor_tau, cur_wid_km)

            history.append({
                'tau':              tau,
                'pga_g':            gm['PGA'][0],
                'pgv_cms':          gm['PGV'][0],
                'arrival_times':    tau + rrup_sta / cfg.s_wave_velocity,
                'pga_g_epi':        float(gm_epi['PGA'][0][0]),
                'pgv_cms_epi':      float(gm_epi['PGV'][0][0]),
                'arrival_time_epi': tau + min_rrup_epi / cfg.s_wave_velocity,

            })

        # ── Phase 2: observable sequence ─────────────────────────────────────
        eq_sequence = []
        for k, t in enumerate(times):
            obs_pga_g   = np.zeros(cfg.n_stations)
            obs_pgv_cms = np.zeros(cfg.n_stations)

            for i in range(cfg.n_stations):
                arr_pga = [s['pga_g'][i]  for s in history
                           if s['arrival_times'][i] <= t]
                arr_pgv = [s['pgv_cms'][i] for s in history
                           if s['arrival_times'][i] <= t]
                if arr_pga:
                    obs_pga_g[i]   = float(np.max(arr_pga))
                    obs_pgv_cms[i] = float(np.max(arr_pgv))

            arr_epi_pga = [s['pga_g_epi']   for s in history
                           if s['arrival_time_epi'] <= t]
            arr_epi_pgv = [s['pgv_cms_epi']  for s in history
                           if s['arrival_time_epi'] <= t]

            obs_epi_pga_g   = 0.0
            obs_epi_pgv_cms = 0.0
            if arr_epi_pga:
                obs_epi_pga_g   = max(float(np.max(arr_epi_pga)),
                                      float(np.max(obs_pga_g)) * 0.8)
                obs_epi_pgv_cms = max(float(np.max(arr_epi_pgv)),
                                      float(np.max(obs_pgv_cms)) * 0.8)

            obs_pga_gal     = obs_pga_g     * 980.0
            obs_pga_gal_epi = obs_epi_pga_g * 980.0

            aug_sta   = np.vstack([self.stations, center])
            pga_field = self._gauss_interp(
                aug_sta, np.append(obs_pga_gal, obs_pga_gal_epi), min_value=1.0)
            pgv_field = self._gauss_interp(
                aug_sta, np.append(obs_pgv_cms, obs_epi_pgv_cms), min_value=0.1)

            intensity = np.zeros(cfg.n_stations)
            for i in range(cfg.n_stations):
                if obs_pga_gal[i] > 1.0:
                    # ShakeMap 风格：低烈度用 PGA，高烈度用 PGV，过渡段线性混合
                    intensity[i] = GMPECalculator.combined_intensity(
                        obs_pga_g[i], obs_pgv_cms[i])
            int_epi = 1.0
            if obs_pga_gal_epi > 1.0:
                int_epi = GMPECalculator.combined_intensity(
                    obs_epi_pga_g, obs_epi_pgv_cms)
            int_field = self._gauss_interp(
                aug_sta, np.append(intensity, int_epi), min_value=1.0)

            # ── Rupture grid label: aligned to t - delta_t ───────────────────
            # field(t) reflects rupture at (t - delta_t).
            # Label = rupture_grid(t - delta_t) ensures each unique field
            # maps to a unique label within the valid frame range.
            #
            # Valid frame condition: t <= t_complete + delta_t
            #   - Before t_complete: rupture still growing, label changes each frame
            #   - [t_complete, t_complete+delta_t]: last new signal still arriving
            #   - After t_complete+delta_t: field still changes but label frozen
            #     → ambiguous, excluded from training (valid_frame=0)
            t_label   = max(0.0, t - delta_t)
            label_idx = int(np.argmin(np.abs(times - t_label)))
            front_fwd_lbl = min(cfg.rupture_velocity * times[label_idx] / self.res_km,
                                final_len_grids)
            front_bwd_lbl = front_fwd_lbl if bilateral else 0.0
            rupture_grid  = all_rupture_grids[label_idx]

            # Current front at t (for visualisation only, not used as label)
            front_fwd_t = min(cfg.rupture_velocity * t / self.res_km,
                              final_len_grids)

            cur_mag = MagnitudeEvolution.get(t, cfg)
            self.magnitude_history.append(cur_mag)

            rup_progress  = float(min(t_label / t_complete, 1.0))                             if t_complete > 0 else 1.0
            rup_complete  = float(t_label >= t_complete)
            # valid_frame: 1 if field still carries new label information
            valid_frame   = float(delta_t <= t <= t_complete + delta_t)

            eq_sequence.append({
                'time':               float(t),
                'magnitude':          float(cur_mag),
                'rupture_grid':       rupture_grid,       # label: t - delta_t
                'rupture_progress':   rup_progress,
                'rupture_complete':   rup_complete,
                'valid_frame':        valid_frame,
                'pga_field':          pga_field,
                'pgv_field':          pgv_field,
                'intensity_field':    int_field,
                'n_active_stations':  int(np.sum(obs_pga_gal > 5.0)),
                'pga_observed':       obs_pga_gal.astype(np.float32),
                'pgv_observed':       obs_pgv_cms.astype(np.float32),
                'intensity_observed': intensity.astype(np.float32),
                'metadata': {
                    'rupture_center': center.tolist(),
                    'fault_type':     cfg.fault_type,
                    'front_fwd':      float(front_fwd_t),
                    't_complete':     float(t_complete),
                    'delta_t':        float(delta_t),
                    't_label':        float(times[label_idx]),
                },
            })

        return eq_sequence