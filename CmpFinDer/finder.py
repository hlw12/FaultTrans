"""Independent, paper-derived FinDer v2 binary template matcher.

References: Bose et al. (2012), GJI 191; Bose et al. (2018), GJI 212.
This is not the proprietary ETH/SED FinDer software.

Grid convention used by this project:
    row index increases with latitude (north-up),
    col index increases with longitude (east-right).
Strike 0° = north, 90° = east, range [0, 180).

Aligned with Bose et al. (2018) where practical:
    - Cua–Heaton eq. (1) with 7.2e-4 and -0.42+log10(1.1)
    - Wells–Coppersmith length (eq. 2)
    - Paper PGA thresholds and M2.5–M8.0 / 0.1 templates
    - Normalized SQDIFF-style misfit (OpenCV / eq. 4 family)
    - Divide-and-conquer strike search (Fig. 2)
    - Template pixel size ~ min(145, 30+70*log10(L+1))
    - Growth-only PGA threshold / magnitude
Site corrections are intentionally not implemented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.signal import fftconvolve


PAPER_PGA_THRESHOLDS = (2.0, 4.6, 10.5, 23.2, 48.6, 90.7, 148.8, 221.3, 304.5)


def rupture_length_km(magnitude: float) -> float:
    """Wells & Coppersmith (1994) strike-slip relation used by FinDer v2."""
    return float(10.0 ** ((float(magnitude) - 4.33) / 1.49))


def magnitude_from_length(length_km: float) -> float:
    """Inverse Wells & Coppersmith relation."""
    length_km = max(float(length_km), 1e-6)
    return float(4.33 + 1.49 * np.log10(length_km))


def cua_heaton_pga(magnitude: float, distance_km: np.ndarray) -> np.ndarray:
    """Cua–Heaton (2009) rock-site PGA [cm/s^2] as in Bose et al. (2018) eq. (1)."""
    m = float(magnitude)
    r = np.asarray(distance_km, dtype=np.float64)
    c_m = 1.16 * np.exp(0.96 * (m - 5.0)) * (np.arctan(m - 5.0) + 0.5 * np.pi)
    r_eff = np.sqrt(r ** 2 + 9.0) + c_m
    log_pga = (
        0.73 * m
        - 7.2e-4 * r_eff
        - 1.48 * np.log10(r_eff)
        - 0.42
        + np.log10(1.1)
    )
    return np.power(10.0, log_pga).astype(np.float32)


def pga_radius_km(
    magnitude: float,
    threshold_cm_s2: float,
    r_max_km: float = 500.0,
    n: int = 2048,
) -> float:
    """Largest distance where theoretical PGA still exceeds the threshold."""
    rs = np.linspace(0.0, float(r_max_km), int(n))
    pga = cua_heaton_pga(magnitude, rs)
    ok = np.where(pga >= float(threshold_cm_s2))[0]
    if len(ok) == 0:
        return 0.0
    return float(rs[ok[-1]])


def point_to_segment_distance_km(
    xs: np.ndarray,
    ys: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> np.ndarray:
    """Distance from points to a finite line segment, in km."""
    dx = x1 - x0
    dy = y1 - y0
    seg2 = dx * dx + dy * dy
    if seg2 < 1e-12:
        return np.sqrt((xs - x0) ** 2 + (ys - y0) ** 2)

    t = ((xs - x0) * dx + (ys - y0) * dy) / seg2
    t = np.clip(t, 0.0, 1.0)
    px = x0 + t * dx
    py = y0 + t * dy
    return np.sqrt((xs - px) ** 2 + (ys - py) ** 2)


def paper_template_size_px(length_km: float, pixel_size_km: float = 1.11) -> int:
    """Bose et al. (2018) template size in pixels: min(145, 30+70*log10(L+1)).

    The paper states this on a ~5 km grid. On our finer grid (~1.11 km),
    converting that formula to the same *physical* km would make templates
    larger than a 150x150 domain, so we keep the paper *pixel* formula on
    the working grid (and still enforce a content minimum of L+radius).
    `pixel_size_km` is accepted for API symmetry but not used for scaling.
    """
    _ = pixel_size_km
    size_px = min(145, int(round(30.0 + 70.0 * np.log10(float(length_km) + 1.0))))
    size_px = max(size_px, 3)
    if size_px % 2 == 0:
        size_px += 1
    return size_px


@dataclass
class FinDerConfig:
    pixel_size_km: float = 1.11
    magnitudes: Sequence[float] = field(
        default_factory=lambda: tuple(np.round(np.arange(2.5, 8.01, 0.1), 1))
    )
    thresholds_cm_s2: Sequence[float] = field(
        default_factory=lambda: PAPER_PGA_THRESHOLDS
    )
    # Fig. 2: divide-and-conquer until this angular resolution
    strike_resolution_deg: float = 5.0
    strike_dac_samples: int = 5
    min_active_pixels: int = 10
    growth_only: bool = True
    pga_unit: str = "cm_s2"  # "cm_s2" | "g"
    template_padding_px: int = 2
    # If True, use paper physical template size; content (L+radius) still enforced as minimum
    use_paper_template_size: bool = True


@dataclass
class FinDerResult:
    found: bool
    strike_deg: float = float("nan")
    length_km: float = float("nan")
    magnitude: float = float("nan")
    threshold_cm_s2: float = float("nan")
    center_row: float = float("nan")
    center_col: float = float("nan")
    misfit: float = float("nan")
    correlation: float = float("nan")
    n_obs_pixels: int = 0


def _to_cm_s2(pga: np.ndarray, unit: str) -> np.ndarray:
    x = np.asarray(pga, dtype=np.float32)
    if unit == "cm_s2":
        return x
    if unit == "g":
        return x * 980.665
    raise ValueError(f"Unsupported pga_unit={unit}")


def _line_endpoints_px(
    length_km: float,
    strike_deg: float,
    pixel_size_km: float,
) -> tuple[float, float, float, float]:
    half = 0.5 * float(length_km) / max(float(pixel_size_km), 1e-6)
    theta = np.deg2rad(float(strike_deg) % 180.0)
    # north-up: +y = north, +x = east
    dx = half * np.sin(theta)
    dy = half * np.cos(theta)
    return -dx, -dy, dx, dy


def build_binary_template(
    magnitude: float,
    strike_deg: float,
    threshold_cm_s2: float,
    pixel_size_km: float,
    padding_px: int = 2,
    use_paper_template_size: bool = True,
) -> np.ndarray:
    """Capsule-shaped binary PGA-exceedance template for one (M, θ, thr)."""
    length_km = rupture_length_km(magnitude)
    radius_km = pga_radius_km(magnitude, threshold_cm_s2)
    half_len_px = 0.5 * length_km / pixel_size_km
    rad_px = radius_km / pixel_size_km
    content_half = int(np.ceil(half_len_px + rad_px + padding_px))
    content_size = max(2 * content_half + 1, 3)

    if use_paper_template_size:
        paper_size = paper_template_size_px(length_km, pixel_size_km)
        size = max(content_size, paper_size)
    else:
        size = content_size
    if size % 2 == 0:
        size += 1

    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = 0.5 * (size - 1)
    xs = (xx - cx) * pixel_size_km
    ys = (yy - cy) * pixel_size_km

    x0, y0, x1, y1 = _line_endpoints_px(length_km, strike_deg, pixel_size_km)
    x0_km, y0_km = x0 * pixel_size_km, y0 * pixel_size_km
    x1_km, y1_km = x1 * pixel_size_km, y1 * pixel_size_km

    if magnitude < 5.0:
        dist = np.sqrt(xs ** 2 + ys ** 2)
    else:
        dist = point_to_segment_distance_km(xs, ys, x0_km, y0_km, x1_km, y1_km)

    pga = cua_heaton_pga(magnitude, dist)
    return (pga >= float(threshold_cm_s2)).astype(np.float32)


def _normalized_misfit(obs: np.ndarray, tmpl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slide template; return normalized misfit map and a correlation-like score.

    Misfit follows the OpenCV TM_SQDIFF_NORMED / paper eq. (4) family:
        sum((T-I)^2) / sqrt(sum(T^2) * sum(I^2))
    For binary maps: sum((T-I)^2) = n_t + n_o - 2*overlap.
    """
    obs = obs.astype(np.float32)
    tmpl = tmpl.astype(np.float32)
    ones_t = np.ones_like(tmpl)

    n_t = float(tmpl.sum())
    n_o_map = fftconvolve(obs, ones_t, mode="valid")
    overlap = fftconvolve(obs, tmpl, mode="valid")

    diff2 = n_t + n_o_map - 2.0 * overlap
    denom = np.sqrt(np.maximum(n_t * n_o_map, 0.0)) + 1e-6
    misfit = diff2 / denom
    # higher is better; useful for tie-breaks / logging
    corr = (2.0 * overlap) / (n_t + n_o_map + 1e-6)
    return misfit, corr


def _best_placement(obs: np.ndarray, tmpl: np.ndarray) -> tuple[float, float, float, float]:
    misfit, corr = _normalized_misfit(obs, tmpl)
    idx = np.unravel_index(int(np.argmin(misfit)), misfit.shape)
    best_m = float(misfit[idx])
    best_c = float(corr[idx])
    th, tw = tmpl.shape
    center_row = float(idx[0] + 0.5 * (th - 1))
    center_col = float(idx[1] + 0.5 * (tw - 1))
    return best_m, best_c, center_row, center_col


class FinDer:
    def __init__(self, config: Optional[FinDerConfig] = None):
        self.config = config or FinDerConfig()
        self._prev: Optional[FinDerResult] = None
        self._template_cache: dict[tuple, np.ndarray] = {}

    def reset(self) -> None:
        self._prev = None

    def _get_template(self, magnitude: float, strike_deg: float, thr: float) -> np.ndarray:
        key = (
            round(float(magnitude), 1),
            round(float(strike_deg) % 180.0, 2),
            round(float(thr), 3),
            round(float(self.config.pixel_size_km), 5),
            bool(self.config.use_paper_template_size),
        )
        if key not in self._template_cache:
            self._template_cache[key] = build_binary_template(
                magnitude=key[0],
                strike_deg=key[1],
                threshold_cm_s2=key[2],
                pixel_size_km=self.config.pixel_size_km,
                padding_px=self.config.template_padding_px,
                use_paper_template_size=self.config.use_paper_template_size,
            )
        return self._template_cache[key]

    def _candidate_magnitudes(self, n_obs: int, thr: float) -> list[float]:
        """Initial L/M from pixel-count match, then ±0.1 neighbours (paper Fig. 2)."""
        mags = [float(m) for m in self.config.magnitudes]
        if n_obs <= 0:
            return mags[:1]

        scores = []
        for m in mags:
            tmpl = self._get_template(m, 0.0, thr)
            scores.append((abs(float(tmpl.sum()) - n_obs), m))
        scores.sort()
        m0 = scores[0][1]
        mag_set = set(mags)
        neigh = sorted({round(m0 - 0.1, 1), round(m0, 1), round(m0 + 0.1, 1)})
        out = [m for m in neigh if m in mag_set]
        return out if out else [m0]

    def _eval_placement(
        self,
        obs: np.ndarray,
        mag: float,
        strike: float,
        thr: float,
    ) -> Optional[tuple[float, float, float, float, float, float]]:
        tmpl = self._get_template(mag, strike, thr)
        if tmpl.shape[0] > obs.shape[0] or tmpl.shape[1] > obs.shape[1]:
            return None
        if tmpl.sum() < 1:
            return None
        mfit, corr, cr, cc = _best_placement(obs, tmpl)
        return (mfit, -corr, mag, float(strike % 180.0), cr, cc)

    def _divide_and_conquer_strike(
        self,
        obs: np.ndarray,
        mag: float,
        thr: float,
    ) -> Optional[tuple[float, float, float, float, float, float]]:
        """Paper Fig. 2 strike search: sample, pick best interval, subdivide."""
        n_samp = max(int(self.config.strike_dac_samples), 3)
        resolution = float(self.config.strike_resolution_deg)

        # initial samples over [0, 180)
        strikes = list(np.linspace(0.0, 180.0, n_samp, endpoint=False))
        best = None
        for s in strikes:
            cand = self._eval_placement(obs, mag, s, thr)
            if cand is None:
                continue
            if best is None or cand[0] < best[0]:
                best = cand

        if best is None:
            return None

        span = 180.0 / float(n_samp)
        # subdivide around current best until angular resolution is reached
        while span > resolution + 1e-9:
            center = best[3]
            local = [
                (center + d) % 180.0
                for d in np.linspace(-span, span, n_samp)
            ]
            for s in local:
                cand = self._eval_placement(obs, mag, s, thr)
                if cand is None:
                    continue
                if cand[0] < best[0]:
                    best = cand
            span *= 0.5

        return best

    def _search_one_threshold(
        self,
        obs: np.ndarray,
        thr: float,
        mag_candidates: Iterable[float],
    ) -> FinDerResult:
        n_obs = int(obs.sum())
        best = FinDerResult(found=False, n_obs_pixels=n_obs)

        for mag in mag_candidates:
            dac = self._divide_and_conquer_strike(obs, mag, thr)
            if dac is None:
                continue
            mfit, _, mag_c, strike_c, cr, cc = dac
            # correlation proxy from final placement
            placed = self._eval_placement(obs, mag_c, strike_c, thr)
            corr = -placed[1] if placed is not None else float("nan")
            if (not best.found) or (mfit < best.misfit):
                best = FinDerResult(
                    found=True,
                    strike_deg=float(strike_c % 180.0),
                    length_km=rupture_length_km(mag_c),
                    magnitude=float(mag_c),
                    threshold_cm_s2=float(thr),
                    center_row=cr,
                    center_col=cc,
                    misfit=float(mfit),
                    correlation=float(corr),
                    n_obs_pixels=n_obs,
                )

        return best

    def fit(self, pga_field: np.ndarray) -> FinDerResult:
        pga = _to_cm_s2(pga_field, self.config.pga_unit)
        best = FinDerResult(found=False)

        thr_list = list(self.config.thresholds_cm_s2)
        if self.config.growth_only and self._prev is not None and self._prev.found:
            thr_list = [t for t in thr_list if t >= self._prev.threshold_cm_s2 - 1e-6]
            if not thr_list:
                thr_list = [self._prev.threshold_cm_s2]

        for thr in thr_list:
            obs = (pga >= float(thr)).astype(np.float32)
            n_obs = int(obs.sum())
            if n_obs < self.config.min_active_pixels:
                continue

            mags = self._candidate_magnitudes(n_obs, thr)
            if self.config.growth_only and self._prev is not None and self._prev.found:
                mags = [m for m in mags if m + 1e-6 >= self._prev.magnitude]
                if not mags:
                    mags = [self._prev.magnitude]

            cand = self._search_one_threshold(obs, thr, mags)
            if cand.found and ((not best.found) or cand.misfit < best.misfit):
                best = cand

        if self.config.growth_only and self._prev is not None and self._prev.found:
            if (not best.found) or (best.misfit > self._prev.misfit + 0.05):
                best = self._prev
            else:
                # non-decreasing length / magnitude / threshold (paper: event only grows)
                if best.length_km + 1e-6 < self._prev.length_km:
                    best.length_km = self._prev.length_km
                    best.magnitude = max(best.magnitude, self._prev.magnitude)
                if best.threshold_cm_s2 + 1e-6 < self._prev.threshold_cm_s2:
                    best.threshold_cm_s2 = self._prev.threshold_cm_s2

        self._prev = best if best.found else self._prev
        return best

    def fit_sequence(self, pga_fields: np.ndarray) -> list[FinDerResult]:
        self.reset()
        fields = np.asarray(pga_fields)
        if fields.ndim != 3:
            raise ValueError(f"Expected [T,H,W], got {fields.shape}")
        return [self.fit(fields[t]) for t in range(fields.shape[0])]


def finder_line_mask(
    result: FinDerResult,
    grid_size: int = 150,
    pixel_size_km: float = 1.11,
    half_width_px: float = 1.5,
) -> np.ndarray:
    """Rasterize FinDer line source to a thin binary mask for angle metrics."""
    mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    if not result.found or not np.isfinite(result.strike_deg):
        return mask
    if not np.isfinite(result.center_row) or not np.isfinite(result.center_col):
        return mask
    if not np.isfinite(result.length_km) or result.length_km <= 0:
        return mask

    half = 0.5 * float(result.length_km) / max(float(pixel_size_km), 1e-6)
    theta = np.deg2rad(float(result.strike_deg) % 180.0)
    dx = half * np.sin(theta)
    dy = half * np.cos(theta)

    cr, cc = float(result.center_row), float(result.center_col)
    n = max(int(np.ceil(2 * half * 4)), 8)
    rs = np.linspace(cr - dy, cr + dy, n)
    cs = np.linspace(cc - dx, cc + dx, n)

    yy, xx = np.mgrid[0:grid_size, 0:grid_size]
    for r, c in zip(rs, cs):
        mask[((yy - r) ** 2 + (xx - c) ** 2) <= (half_width_px ** 2)] = 1
    return mask
