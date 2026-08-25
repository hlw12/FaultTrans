import unittest

import numpy as np

from finder import (
    FinDer,
    FinDerConfig,
    cua_heaton_pga,
    magnitude_from_length,
    rupture_length_km,
)


class FinDerTests(unittest.TestCase):
    def test_length_magnitude_round_trip(self):
        for magnitude in (2.5, 5.0, 6.5, 8.0):
            self.assertAlmostEqual(
                magnitude_from_length(rupture_length_km(magnitude)),
                magnitude,
                places=10,
            )

    def test_pga_decreases_with_distance(self):
        pga = cua_heaton_pga(7.0, np.array([0.0, 10.0, 50.0, 100.0]))
        self.assertTrue(np.all(np.diff(pga) < 0.0))

    def test_recovers_synthetic_template(self):
        config = FinDerConfig(
            pixel_size_km=2.0,
            magnitudes=(6.4, 6.5, 6.6),
            pga_thresholds=(48.6,),
            min_active_pixels=3,
            final_strike_step_deg=2.0,
            pga_unit="cm_s2",
            growth_only=False,
        )
        finder = FinDer(config)
        template = finder._binary_template(6.5, 48.6, 40.0)
        image = np.zeros((151, 151), dtype=float)
        h, w = template.shape
        y0, x0 = 75 - h // 2, 75 - w // 2
        image[y0 : y0 + h, x0 : x0 + w] = template * 60.0

        result = finder.fit(image)
        self.assertLessEqual(abs(result.magnitude - 6.5), 0.1)
        strike_error = abs(result.strike_deg - 40.0)
        strike_error = min(strike_error, 180.0 - strike_error)
        self.assertLessEqual(strike_error, 4.0)
        self.assertLess(result.misfit, 0.1)


if __name__ == "__main__":
    unittest.main()
