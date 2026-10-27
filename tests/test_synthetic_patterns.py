"""Vérification des générateurs de motifs et de bruits colorés."""

import numpy as np
import pytest

from eoqual.measure.synthetic import pattern_rng, power_law_noise


def _spectral_slope_1d(x):
    p = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size)
    keep = (f > 0.002) & (f < 0.4)
    return np.polyfit(np.log(f[keep]), np.log(p[keep]), 1)[0]


@pytest.mark.parametrize("beta", [0.0, 1.0, 2.0])
def test_power_law_1d_slope(beta):
    rng = pattern_rng(0, "test")
    slopes = [_spectral_slope_1d(power_law_noise(8192, beta, rng)) for _ in range(8)]
    assert np.mean(slopes) == pytest.approx(-beta, abs=0.1)


def test_power_law_normalised():
    x = power_law_noise((128, 128), 1.5, pattern_rng(1, "test"))
    assert x.mean() == pytest.approx(0.0, abs=1e-12)
    assert x.std() == pytest.approx(1.0, rel=1e-12)


def test_power_law_2d_radial_slope():
    x = power_law_noise((256, 256), 2.0, pattern_rng(2, "test"))
    p = np.abs(np.fft.fft2(x)) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(256), np.fft.fftfreq(256), indexing="ij")
    f = np.hypot(fx, fy)
    keep = (f > 0.01) & (f < 0.4)
    slope = np.polyfit(np.log(f[keep]), np.log(p[keep]), 1)[0]
    assert slope == pytest.approx(-2.0, abs=0.1)
