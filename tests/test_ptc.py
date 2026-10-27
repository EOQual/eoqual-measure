"""Validation de `measure_ptc` contre la vérité du simulateur."""

import numpy as np
import pytest
from conftest import SEED_OFFSET

from eoqual.measure.radiometry import measure_dark_frame, measure_ptc
from eoqual.measure.synthetic import MatrixSensor, SensorParams

LEVELS = np.array([50, 150, 400, 800, 1500, 2200, 3000, 3600, 3900, 4100, 4500.0])


def _series(sensor, levels, seed0=10, n_frames=2):
    t = sensor.truth
    photons = [t.photons_for_signal(s) for s in levels]
    flats = [sensor.expose(p, n_frames=n_frames, seed=seed0 + i).frames for i, p in enumerate(photons)]
    return flats, photons


@pytest.fixture(scope="module")
def ptc(sensor):
    t = sensor.truth
    flats, photons = _series(sensor, LEVELS)
    dark = sensor.dark(n_frames=16, seed=1).frames
    return measure_ptc(flats, dark, exposures=photons, saturation_dn=t.saturation_dn)


def test_gain(sensor, ptc):
    t = sensor.truth
    assert abs(ptc.gain - t.gain) < 5 * ptc.u_gain
    assert ptc.gain == pytest.approx(t.gain, rel=0.02)
    assert abs(ptc.a - t.a) < 5 * ptc.u_a


def test_read_noise(sensor, ptc):
    t = sensor.truth
    assert abs(ptc.read_noise_e - t.params.read_noise) < 5 * ptc.u_read_noise_e + 0.01 * t.params.read_noise
    assert abs(ptc.b - t.b) < 5 * ptc.u_b + 0.005 * t.b


def test_fit_quality(ptc):
    # χ² réduit : loi χ²_ν/ν, ν = nombre de points − 2.
    nu = ptc.used.sum() - 2
    assert abs(ptc.reduced_chi2 - 1) < 5 * np.sqrt(2 / nu)


def test_saturation_excluded(sensor, ptc):
    t = sensor.truth
    adc_limit = t.saturation_dn - t.params.offset
    assert 0.9 * adc_limit < ptc.saturation_signal_dn <= adc_limit
    assert not ptc.used[ptc.signal > 0.7 * ptc.saturation_signal_dn].any()
    assert ptc.saturated_fraction[-1] > 0.5


def test_linearity(ptc):
    assert ptc.linearity_error is not None
    assert ptc.linearity_error < 0.2      # simulateur linéaire : écarts dus au bruit seul


def test_full_well_saturation():
    """
    Puits saturant avant le convertisseur.

    Le point de saturation (variance maximale) est sous la capacité du
    puits FW : la variance de min(X, FW), X ~ N(μ_e, μ_e), culmine vers
    FW − 2,8·√FW, et son pic est plat. Le point mesuré doit tomber là où
    la variance attendue reste à moins de 5 incertitudes du maximum.
    """
    fw = 6000.0
    s = MatrixSensor(SensorParams(seed=4 + SEED_OFFSET, full_well=fw))
    t = s.truth
    levels = np.array([50, 300, 700, 1200, 1600, 1850, 1900, 1925, 1950, 1975, 2000, 2100, 2400.0])
    flats, _ = _series(s, levels)
    r = measure_ptc(flats, s.dark(n_frames=16, seed=2).frames, saturation_dn=t.saturation_dn)

    mu = np.linspace(fw - 10 * np.sqrt(fw), fw, 2001)
    expected = _clipped_variance_dn(mu, fw, t.gain, t.b)
    i_sat = int(np.argmin(np.abs(r.signal - r.saturation_signal_dn)))
    tol = 5 * r.u_variance[i_sat]
    plateau = mu[expected >= expected.max() - tol]
    assert plateau.min() <= r.saturation_e <= plateau.max() * (1 + 3 * r.u_gain / r.gain)
    assert r.gain == pytest.approx(t.gain, rel=0.03)


def _clipped_variance_dn(mu_e, fw, gain, b):
    """Variance temporelle (DN²) d'un signal poissonnien (≈ normal) écrêté à fw."""
    from scipy import stats
    s = np.sqrt(mu_e)
    z = (fw - mu_e) / s
    cdf, pdf = stats.norm.cdf(z), stats.norm.pdf(z)
    m1 = mu_e * cdf - s * pdf + fw * (1 - cdf)
    m2 = (mu_e**2 + s**2) * cdf - s * (mu_e + fw) * pdf + fw**2 * (1 - cdf)
    return (m2 - m1**2) / gain**2 + b


def test_saturation_not_reached(sensor):
    flats, _ = _series(sensor, [100, 500, 1000, 1500.0])
    r = measure_ptc(flats, sensor.dark(n_frames=8, seed=3).frames)
    assert r.saturation_signal_dn is None
    assert r.used.all()


def test_dark_result_accepted(sensor):
    flats, _ = _series(sensor, [100, 500, 1000.0])
    dark = measure_dark_frame(sensor.dark(n_frames=8, seed=3).frames)
    r = measure_ptc(flats, dark)
    assert r.signal[0] == 0.0 and r.variance[0] == dark.temporal_variance


def test_input_checks(sensor):
    flats, _ = _series(sensor, [100, 500.0])
    dark = sensor.dark(n_frames=4, seed=0).frames
    with pytest.raises(ValueError):
        measure_ptc(flats[:1], dark)
    with pytest.raises(ValueError):
        measure_ptc(flats, dark, include_dark=False)       # deux points seulement
    with pytest.raises(ValueError):
        measure_ptc(flats, dark, exposures=[1.0])
