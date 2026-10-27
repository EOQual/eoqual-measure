"""Validation de `snr_at_levels` : valeur et propagation des incertitudes."""

import numpy as np
import pytest
from conftest import SEED_OFFSET

from eoqual.measure.radiometry import measure_ptc, snr_at_levels
from eoqual.measure.synthetic import MatrixSensor, SensorParams

LEVELS = [30.0, 300.0, 3000.0]


def _ptc(sensor, seed0):
    t = sensor.truth
    flats = [sensor.expose(t.photons_for_signal(s), n_frames=2, seed=seed0 + i).frames
             for i, s in enumerate([50, 200, 600, 1200, 2000, 3000])]
    return measure_ptc(flats, sensor.dark(n_frames=8, seed=seed0 + 99).frames)


def test_snr_value(sensor):
    t = sensor.truth
    r = snr_at_levels(_ptc(sensor, 10), LEVELS)
    true_snr = np.array(LEVELS) / np.sqrt(t.a * np.array(LEVELS) + t.b)
    assert np.all(np.abs(r.snr - true_snr) < 5 * r.u_snr)


def test_uncertainty_by_monte_carlo():
    """
    L'écart-type du SNR sur n PTC indépendantes égale l'incertitude propagée.

    Tolérance : 5 fois l'incertitude relative d'un écart-type estimé sur n
    tirages, 1/√(2(n − 1)), plus 5 % pour l'approximation linéaire.
    """
    n = 150
    s = MatrixSensor(SensorParams(shape=(32, 32), seed=30 + SEED_OFFSET))
    runs = [snr_at_levels(_ptc(s, 1000 * k), LEVELS) for k in range(n)]
    snr = np.array([r.snr for r in runs])
    u = np.array([r.u_snr for r in runs]).mean(axis=0)
    np.testing.assert_allclose(snr.std(axis=0, ddof=1), u, rtol=5 / np.sqrt(2 * (n - 1)) + 0.05)


def test_levels_in_electrons(sensor):
    m = _ptc(sensor, 20)
    r_dn = snr_at_levels(m, [1000.0])
    r_e = snr_at_levels(m, [1000.0 * m.gain], gain=m.gain)
    assert r_e.snr[0] == pytest.approx(r_dn.snr[0])


def test_fixed_b_model():
    class Model:
        a, b, cov = 0.5, 4.0, np.array([[1e-6]])
    r = snr_at_levels(Model(), [100.0])
    assert r.snr[0] == pytest.approx(100 / np.sqrt(54))
    assert r.u_sigma[0] == pytest.approx(100 / (2 * np.sqrt(54)) * 1e-3)


def test_input_checks(sensor):
    m = _ptc(sensor, 30)
    with pytest.raises(ValueError):
        snr_at_levels(m, [0.0])
