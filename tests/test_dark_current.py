"""Validation de `measure_dark_current` contre la vérité du simulateur S1."""

import numpy as np
import pytest
from conftest import SEED_OFFSET

from eoqual.measure.radiometry import measure_dark_current
from eoqual.measure.synthetic import MatrixSensor, SensorParams
from eoqual.measure.synthetic.dark import doubling_temperature, temperature_factor

TIMES = [1.0, 10.0, 30.0, 60.0, 100.0]


@pytest.fixture(scope="module")
def case():
    s = MatrixSensor(SensorParams(seed=90 + SEED_OFFSET, dark_current=2.0, dark_nonuniformity=0.3,
                                  hot_pixel_fraction=1e-3, hot_pixel_range=(30.0, 1000.0)))
    t = s.truth
    stacks = [s.dark(n_frames=16, seed=i, exposure_time=te).frames for i, te in enumerate(TIMES)]
    r = measure_dark_current(stacks, TIMES, saturation_dn=t.saturation_dn)
    return s, t, r


def test_mean_and_dsnu(case):
    _, t, r = case
    d = t.dark_current
    ordinary = r.valid & ~d.hot
    true_dn = d.current[ordinary] / t.gain
    assert abs(r.mean_current - true_dn.mean()) < 5 * r.u_mean_current
    assert abs(r.dsnu - true_dn.std()) < 5 * r.u_dsnu
    assert r.dsnu_relative == pytest.approx(np.sqrt(np.exp(0.09) - 1), rel=0.05)


def test_current_map_and_offset(case):
    _, t, r = case
    d = t.dark_current
    ordinary = r.valid & ~d.hot
    err = (r.current_map - d.current / t.gain)[ordinary]
    assert err.std() == pytest.approx(np.nanmean(r.current_noise[ordinary]), rel=0.05)
    assert abs(err.mean()) < 5 * err.std() / np.sqrt(err.size)
    off_err = (r.offset_map - t.offset_map)[ordinary]
    assert abs(off_err.mean()) < 5 * off_err.std() / np.sqrt(off_err.size)


def test_hot_pixels(case):
    _, t, r = case
    d = t.dark_current
    assert not np.any(r.hot_pixels & ~d.hot)                      # aucune fausse détection
    assert (r.hot_pixels & d.hot).sum() >= 0.98 * d.hot.sum()
    # Courant des pixels chauds, y compris ceux qui saturent aux longues poses.
    both = d.hot & r.valid
    np.testing.assert_allclose(np.median(r.current_map[both] / (d.current[both] / t.gain)), 1.0, atol=0.01)


def test_poisson_gain(case):
    _, t, r = case
    assert abs(r.poisson_gain - t.gain) < 5 * r.u_poisson_gain
    assert r.mean_current_e == pytest.approx(r.mean_current * r.poisson_gain)


def test_temperature_law():
    s = MatrixSensor(SensorParams(shape=(128, 128), seed=91 + SEED_OFFSET, dark_current=2.0,
                                  dark_nonuniformity=0.2))
    t = s.truth
    conds = [(te, T) for T in (0.0, 10.0, 20.0, 30.0) for te in (10.0, 100.0)]
    stacks = [s.dark(n_frames=16, seed=i, exposure_time=te, temperature=T).frames
              for i, (te, T) in enumerate(conds)]
    r = measure_dark_current(stacks, [c[0] for c in conds], temperatures=[c[1] for c in conds],
                             gain=t.gain, saturation_dn=t.saturation_dn)
    f = r.temperature_fit
    assert f is not None and np.array_equal(f.temperatures, [0.0, 10.0, 20.0, 30.0])
    assert abs(f.bandgap - 1.12) < 5 * f.u_bandgap
    assert f.doubling_temperature == pytest.approx(doubling_temperature(f.reference_temperature, 1.12), rel=0.01)
    true_means = [(t.dark_current.current / t.gain).mean() * temperature_factor(T, 20.0, 1.12)
                  for T in f.temperatures]
    np.testing.assert_allclose(f.mean_current, true_means, rtol=0.01)
    assert r.mean_current_e == pytest.approx(r.mean_current * t.gain)


def test_glow_appears_in_current_map():
    s = MatrixSensor(SensorParams(shape=(128, 128), seed=92 + SEED_OFFSET, dark_current=1.0,
                                  glow_rate=40.0, glow_scale=10.0))
    t = s.truth
    stacks = [s.dark(n_frames=8, seed=i, exposure_time=te).frames for i, te in enumerate([5.0, 50.0])]
    r = measure_dark_current(stacks, [5.0, 50.0], saturation_dn=t.saturation_dn)
    expected = (t.dark_current.current + t.dark_current.glow) / t.gain
    assert r.current_map[0, 0] == pytest.approx(expected[0, 0], rel=0.05)
    assert r.hot_pixels[0, 0]                                      # la lueur passe pour des pixels chauds


def test_input_checks(case):
    s, _, _ = case
    a = s.dark(n_frames=2, seed=0, exposure_time=1.0).frames
    with pytest.raises(ValueError):
        measure_dark_current([a, a], [1.0])
    with pytest.raises(ValueError):
        measure_dark_current([a, a], [1.0, 1.0])                   # un seul temps distinct
    with pytest.raises(ValueError):
        measure_dark_current([a, a], [1.0, -2.0])
