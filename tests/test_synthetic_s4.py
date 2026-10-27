"""Vérification des effets S4 : transmission optique, non-linéarité, pixels morts et bloqués."""

import numpy as np
import pytest
from conftest import SEED_OFFSET

from eoqual.measure.synthetic import MatrixSensor, PushbroomSensor, SensorParams


def _sensor(**kw):
    return MatrixSensor(SensorParams(seed=100 + SEED_OFFSET, **kw))


def test_off_by_default(sensor):
    t = sensor.truth
    assert t.transmission is None and t.nonlinearity_map is None and t.defects is None


def test_transmission():
    s = _sensor(vignetting=0.15, n_dust=5, dust_depth=0.08)
    t = s.truth
    tr = t.transmission
    h, w = t.params.shape
    assert tr.vignetting[0, 0] == pytest.approx(0.85) and tr.vignetting[-1, -1] == pytest.approx(0.85)
    assert tr.vignetting.max() == pytest.approx(1.0, abs=1e-4)
    # Anneaux d'atténuation d : les poussières qui se chevauchent multiplient leurs atténuations.
    assert (1 - 0.08) ** 5 - 1e-12 <= tr.dust.min() < 1 - 0.07
    np.testing.assert_allclose(tr.total, tr.vignetting * tr.dust)
    # Le flat moyen suit T·g (la transmission ne multiplie que la lumière).
    k = 16
    photons = t.photons_for_signal(2000.0)
    frames = s.expose(photons, n_frames=k, seed=1).frames.astype(float)
    err = frames.mean(axis=0) - t.expected_mean(photons)
    assert abs(err.mean()) < 5 * np.sqrt(t.expected_variance(photons).mean() / (k * h * w))
    np.testing.assert_allclose(t.signal_electrons(photons), t.params.qe * photons * t.gain_map * tr.total)


@pytest.mark.parametrize("signal_dn", [500.0, 3000.0])
def test_nonlinearity_mean_and_variance(signal_dn):
    s = _sensor(nonlinearity=0.1, nonlinearity_dispersion=0.02)
    t = s.truth
    k = 32
    photons = t.photons_for_signal(signal_dn)
    frames = s.expose(photons, n_frames=k, seed=2).frames.astype(float)
    n = frames[0].size
    mean_err = frames.mean(axis=0) - t.expected_mean(photons)
    var_true = t.expected_variance(photons)
    assert abs(mean_err.mean()) < 5 * np.sqrt(var_true.mean() / (k * n)) + 0.01
    var = frames.var(axis=0, ddof=1).mean()
    assert var == pytest.approx(var_true.mean(), rel=5 * np.sqrt(2 / ((k - 1) * n)) + 0.005)
    # La compression est visible : signal mesuré < linéaire.
    linear = t.offset_map + t.signal_electrons(photons) / t.gain
    assert (frames.mean(axis=0) - linear).mean() < -0.5 * 0.1 * signal_dn**2 * t.gain / t.params.full_well
    assert t.nonlinearity_map.std() == pytest.approx(0.02, rel=0.02)


def test_dead_and_stuck_pixels():
    s = _sensor(dead_pixel_fraction=2e-3, stuck_pixel_fraction=1e-3)
    t = s.truth
    d = t.defects
    n = t.params.shape[0] * t.params.shape[1]
    assert d.dead.sum() == round(2e-3 * n) and d.stuck.sum() == round(1e-3 * n)
    assert not np.any(d.dead & d.stuck)
    frames = s.expose(t.photons_for_signal(2500.0), n_frames=16, seed=3).frames.astype(float)
    # Morts : offset et bruit de lecture seulement.
    dead_resid = frames[:, d.dead] - t.offset_map[d.dead]
    assert abs(dead_resid.mean()) < 5 * np.sqrt(t.b / dead_resid.size)
    assert dead_resid.var() == pytest.approx(t.b, rel=0.2)
    # Bloqués : code constant, égal à la vérité.
    np.testing.assert_array_equal(frames[:, d.stuck], np.broadcast_to(d.stuck_value[d.stuck], (16, d.stuck.sum())))


def test_pushbroom_response_effects():
    pb = PushbroomSensor(SensorParams(shape=(1, 256), seed=101 + SEED_OFFSET, nonlinearity=0.1,
                                      dead_pixel_fraction=0.02, stuck_pixel_fraction=0.02, vignetting=0.1))
    t = pb.truth
    photons = t.photons_for_signal(3000.0)
    img = pb.expose(photons, n_lines=1000, seed=4).image.astype(float)
    err = img.mean(axis=0) - t.expected_mean(photons)[0]
    ok = ~t.defects.stuck[0]
    assert np.abs(err[ok]).max() < 5 * np.sqrt(t.expected_variance(photons)[0].max() / 1000)
    assert img[:, t.defects.stuck[0]].var(axis=0).max() == 0.0


def test_s4_validation():
    with pytest.raises(ValueError):
        SensorParams(vignetting=1.2)
    with pytest.raises(ValueError):
        SensorParams(nonlinearity=0.8)
    with pytest.raises(ValueError):
        SensorParams(dead_pixel_fraction=0.6, stuck_pixel_fraction=0.6)
