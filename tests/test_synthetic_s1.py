"""Vérification des effets S1 : courant d'obscurité, température, pixels chauds, lueur, offset, overscan."""

import numpy as np
import pytest
from conftest import SEED_OFFSET

from eoqual.measure.synthetic import MatrixSensor, PushbroomSensor, SensorParams
from eoqual.measure.synthetic.dark import doubling_temperature, temperature_factor


def _sensor(**kw):
    base = dict(seed=80 + SEED_OFFSET, dark_current=5.0, dark_nonuniformity=0.3)
    base.update(kw)
    return MatrixSensor(SensorParams(**base))


def test_off_by_default(sensor):
    t = sensor.truth
    assert t.dark_current is None and t.hot_pixels is None and t.overscan_offset is None
    a = sensor.dark(n_frames=2, seed=3)
    b = sensor.dark(n_frames=2, seed=3, exposure_time=100.0)
    np.testing.assert_array_equal(a.frames, b.frames)              # pas de courant : t sans effet
    assert a.bias_offsets is None and a.overscan is None


def test_dark_mean_and_poisson_variance():
    s = _sensor()
    t = s.truth
    k, texp = 32, 20.0
    frames = s.dark(n_frames=k, seed=1, exposure_time=texp).frames.astype(float)
    dark_dn = t.dark_electrons(texp) / t.gain
    err = frames.mean(axis=0) - (t.offset_map + dark_dn)
    var_expected = t.b + dark_dn / t.gain                          # Poisson : a·μ_d + b
    assert abs(err.mean()) < 5 * np.sqrt(var_expected.mean() / (k * err.size))
    assert err.std() == pytest.approx(np.sqrt(var_expected.mean() / k), rel=0.03)
    var = frames.var(axis=0, ddof=1).mean()
    assert var == pytest.approx(var_expected.mean(), rel=5 * np.sqrt(2 / ((k - 1) * err.size)) + 0.005)


def test_dark_linear_in_exposure_time():
    s = _sensor()
    t = s.truth
    means = [(s.dark(n_frames=4, seed=i, exposure_time=te).frames - t.offset_map).mean()
             for i, te in enumerate([0.0, 10.0, 20.0, 40.0])]
    slope = np.polyfit([0, 10, 20, 40], means, 1)[0]
    assert slope == pytest.approx(t.dark_current.current.mean() / t.gain, rel=0.01)


def test_temperature_law():
    assert doubling_temperature(20.0, 1.12) == pytest.approx(8.29, abs=0.01)
    assert temperature_factor(20.0, 20.0, 1.12) == 1.0
    s = _sensor()
    t = s.truth
    d0 = t.dark_electrons(10.0)
    d1 = t.dark_electrons(10.0, temperature=20.0 + doubling_temperature(20.0, 1.12))
    np.testing.assert_allclose(d1 / d0, 2.0, rtol=1e-9)            # doublement exact


def test_lognormal_map_and_hot_pixels():
    s = _sensor(shape=(512, 512), hot_pixel_fraction=1e-3, hot_pixel_range=(10.0, 1000.0))
    d = s.truth.dark_current
    n = 512 * 512
    assert d.relative.mean() == pytest.approx(1.0, abs=5 * 0.31 / np.sqrt(n))
    assert d.relative.std() == pytest.approx(np.sqrt(np.exp(0.09) - 1), rel=0.02)
    assert d.hot.sum() == round(1e-3 * n)
    f = d.hot_factor[d.hot]
    assert f.min() >= 10 and f.max() <= 1000
    # Log-uniforme : ln f uniforme sur [ln 10, ln 1000].
    assert np.mean(np.log(f)) == pytest.approx(np.log(100), abs=5 * np.log(100) / np.sqrt(3 * f.size))
    np.testing.assert_array_equal(s.truth.hot_pixels, d.hot)


def test_amplifier_glow():
    s = _sensor(dark_current=0.0, dark_nonuniformity=0.0, glow_rate=100.0, glow_scale=15.0)
    g = s.truth.dark_current.glow
    assert g[0, 0] == pytest.approx(100.0)
    assert g[0, 15] == pytest.approx(100.0 / np.e)
    frames = s.dark(n_frames=8, seed=4, exposure_time=5.0).frames.astype(float)
    corner = (frames.mean(axis=0) - s.truth.offset_map)[:3, :3].mean()
    assert corner == pytest.approx((g[:3, :3] * 5.0).mean() / s.truth.gain, rel=0.05)


def test_bias_and_overscan_track_each_other():
    s = _sensor(bias_jitter=0.5, bias_drift=0.05, overscan_columns=16)
    t = s.truth
    k = 64
    a = s.dark(n_frames=k, seed=5, exposure_time=10.0)
    assert a.overscan.shape == (k, t.params.shape[0], 16)
    assert np.diff(a.bias_offsets - 0.05 * np.arange(k)).std() > 0           # gigue présente
    os_level = a.overscan.astype(float).mean(axis=(1, 2)) - t.overscan_offset.mean()
    dark_os = t.params.dark_current * 10.0 / t.gain
    var_os = t.b + dark_os / t.gain
    n_os = 16 * t.params.shape[0]
    assert np.std(os_level - dark_os - a.bias_offsets) == pytest.approx(np.sqrt(var_os / n_os), rel=0.25)
    # L'overscan corrige la variation d'offset des pixels actifs.
    active = (a.frames.astype(float) - t.offset_map).mean(axis=(1, 2))
    corrected = active - (os_level - dark_os)
    assert corrected.std() < 0.2 * active.std()


def test_pushbroom_dark_and_drift():
    pb = PushbroomSensor(SensorParams(shape=(1, 512), seed=81 + SEED_OFFSET, dark_current=10.0,
                                      dark_nonuniformity=0.2, bias_drift=0.01, dsnu_pixel=0.0, prnu_pixel=0.0))
    t = pb.truth
    acq = pb.dark(n_lines=2000, seed=6, exposure_time=0.5)
    resid = acq.image - t.detector_offset - t.dark_electrons(0.5)[0] / t.gain
    assert np.polyfit(np.arange(2000), resid.mean(axis=1), 1)[0] == pytest.approx(0.01, rel=0.05)
    with pytest.raises(ValueError):
        PushbroomSensor(SensorParams(shape=(1, 64), overscan_columns=4))


def test_s1_validation():
    with pytest.raises(ValueError):
        SensorParams(hot_pixel_fraction=1.5)
    with pytest.raises(ValueError):
        SensorParams(hot_pixel_range=(0.5, 2.0))
    with pytest.raises(ValueError):
        _sensor().dark(n_frames=1, seed=0, exposure_time=-1.0)
