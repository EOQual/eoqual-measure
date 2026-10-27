"""Validation de `measure_spatial_noise` contre la vérité du simulateur."""

import numpy as np
import pytest
from conftest import SEED_OFFSET

from eoqual.measure.radiometry import (
    band_structure,
    decompose_spatial,
    measure_dark_frame,
    measure_spatial_noise,
)
from eoqual.measure.synthetic import PushbroomSensor, SensorParams

MU = 1500.0


@pytest.fixture(scope="module")
def stack_case(structured_sensor):
    s = structured_sensor
    t = s.truth
    dark = measure_dark_frame(s.dark(n_frames=32, seed=1).frames)
    photons = t.photons_for_signal(MU)
    frames = s.expose(photons, n_frames=16, seed=2).frames
    return t, photons, dark, frames


def test_stack_with_dark(stack_case):
    t, photons, dark, frames = stack_case
    r = measure_spatial_noise(frames, dark=dark)
    true_map = t.signal_electrons(photons) / t.gain           # μ·g, DSNU retirée
    assert abs(r.spatial_sigma - true_map.std()) < 5 * r.u_spatial_sigma
    assert r.relative == pytest.approx(t.prnu_rms, rel=0.01)
    assert r.noise_variance == pytest.approx(r.temporal_variance / 16 + dark.temporal_variance / 32)


def test_stack_decomposition(stack_case):
    t, photons, dark, frames = stack_case
    r = measure_spatial_noise(frames, dark=dark)
    exact = decompose_spatial(t.signal_electrons(photons) / t.gain)
    h, w = t.params.shape
    tol = 5 * np.sqrt(r.noise_variance / (h * w)) * 4
    for name in ("row", "col", "pixel"):
        assert abs(getattr(r.decomposition, name) - getattr(exact, name)) < tol + 0.002 * getattr(exact, name)


def test_stack_without_dark(stack_case):
    t, photons, _, frames = stack_case
    r = measure_spatial_noise(frames)
    true_map = t.expected_mean(photons)                        # offset + DSNU + μ·g
    assert abs(r.spatial_sigma - true_map.std()) < 5 * r.u_spatial_sigma


@pytest.fixture(scope="module")
def pb_case():
    pb = PushbroomSensor(SensorParams(shape=(1, 1024), seed=21 + SEED_OFFSET, dsnu_pixel=0.0, dsnu_col=0.8,
                                      prnu_pixel=0.0, prnu_col=0.01))
    t = pb.truth
    dark = measure_dark_frame(pb.dark(n_lines=1000, seed=1).image, layout="pushbroom")
    img = pb.expose(t.photons_for_signal(MU), n_lines=500, seed=3).image.astype(float)
    return t, dark, img


def test_pushbroom(pb_case):
    t, dark, img = pb_case
    r = measure_spatial_noise(img, layout="pushbroom", dark=dark)
    true_profile = MU * t.detector_gain
    assert r.mean_map.shape == (1, 1024) and r.decomposition is None
    assert abs(r.spatial_sigma - true_profile.std()) < 5 * r.u_spatial_sigma


def test_pushbroom_temporal_insensitive_to_along_track_ramp(pb_case):
    """Variation lente le long de la trace : σ²_temp par différences de lignes inchangé."""
    t, dark, img = pb_case
    r0 = measure_spatial_noise(img, layout="pushbroom", dark=dark)
    ramp = np.linspace(-20, 20, img.shape[0])[:, None]
    r1 = measure_spatial_noise(img + ramp, layout="pushbroom", dark=dark)
    assert r1.temporal_variance == pytest.approx(r0.temporal_variance, rel=0.002)
    assert r1.spatial_sigma == pytest.approx(r0.spatial_sigma, rel=1e-6)


def test_bands(pb_case):
    """Bandes de période 4 injectées : σ des phases = celui du profil vrai complet."""
    t, dark, img = pb_case
    band = 2.0 * np.tile([1.0, -1.0, 0.5, -0.5], 256)
    r = measure_spatial_noise(img + band[None, :], layout="pushbroom", dark=dark, period=4)
    exact = band_structure(MU * t.detector_gain + band, 4)
    n_per = 256
    assert abs(r.bands.sigma - exact.sigma) < 5 * np.sqrt(r.noise_variance / n_per)
    np.testing.assert_allclose(r.bands.profile, exact.profile, atol=5 * np.sqrt(r.noise_variance / n_per))


def test_band_structure_exact():
    rng = np.random.default_rng(0)
    q = np.array([3.0, -1.0, -2.0])
    p = np.tile(q, 4000) + rng.normal(0, 1.0, 12000)
    b = band_structure(p, 3)
    assert b.sigma == pytest.approx(q.std(), rel=0.01)


def test_input_checks(pb_case):
    _, _, img = pb_case
    with pytest.raises(ValueError):
        measure_spatial_noise(img, layout="whiskbroom")
    with pytest.raises(ValueError):
        measure_spatial_noise(img[:1], layout="pushbroom")
    with pytest.raises(ValueError):
        measure_spatial_noise(img, layout="pushbroom", dark=np.zeros(10))
    with pytest.raises(ValueError):
        band_structure(np.zeros(5), 4)
