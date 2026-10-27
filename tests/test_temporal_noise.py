"""Validation de `measure_temporal_noise` contre la vérité du simulateur."""

import numpy as np
import pytest
from conftest import SEED_OFFSET

from eoqual.measure.radiometry import measure_dark_frame, measure_temporal_noise
from eoqual.measure.synthetic import MatrixSensor, SensorParams


@pytest.mark.parametrize("signal_dn", [20.0, 1000.0, 3000.0])
def test_pair(sensor, signal_dn):
    t = sensor.truth
    photons = t.photons_for_signal(signal_dn)
    r = measure_temporal_noise(sensor.expose(photons, n_frames=2, seed=5).frames, dark=t.params.offset)
    expected = t.a * signal_dn + t.b
    assert r.method == "pair"
    assert abs(r.temporal_variance - expected) < 5 * r.u_temporal_variance + 0.005 * t.b
    assert abs(r.signal - signal_dn) < 5 * np.sqrt(expected / (2 * t.params.shape[0] * t.params.shape[1])) + 0.05


def test_stack(sensor):
    t = sensor.truth
    photons = t.photons_for_signal(800.0)
    r = measure_temporal_noise(sensor.expose(photons, n_frames=16, seed=6).frames)
    expected = t.a * 800.0 + t.b
    assert r.method == "stack"
    assert r.variance_map.shape == t.params.shape
    assert abs(r.temporal_variance - expected) < 5 * r.u_temporal_variance + 0.005 * t.b
    assert r.signal is None


def test_pair_removes_fixed_pattern():
    """Forte PRNU (3 %) : la variance spatiale explose, la différence de paire non."""
    s = MatrixSensor(SensorParams(seed=8 + SEED_OFFSET, prnu_pixel=0.03, dsnu_pixel=2.0))
    t = s.truth
    signal = 3000.0
    frames = s.expose(t.photons_for_signal(signal), n_frames=2, seed=7).frames
    r = measure_temporal_noise(frames)
    expected = t.a * signal + t.b
    assert frames[0].astype(float).var() > 5 * expected      # FPN dominant
    assert abs(r.temporal_variance - expected) < 5 * r.u_temporal_variance + 0.005 * t.b


def test_pair_insensitive_to_level_shift(sensor):
    """Écart d'offset global entre les deux poses : sans effet sur σ²."""
    t = sensor.truth
    frames = sensor.expose(t.photons_for_signal(500.0), n_frames=2, seed=8).frames.astype(float)
    r0 = measure_temporal_noise(frames)
    frames[1] += 7.0
    r1 = measure_temporal_noise(frames)
    assert r1.temporal_variance == pytest.approx(r0.temporal_variance, rel=1e-12)


def test_dark_inputs(sensor):
    t = sensor.truth
    dark = measure_dark_frame(sensor.dark(n_frames=16, seed=9).frames)
    frames = sensor.expose(t.photons_for_signal(600.0), n_frames=2, seed=10).frames
    r_res = measure_temporal_noise(frames, dark=dark)
    r_map = measure_temporal_noise(frames, dark=dark.offset_map)
    assert r_res.signal == pytest.approx(r_map.signal)
    with pytest.raises(ValueError):
        measure_temporal_noise(frames, dark=sensor.dark(n_frames=2, seed=0).frames)


def test_input_checks(sensor):
    frames = sensor.dark(n_frames=2, seed=0).frames
    with pytest.raises(ValueError):
        measure_temporal_noise(frames[:1])
    with pytest.raises(ValueError):
        measure_temporal_noise(frames, method="stack")
    with pytest.raises(ValueError):
        measure_temporal_noise(frames, method="median")


@pytest.mark.parametrize("eps", [0.01, 0.05])
def test_pair_source_fluctuation_bias(eps):
    """
    Fluctuation multiplicative ε de la source entre les deux poses :
    biais ≈ ε²·var_spatiale(x₁)/2 (REFERENCE_TECHNIQUE.md, partie IV, §3.2.3).
    """
    s = MatrixSensor(SensorParams(seed=9 + SEED_OFFSET, prnu_pixel=0.03))
    t = s.truth
    photons = t.photons_for_signal(3000.0)
    x1 = s.expose(photons, n_frames=1, seed=1).frames[0].astype(float)
    x2 = s.expose(photons * (1 + eps), n_frames=1, seed=2).frames[0].astype(float)
    r = measure_temporal_noise(np.stack([x1, x2]))
    # Variance temporelle vraie moyenne des deux poses
    true_var = (t.expected_variance(photons).mean() + t.expected_variance(photons * (1 + eps)).mean()) / 2
    predicted_bias = eps**2 * x1.var() / 2
    bias = r.temporal_variance - true_var
    assert abs(bias - predicted_bias) < 5 * r.u_temporal_variance
