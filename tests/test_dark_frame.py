"""Validation de `measure_dark_frame` contre la vérité du simulateur."""

import numpy as np
import pytest
from conftest import SEED_OFFSET

from eoqual.measure.radiometry import decompose_spatial, measure_dark_frame
from eoqual.measure.synthetic import MatrixSensor, SensorParams

K = 64


@pytest.fixture(scope="module")
def dark(structured_sensor):
    return structured_sensor.dark(n_frames=K, seed=1)


@pytest.fixture(scope="module")
def result(dark):
    return measure_dark_frame(dark.frames)


def test_offset_map(dark, result):
    t = dark.truth
    err = result.offset_map - t.offset_map
    assert err.std() == pytest.approx(np.sqrt(t.b / K), rel=0.02)
    assert abs(result.mean_offset - t.offset_map.mean()) < 5 * result.u_mean_offset


def test_temporal_variance_is_b(dark, result):
    # Tolérance : incertitude statistique + validité du terme 1/12 (≈ 0,5 %).
    tol = 5 * result.u_temporal_variance + 0.005 * dark.truth.b
    assert abs(result.temporal_variance - dark.truth.b) < tol
    assert result.temporal_noise == pytest.approx(np.sqrt(result.temporal_variance))


def test_dsnu_total(dark, result):
    assert abs(result.dsnu - dark.truth.dsnu_rms) < 5 * result.u_dsnu


def _u_component(sigma, s2_pe, n_other, n_self):
    """Incertitude-type sur un écart-type de composante ligne (ou colonne)."""
    cross = 2 * sigma * np.sqrt(s2_pe / n_other) / np.sqrt(n_self)
    spread = np.sqrt(2 / n_self) * s2_pe / n_other
    return np.hypot(cross, spread) / (2 * sigma)


def test_dsnu_decomposition(dark, result):
    t = dark.truth
    d = result.decomposition
    h, w = t.params.shape
    s2_pe = d.pixel_var + d.noise_variance
    assert abs(d.row - t.dsnu.row.std()) < 5 * _u_component(t.dsnu.row.std(), s2_pe, w, h)
    assert abs(d.col - t.dsnu.col.std()) < 5 * _u_component(t.dsnu.col.std(), s2_pe, h, w)
    assert d.pixel == pytest.approx(t.dsnu.pixel.std(), rel=0.01)
    np.testing.assert_allclose(d.col_profile, t.dsnu.col, atol=5 * np.sqrt(s2_pe / h))


def test_stationary_stack(result):
    assert abs(result.drift_per_frame) < 5 * result.u_drift
    assert result.stationarity_ratio == pytest.approx(1.0, abs=5 * np.sqrt(2 / (K - 1)))


def test_drift_is_detected(dark):
    drift = 0.02   # DN par pose, ajoutée à la pile simulée
    frames = dark.frames + drift * np.arange(K)[:, None, None]
    r = measure_dark_frame(frames)
    assert abs(r.drift_per_frame - drift) < 5 * r.u_drift
    assert r.stationarity_ratio > 10


def test_pushbroom_block():
    """Bloc de lignes push broom : un détecteur par colonne."""
    s = MatrixSensor(SensorParams(shape=(1, 512), seed=3 + SEED_OFFSET, dsnu_pixel=0.0, dsnu_col=0.8))
    lines = s.dark(n_frames=400, seed=2).frames[:, 0, :]          # (M, W)
    r = measure_dark_frame(lines, layout="pushbroom")
    t = s.truth
    assert r.offset_map.shape == (1, 512)
    assert r.decomposition.row is None
    assert abs(r.dsnu - t.dsnu_rms) < 5 * r.u_dsnu
    assert abs(r.temporal_variance - t.b) < 5 * r.u_temporal_variance + 0.005 * t.b


def test_input_checks(dark):
    with pytest.raises(ValueError):
        measure_dark_frame(dark.frames[:1])
    with pytest.raises(ValueError):
        measure_dark_frame(dark.frames[0])
    with pytest.raises(ValueError):
        measure_dark_frame(dark.frames, layout="whiskbroom")


def test_decomposition_exact_map():
    """Carte sans bruit : composantes retrouvées, profils exacts à σ_p/√n près."""
    rng = np.random.default_rng(0)
    h, w = 300, 200
    r = rng.normal(0, 2.0, h); r -= r.mean()
    c = rng.normal(0, 1.0, w); c -= c.mean()
    p = rng.normal(0, 0.5, (h, w)); p -= p.mean()
    d = decompose_spatial(10 + p + r[:, None] + c[None, :])
    assert d.row == pytest.approx(r.std(), rel=0.02)
    assert d.col == pytest.approx(c.std(), rel=0.03)
    assert d.pixel == pytest.approx(p.std(), rel=0.01)
    assert d.total == pytest.approx((p + r[:, None] + c[None, :]).std(), rel=1e-12)
