"""Validation de `measure_pushbroom_noise` contre la vérité du simulateur push broom."""

import numpy as np
import pytest
from conftest import SEED_OFFSET

from eoqual.measure.radiometry import measure_dark_frame, measure_pushbroom_noise
from eoqual.measure.radiometry.pushbroom import _filtered_segments, effective_dof, rho1_null
from eoqual.measure.synthetic import PushbroomSensor, SensorParams, natural_scene, pattern_rng

LEVELS = [200, 600, 1200, 2000, 3000]
SHAPE = (1024, 513)


@pytest.fixture(scope="module")
def pb():
    return PushbroomSensor(SensorParams(shape=(1, SHAPE[1]), seed=21 + SEED_OFFSET,
                                        dsnu_pixel=0.0, dsnu_col=0.8,
                                        prnu_pixel=0.0, prnu_col=0.01))


@pytest.fixture(scope="module")
def dark(pb):
    return measure_dark_frame(pb.dark(n_lines=1000, seed=1).image, layout="pushbroom")


def _images(pb, contrast, beta=3.0, psf=0.6):
    t = pb.truth
    out = []
    for i, lvl in enumerate(LEVELS):
        sc = natural_scene(SHAPE, t.photons_for_signal(lvl), pattern_rng(i + 100 * SEED_OFFSET, "scene"),
                           contrast=contrast, beta=beta, psf_sigma=psf,
                           gradient_along=0.1, gradient_across=0.05)
        out.append(pb.expose(sc.photons, seed=100 + i).image)
    return out


@pytest.mark.parametrize("order", [1, 2])
def test_null_statistics(pb, dark, order):
    """Bruit pur : dispersion de ρ₁ (Bartlett) et degrés de liberté effectifs conformes."""
    t = pb.truth
    img = pb.expose(t.photons_for_signal(800.0), n_lines=32 * 400, seed=9).image.astype(float) - dark.offset_map[0]
    mu, var, rho, _, _, n = _filtered_segments(img, 32, order, 2)
    _, sd = rho1_null(n, order)
    assert rho.std() == pytest.approx(sd, rel=0.05)
    expected = t.a * mu.mean() + t.b
    dof = effective_dof(n, order)
    # Var(s²) = 2σ⁴/ν : rapport à 1 à 5 % près (approximation gaussienne).
    assert var.var() / expected**2 * dof / 2 == pytest.approx(1.0, abs=0.05)
    assert var.mean() == pytest.approx(expected, rel=5 * np.sqrt(2 / (dof * var.size)) + 0.005)


@pytest.fixture(scope="module")
def uniform_result(pb, dark):
    return measure_pushbroom_noise(_images(pb, contrast=0.0), dark=dark)


def test_uniform_scene(pb, uniform_result):
    t = pb.truth
    r = uniform_result
    assert abs(r.a - t.a) < 5 * r.u_a
    assert abs(r.b - t.b) < 5 * r.u_b
    assert abs(r.reduced_chi2 - 1) < 0.05
    assert not r.texture_suspected
    assert r.rejected_fraction == 0.0


def test_b_from_dark(pb, dark):
    t = pb.truth
    r = measure_pushbroom_noise(_images(pb, contrast=0.0), dark=dark, b=dark.temporal_variance)
    assert r.b_fixed and r.u_b == 0.0 and r.cov.shape == (1, 1)
    assert abs(r.a - t.a) < 5 * r.u_a + abs(dark.temporal_variance - t.b) / 1000


def test_smooth_blurred_texture(pb, dark):
    """Texture lisse (β = 3) de 3 %, optique σ = 0,6 px : mesure non biaisée."""
    t = pb.truth
    r = measure_pushbroom_noise(_images(pb, contrast=0.03), dark=dark)
    assert abs(r.a - t.a) < 5 * r.u_a
    assert not r.texture_suspected


def test_rough_texture_is_flagged(pb, dark):
    """Texture rugueuse (β = 2) : σ² surestimé, et le diagnostic le signale."""
    t = pb.truth
    r = measure_pushbroom_noise(_images(pb, contrast=0.03, beta=2.0), dark=dark)
    assert r.a > t.a + 3 * r.u_a
    assert r.texture_suspected


@pytest.mark.parametrize("along, across", [(1, 1), (2, 0), (2, 1)])
def test_other_filters_on_uniform_scene(pb, dark, along, across):
    t = pb.truth
    r = measure_pushbroom_noise(_images(pb, contrast=0.0), dark=dark, along_order=along, across_order=across)
    assert abs(r.a - t.a) < 5 * r.u_a


def test_rejection_option(pb, dark):
    r = measure_pushbroom_noise(_images(pb, contrast=0.0), dark=dark, rho_threshold=3.0)
    assert 0 < r.rejected_fraction < 0.05


def test_binned_summary(uniform_result):
    mu, var, u = uniform_result.binned(8)
    assert mu.size == 8 and np.all(np.diff(mu) > 0)
    assert np.all(np.abs(var - uniform_result.sigma(mu) ** 2) < 5 * u)


def test_input_checks(pb):
    img = pb.expose(100.0, n_lines=64, seed=0).image
    with pytest.raises(ValueError):
        measure_pushbroom_noise(img, segment_length=4)
    with pytest.raises(ValueError):
        measure_pushbroom_noise([img, img[:, :100]])
    with pytest.raises(ValueError):
        measure_pushbroom_noise(img, along_order=3)
    with pytest.raises(ValueError):
        measure_pushbroom_noise(img, dark=np.zeros(10))
