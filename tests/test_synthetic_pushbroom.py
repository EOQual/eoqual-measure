"""Vérification du simulateur push broom (S2) et des scènes."""

import numpy as np
import pytest
from conftest import SEED_OFFSET

from eoqual.measure.synthetic import PushbroomSensor, SensorParams, natural_scene, pattern_rng


@pytest.fixture(scope="module")
def pb():
    return PushbroomSensor(SensorParams(shape=(1, 512), seed=21 + SEED_OFFSET,
                                        dsnu_pixel=0.0, dsnu_col=0.8,
                                        prnu_pixel=0.0, prnu_col=0.01))


def test_shapes_and_format(pb):
    acq = pb.expose(1000.0, n_lines=300, seed=1)
    assert acq.image.shape == (300, 512) and acq.image.dtype == np.uint16
    assert pb.truth.detector_offset.shape == (512,)


def test_detector_statistics(pb):
    """Le long de la trace : moyenne = offset + signal × gain du détecteur ; variance = a·μ + b."""
    t = pb.truth
    m = 2000
    signal = 1500.0
    img = pb.expose(t.photons_for_signal(signal), n_lines=m, seed=2).image.astype(float)
    mu_det = signal * t.detector_gain
    expected_mean = t.detector_offset + mu_det
    var_det = t.a * mu_det + t.b
    err = img.mean(axis=0) - expected_mean
    assert err.std() == pytest.approx(np.sqrt(var_det.mean() / m), rel=0.1)
    var = img.var(axis=0, ddof=1).mean()
    assert var == pytest.approx(var_det.mean(), rel=5 * np.sqrt(2 / ((m - 1) * 512)) + 0.005)


def test_lines_are_independent(pb):
    """Bruit temporel blanc le long de la trace : autocorrélation au retard 1 nulle."""
    img = pb.expose(pb.truth.photons_for_signal(800.0), n_lines=4000, seed=3).image.astype(float)
    x = img - img.mean(axis=0)
    rho = np.sum(x[1:] * x[:-1]) / np.sum(x * x)
    assert abs(rho) < 5 / np.sqrt(x.size)


@pytest.mark.parametrize("n", [1, 4, 16])
def test_tdi(n):
    """TDI à N étages : signal ×N pour un même flux par étage, bruit de lecture inchangé."""
    p = SensorParams(shape=(1, 256), seed=5 + SEED_OFFSET, prnu_pixel=0.0, prnu_col=0.0)
    s = PushbroomSensor(p, tdi_stages=n)
    t = s.truth
    phi = 100.0                                   # photons par étage
    img = s.expose(phi, n_lines=4000, seed=4).image.astype(float)
    mu = n * p.qe * phi / p.conversion_gain
    assert (img.mean(axis=0) - t.detector_offset).mean() == pytest.approx(mu, rel=0.002)
    assert img.var(axis=0, ddof=1).mean() == pytest.approx(t.a * mu + t.b, rel=0.02)
    assert t.photons_for_signal(mu) == pytest.approx(phi)


def test_scene_input(pb):
    sc = natural_scene((400, 512), 2000.0, pattern_rng(0, "scene"), contrast=0.05, gradient_along=0.1)
    acq = pb.expose(sc.photons, seed=5)
    assert acq.image.shape == (400, 512)
    with pytest.raises(ValueError):
        pb.expose(np.ones((10, 3)), seed=0)
    with pytest.raises(ValueError):
        pb.expose(1.0, seed=0)                    # n_lines manquant
    with pytest.raises(ValueError):
        PushbroomSensor(SensorParams(shape=(2, 10)))


def test_natural_scene_components():
    rng = pattern_rng(3, "scene")
    sc = natural_scene((256, 512), 1000.0, rng, contrast=0.03, beta=3.0,
                       gradient_along=0.2, gradient_across=-0.1)
    assert sc.texture.std() == pytest.approx(1.0)
    assert sc.texture.mean() == pytest.approx(0.0, abs=1e-12)
    # Gradient : 20 % d'un bord à l'autre le long de la trace (centres de pixels).
    span = sc.illumination[-1, 0] - sc.illumination[0, 0]
    assert span == pytest.approx(0.2 * (1 - 1 / 256))
    assert sc.photons.mean() == pytest.approx(1000.0, rel=0.01)
    flat = natural_scene((64, 64), 500.0, rng, contrast=0.0)
    np.testing.assert_allclose(flat.photons, 500.0)


def test_scene_not_periodic():
    """La texture recadrée ne se raccorde pas d'un bord à l'autre."""
    sc = natural_scene((256, 256), 1000.0, pattern_rng(4, "scene"), contrast=0.05, beta=3.0)
    t = sc.texture
    edge_jump = np.abs(t[0] - t[-1]).mean()
    inner_step = np.abs(np.diff(t, axis=0)).mean()
    assert edge_jump > 5 * inner_step


def test_scene_psf_attenuates_high_frequencies():
    """
    Flou gaussien σ : spectre 2D de la texture multiplié par FTM² =
    exp(−4π²σ²f²), à une constante près (les textures sont renormalisées).
    Sans recadrage (pad = 1) le champ est périodique et la relation exacte.
    """
    sigma = 0.6
    kw = dict(contrast=0.05, beta=2.0, pad=1)
    sharp = natural_scene((128, 128), 1000.0, pattern_rng(5, "scene"), **kw)
    blur = natural_scene((128, 128), 1000.0, pattern_rng(5, "scene"), psf_sigma=sigma, **kw)
    ratio = np.abs(np.fft.fft2(blur.texture)) ** 2 / np.abs(np.fft.fft2(sharp.texture)) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(128), np.fft.fftfreq(128), indexing="ij")
    mtf2 = np.exp(-4 * np.pi**2 * sigma**2 * (fx**2 + fy**2))
    k = ratio / mtf2
    k = k[np.isfinite(k) & (mtf2 < 1)]
    assert k.std() / k.mean() < 1e-6
