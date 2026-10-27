"""
Validation de `measure_equalization_noise`.

Scénario : une barrette push broom est égalisée à partir d'une
calibration (obscurité et champ uniforme de M_cal lignes) ; le résidu vrai
de l'image égalisée est connu exactement à partir de la vérité et des
coefficients estimés. Une erreur de gain à grande échelle (BF) est
ajoutée volontairement.
"""

import numpy as np
import pytest
from conftest import SEED_OFFSET
from scipy.ndimage import gaussian_filter1d

from eoqual.measure.radiometry import measure_equalization_noise, profile_spectrum
from eoqual.measure.radiometry.equalization import _gaussian_sigma_for_cutoff
from eoqual.measure.synthetic import PushbroomSensor, SensorParams

W = 1024
MU = 1500.0


def _equalized(m_cal, lf_amplitude, seed):
    pb = PushbroomSensor(SensorParams(shape=(1, W), seed=seed, dsnu_pixel=0.0, dsnu_col=0.8,
                                      prnu_pixel=0.0, prnu_col=0.01))
    t = pb.truth
    cal_dark = pb.dark(n_lines=m_cal, seed=10).image.astype(float).mean(axis=0)
    cal_flat = pb.expose(t.photons_for_signal(2000.0), n_lines=m_cal, seed=11).image.astype(float).mean(axis=0)
    gain = (cal_flat - cal_dark) / np.mean(cal_flat - cal_dark)
    j = np.arange(W)
    gain = gain * (1 + lf_amplitude * np.sin(2 * np.pi * 3 * j / W))      # erreur BF (3 cycles)
    x = pb.expose(t.photons_for_signal(MU), n_lines=1000, seed=12).image.astype(float)
    eq = (x - cal_dark) / gain
    truth = (t.detector_offset + MU * t.detector_gain - cal_dark) / gain   # espérance du profil égalisé
    return eq, truth - truth.mean()


@pytest.fixture(scope="module", params=[(200, 0.004), (2000, 0.0)])
def case(request):
    m_cal, lf = request.param
    return _equalized(m_cal, lf, 40 + SEED_OFFSET)


def test_fpn(case):
    eq, truth = case
    r = measure_equalization_noise(eq)
    assert abs(r.fpn - truth.std()) < 5 * r.u_fpn
    assert r.fpn_relative == pytest.approx(r.fpn / r.mean_level)


def test_lf_hf(case):
    eq, truth = case
    r = measure_equalization_noise(eq)
    s = _gaussian_sigma_for_cutoff(r.cutoff)
    lf = gaussian_filter1d(truth, s, mode="reflect")
    assert abs(r.lf - lf.std()) < 5 * r.u_lf
    assert abs(r.hf - (truth - lf).std()) < 5 * r.u_hf


def test_noise_fractions_on_white_noise():
    """Bruit blanc : var(BF) ≈ v·Σh², var(HF) ≈ v·Σ(δ − h)² (à 5 σ statistiques et effets de bord près)."""
    rng = np.random.default_rng(1 + SEED_OFFSET)
    img = 100 + rng.normal(0, 3.0, (400, W))
    r = measure_equalization_noise(img)
    v_profile = r.profile.var()
    assert r.lf_profile.var() == pytest.approx(v_profile * r.lf_noise_fraction, rel=0.35)
    assert r.hf_profile.var() == pytest.approx(v_profile * r.hf_noise_fraction, rel=5 * np.sqrt(2 / W) + 0.02)
    assert r.lf_noise_fraction + r.hf_noise_fraction <= 1.0
    assert r.fpn < 5 * r.u_fpn                                  # aucun FPN


def test_spectrum_normalisation_and_white_level():
    rng = np.random.default_rng(2)
    p = rng.normal(0, 2.0, 4096)
    f, s = profile_spectrum(p)
    assert np.sum(s) * (1 / 4096) == pytest.approx(p.var(), rel=1e-12)
    assert s.mean() == pytest.approx(2 * 4.0, rel=5 / np.sqrt(f.size))     # S = 2σ² (unilatérale)
    f_odd, s_odd = profile_spectrum(p[:-1])
    assert np.sum(s_odd) / 4095 == pytest.approx(p[:-1].var(), rel=1e-12)


def test_lf_error_seen_in_spectrum():
    eq, _ = _equalized(2000, 0.004, 50 + SEED_OFFSET)
    r = measure_equalization_noise(eq)
    k = np.argmax(r.spectrum)
    assert r.frequency[k] == pytest.approx(3 / W)


def test_cutoff_convention():
    """Gain du lissage gaussien = 1/2 à la coupure."""
    fc = 1 / 32
    s = _gaussian_sigma_for_cutoff(fc)
    assert np.exp(-2 * np.pi**2 * s**2 * fc**2) == pytest.approx(0.5)


def test_input_checks():
    img = np.zeros((10, 64))
    with pytest.raises(ValueError):
        measure_equalization_noise(img[:1])
    with pytest.raises(ValueError):
        measure_equalization_noise(img, cutoff=0.6)
