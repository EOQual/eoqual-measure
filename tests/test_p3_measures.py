"""Validation des mesures P3 : pixels télégraphiques, histogramme du convertisseur, CNR ; whisk broom."""

import numpy as np
import pytest
from conftest import SEED_OFFSET

from eoqual.measure.contrast import measure_cnr
from eoqual.measure.radiometry import adc_histogram, band_structure, find_rts_pixels, measure_spatial_noise
from eoqual.measure.synthetic import MatrixSensor, SensorParams, WhiskbroomSensor, apply_kernel, gaussian_kernel


# --- Pixels télégraphiques ----------------------------------------------------

@pytest.fixture(scope="module")
def rts_case():
    s = MatrixSensor(SensorParams(shape=(128, 128), seed=130 + SEED_OFFSET, rts_fraction=3e-3,
                                  rts_amplitude=(2.0, 10.0), rts_rate=(0.02, 0.2)))
    acq = s.dark(n_frames=2000, seed=1)
    return s.truth, acq, find_rts_pixels(acq.frames)


def test_rts_detection(rts_case):
    t, _, r = rts_case
    snr = t.rts.amplitude / np.sqrt(t.b)
    assert not np.any(~np.isin(r.index, t.rts.index))                    # aucune fausse détection
    strong = t.rts.index[snr >= 2.5]
    assert np.all(np.isin(strong, r.index))                              # tous les sauts ≥ 2,5 σ


def test_rts_parameters(rts_case):
    t, acq, r = rts_case
    true_i = np.searchsorted(t.rts.index, r.index)
    amp = t.rts.amplitude[true_i]
    sigma = np.sqrt(t.b)
    # Amplitude : différence de deux moyennes de niveaux, incertitude ≈ σ·√(1/n_bas + 1/n_haut).
    n_hi = r.states.sum(axis=0)
    n_lo = r.states.shape[0] - n_hi
    u_amp = sigma * np.sqrt(1 / n_hi + 1 / n_lo)
    assert np.all(np.abs(r.amplitude - amp) < 5 * u_amp + 0.05 * amp)
    # Bascules : fiables pour un saut ≥ 4 σ (Viterbi manque des séjours courts en dessous).
    good = amp / sigma >= 4
    n_up = np.sum(~r.states[:-1] & r.states[1:], axis=0)
    rel = np.abs(r.p_up / t.rts.p_up[true_i] - 1)
    assert np.all(rel[good] < 5 / np.sqrt(n_up[good]) + 0.05)


def test_no_rts_on_clean_sensor():
    s = MatrixSensor(SensorParams(shape=(96, 96), seed=131 + SEED_OFFSET))
    r = find_rts_pixels(s.dark(n_frames=1000, seed=2).frames)
    assert r.index.size == 0


# --- Histogramme du convertisseur ---------------------------------------------

def _adc_frames(ramp, seed_offset=0, n_frames=16):
    s = MatrixSensor(SensorParams(shape=(256, 256), seed=132 + SEED_OFFSET + seed_offset, bit_depth=10,
                                  adc_dnl=0.15, adc_missing_codes=4, read_noise=30.0, conversion_gain=1.0,
                                  offset=50.0, dsnu_pixel=0.0, dsnu_col=0.0, prnu_pixel=0.0, prnu_col=0.0))
    t = s.truth
    scene = t.photons_for_signal(1.0) * np.asarray(ramp)[None, :].repeat(256, axis=0)
    return t, s.expose(scene, n_frames=n_frames, seed=1).frames


@pytest.mark.parametrize("ramp", [np.linspace(100, 800, 256), np.linspace(150, 900, 256)],
                         ids=["sans écrêtage", "avec écrêtage"])
def test_adc_dnl_local(ramp):
    t, frames = _adc_frames(ramp)
    r = adc_histogram(frames, bit_depth=10, density="local")
    z = (r.dnl - t.adc.dnl[r.codes]) / r.u_dnl
    assert abs(z.mean()) < 5 / np.sqrt(z.size) + 0.05
    assert z.std() == pytest.approx(1.0, abs=0.15)
    in_range = t.adc.missing[(t.adc.missing >= r.codes.min()) & (t.adc.missing <= r.codes.max())]
    np.testing.assert_array_equal(r.missing, in_range)


def test_adc_uniform_ramp_gives_inl():
    t, frames = _adc_frames(np.linspace(50, 900, 256), n_frames=128)
    r = adc_histogram(frames, bit_depth=10, density="uniform")
    z = (r.dnl - t.adc.dnl[r.codes]) / r.u_dnl
    assert z.std() == pytest.approx(1.0, abs=0.15)
    true = np.cumsum(t.adc.dnl[r.codes])
    true -= np.linspace(true[0], true[-1], true.size)
    true -= true.mean()
    # INL : somme cumulée de n erreurs de 1/√n̂, droite des extrémités retirée :
    # pont brownien d'écart-type ≤ √(n/(4n̂)) (au milieu de la plage).
    bridge = np.sqrt(r.codes.size / (4 * r.expected.mean()))
    assert np.abs(r.inl - true).max() < 4 * bridge
    assert np.ptp(true) > 2 * 4 * bridge                                  # INL résolue


def test_adc_checks():
    with pytest.raises(ValueError):
        adc_histogram(np.array([0.5, 1.5]))
    with pytest.raises(ValueError):
        adc_histogram(np.zeros(100, dtype=int), density="sine")


# --- CNR ----------------------------------------------------------------------

def _target(seed, contrast_dn=100.0):
    s = MatrixSensor(SensorParams(seed=seed, prnu_pixel=0.01))
    t = s.truth
    h, w = t.params.shape
    i, j = np.mgrid[0:h, 0:w]
    disk = (i - 128) ** 2 + (j - 128) ** 2 < 60**2
    scene = np.where(disk, t.photons_for_signal(500.0 + contrast_dn), t.photons_for_signal(500.0))
    scene = apply_kernel(scene, gaussian_kernel(1.5))
    return s, t, scene, disk


def _true_cnr(t, scene, b, d):
    em, ev = t.expected_mean(scene), t.expected_variance(scene)
    mb, vb = em[b].mean(), ev[b].mean() + em[b].var()
    md, vd = em[d].mean(), ev[d].mean() + em[d].var()
    return (mb - md) / np.sqrt(vb + vd)


def test_cnr_detected_zones():
    s, t, scene, disk = _target(133 + SEED_OFFSET)
    r = measure_cnr(s.expose(scene, n_frames=20, seed=1).frames)
    assert abs(r.cnr - _true_cnr(t, scene, r.bright_mask, r.dark_mask)) < 5 * r.u_cnr
    assert not np.any(r.bright_mask & ~disk) and not np.any(r.dark_mask & disk)


def test_cnr_given_masks_and_uncertainty():
    s, t, scene, disk = _target(134 + SEED_OFFSET, contrast_dn=30.0)
    i, j = np.mgrid[0:256, 0:256]
    b = (i - 128) ** 2 + (j - 128) ** 2 < 50**2
    d = (i - 128) ** 2 + (j - 128) ** 2 > 70**2
    frames = s.expose(scene, n_frames=40, seed=2).frames
    r = measure_cnr(frames, bright_mask=b, dark_mask=d)
    assert r.threshold is None
    assert abs(r.cnr - _true_cnr(t, scene, b, d)) < 5 * r.u_cnr
    # Incertitude analytique d'une image ≈ dispersion entre images.
    single = [measure_cnr(f, bright_mask=b, dark_mask=d) for f in frames[:2]]
    assert single[0].u_cnr == pytest.approx(r.cnr_per_image.std(ddof=1), rel=5 / np.sqrt(2 * 39) + 0.1)
    with pytest.raises(ValueError):
        measure_cnr(frames[0], bright_mask=b, dark_mask=b)


# --- Whisk broom --------------------------------------------------------------

def test_whiskbroom_spatial_noise():
    w = WhiskbroomSensor(SensorParams(shape=(1, 10), seed=135 + SEED_OFFSET, dsnu_pixel=0.0, dsnu_col=1.0,
                                      prnu_pixel=0.0, prnu_col=0.01), scan_offset=0.8, scan_gain=0.002)
    t = w.truth
    a = w.expose(t.photons_for_signal(1500.0), n_scans=60, width=512, seed=2)
    r = measure_spatial_noise(a.image, layout="whiskbroom", period=10)
    fb_true = a.expected_profile[a.backward].mean() - a.expected_profile[~a.backward].mean()
    assert abs(r.forward_backward - fb_true) < 5 * r.u_forward_backward
    exact = band_structure(a.expected_profile, 10)
    noise = np.sqrt((t.a * 1500 + t.b) / 512)
    assert abs(r.bands.sigma - exact.sigma) < 5 * noise / np.sqrt(60) + 0.01 * exact.sigma
    assert abs(r.spatial_sigma - a.expected_profile.std()) < 5 * r.u_spatial_sigma
    with pytest.raises(ValueError):
        measure_spatial_noise(a.image, layout="whiskbroom")
