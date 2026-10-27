"""Vérification de la fin de S4 : pixels télégraphiques, convertisseur à pas inégaux, whisk broom."""

import numpy as np
import pytest
from conftest import SEED_OFFSET

from eoqual.measure.synthetic import MatrixSensor, PushbroomSensor, SensorParams, WhiskbroomSensor
from eoqual.measure.synthetic.response import adc_transfer
from eoqual.measure.synthetic.patterns import pattern_rng


# --- Pixels télégraphiques ----------------------------------------------------

def test_rts_levels_and_rates():
    s = MatrixSensor(SensorParams(shape=(128, 128), seed=120 + SEED_OFFSET, rts_fraction=2e-3,
                                  rts_amplitude=(5.0, 10.0), rts_rate=(0.05, 0.2)))
    t = s.truth
    r = t.rts
    k = 4000
    acq = s.dark(n_frames=k, seed=1)
    st = acq.rts_states
    assert st.shape == (k, r.index.size) and r.mask.sum() == r.index.size
    x = acq.frames.reshape(k, -1)[:, r.index].astype(float) - t.offset_map.ravel()[r.index]
    # Niveau haut − niveau bas = amplitude, à l'incertitude près.
    hi = np.array([x[st[:, i], i].mean() for i in range(r.index.size)])
    lo = np.array([x[~st[:, i], i].mean() for i in range(r.index.size)])
    np.testing.assert_allclose(hi - lo, r.amplitude, atol=0.3)
    # Fréquences de bascule et fraction de temps à l'état haut.
    # Incertitude relative d'une fréquence estimée sur n bascules : ≈ 1/√n.
    n_up = np.sum(~st[:-1] & st[1:], axis=0)
    n_down = np.sum(st[:-1] & ~st[1:], axis=0)
    up = n_up / np.sum(~st[:-1], axis=0)
    down = n_down / np.sum(st[:-1], axis=0)
    assert np.all(np.abs(up / r.p_up - 1) < 5 / np.sqrt(n_up))
    assert np.all(np.abs(down / r.p_down - 1) < 5 / np.sqrt(n_down))
    assert np.mean(st) == pytest.approx(np.mean(r.high_fraction), abs=0.05)


def test_rts_pushbroom_along_lines():
    pb = PushbroomSensor(SensorParams(shape=(1, 512), seed=121 + SEED_OFFSET, rts_fraction=0.02,
                                      dsnu_pixel=0.0, prnu_pixel=0.0))
    acq = pb.dark(n_lines=3000, seed=2)
    assert acq.rts_states.shape == (3000, pb.truth.rts.index.size)


# --- Convertisseur à pas inégaux ----------------------------------------------

def test_ideal_adc_equals_rounding():
    adc = adc_transfer(10, 0.0, 0, pattern_rng(0, "adc"))
    x = np.random.default_rng(1).uniform(-2, 1030, 10000)
    np.testing.assert_array_equal(np.clip(adc.quantize(x), 0, 1023), np.clip(np.floor(x + 0.5), 0, 1023))
    np.testing.assert_allclose(adc.widths[1:-1], 1.0)


def test_unequal_steps_and_missing_codes():
    s = MatrixSensor(SensorParams(shape=(256, 256), seed=122 + SEED_OFFSET, bit_depth=10,
                                  adc_dnl=0.2, adc_missing_codes=5, read_noise=40.0, conversion_gain=1.0,
                                  offset=100.0, dsnu_pixel=0.0, dsnu_col=0.0))
    adc = s.truth.adc
    assert adc.missing.size == 5 and np.all(adc.widths[adc.missing] == 0)
    assert np.all(np.diff(adc.thresholds) >= 0)
    # Rampe lente : chaque code reçoit une part proportionnelle à sa largeur.
    ramp = np.linspace(200, 800, 256)[None, :].repeat(256, axis=0)
    frames = s.expose(s.truth.photons_for_signal(1.0) * ramp, n_frames=8, seed=3).frames
    counts = np.bincount(frames.ravel(), minlength=1024).astype(float)
    assert np.all(counts[adc.missing] == 0)
    codes = np.arange(400, 800)
    density = counts[codes] / adc.widths[codes].clip(1e-9)
    ok = adc.widths[codes] > 0.3
    # Densité par unité de largeur ≈ constante (rampe lisse, bruit de 40 DN qui lisse l'histogramme).
    assert np.std(density[ok]) / np.mean(density[ok]) < 0.1
    assert adc.inl.size == 1024


# --- Whisk broom --------------------------------------------------------------

def test_whiskbroom_bands_and_scan_direction():
    w = WhiskbroomSensor(SensorParams(shape=(1, 10), seed=123 + SEED_OFFSET, dsnu_pixel=0.0, dsnu_col=1.0,
                                      prnu_pixel=0.0, prnu_col=0.01), scan_offset=0.8, scan_gain=0.002)
    t = w.truth
    a = w.expose(t.photons_for_signal(1500.0), n_scans=40, width=512, seed=4)
    assert a.image.shape == (400, 512)
    np.testing.assert_array_equal(a.detector, np.arange(400) % 10)
    prof = a.image.mean(axis=1)
    sigma = np.sqrt((t.a * 1500 + t.b) / 512)
    assert np.std(prof - a.expected_profile) == pytest.approx(sigma, rel=0.15)
    diff = prof[a.backward].mean() - prof[~a.backward].mean()
    assert diff == pytest.approx(0.8 + 1500 * 0.002, abs=5 * sigma * np.sqrt(2 / 200))
    with pytest.raises(ValueError):
        WhiskbroomSensor(SensorParams(shape=(2, 10)))
    with pytest.raises(ValueError):
        w.expose(1.0, seed=0)


def test_s4_rest_validation():
    with pytest.raises(ValueError):
        SensorParams(rts_fraction=2.0)
    with pytest.raises(ValueError):
        SensorParams(rts_rate=(0.5, 0.1))
    with pytest.raises(ValueError):
        SensorParams(adc_dnl=-1.0)
