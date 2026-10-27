"""Validation de `noise_power_spectrum` et `find_structured_noise` contre la vérité du simulateur."""

import numpy as np
import pytest
from conftest import SEED_OFFSET

from eoqual.measure.noise import find_structured_noise, noise_power_spectrum
from eoqual.measure.synthetic import (
    MatrixSensor,
    PeriodicPattern,
    SensorParams,
    apply_kernel,
    gaussian_kernel,
    kernel_power_transfer,
)


def _away_from_dc(r):
    f = r.frequency
    return np.hypot(f[:, None], f[None, :]) >= 3.0 / r.patch_size


# --- NPS ------------------------------------------------------------------

@pytest.mark.parametrize("window", ["hann", "none"])
def test_white_noise_level(window):
    rng = np.random.default_rng(1 + SEED_OFFSET)
    x = rng.normal(0, 2.0, (512, 512))
    r = noise_power_spectrum(x, window=window)
    keep = _away_from_dc(r)
    # Niveau plat σ² ; moyenne sur de nombreuses fréquences (valeurs corrélées par la fenêtre).
    assert r.nps[keep].mean() == pytest.approx(4.0, rel=0.02)
    assert r.variance == pytest.approx(x.var(), rel=0.02)
    assert np.median(r.u_nps[keep] / r.nps[keep]) == pytest.approx(1 / np.sqrt(r.n_patches), rel=0.2)


def test_filtered_noise_follows_transfer_function():
    """Bruit blanc filtré par un noyau connu : NPS = σ²·|H|²."""
    rng = np.random.default_rng(2 + SEED_OFFSET)
    k = gaussian_kernel(1.0)
    y = apply_kernel(rng.normal(0, 2.0, (8, 256, 256)), k)
    r = noise_power_spectrum(y)
    expected = 4.0 * np.fft.fftshift(kernel_power_transfer(k, (64, 64)))
    keep = (expected > 0.05 * expected.max()) & _away_from_dc(r)
    z = (r.nps[keep] - expected[keep]) / r.u_nps[keep]
    assert abs(np.mean(r.nps[keep] / expected[keep]) - 1) < 0.02
    assert np.mean(np.abs(z) < 5) > 0.99


def test_difference_mode_removes_fixed_pattern():
    s = MatrixSensor(SensorParams(seed=70 + SEED_OFFSET, dsnu_pixel=3.0, dsnu_col=1.0))
    t = s.truth
    frames = s.dark(n_frames=8, seed=1).frames
    img = noise_power_spectrum(frames, mode="image")
    diff = noise_power_spectrum(frames, mode="difference")
    assert diff.variance == pytest.approx(t.b, rel=0.03)
    assert img.variance > t.b + 0.8 * 3.0**2
    assert diff.n_images == 4


def test_radial_and_cuts():
    rng = np.random.default_rng(3 + SEED_OFFSET)
    r = noise_power_spectrum(rng.normal(0, 1.0, (512, 512)))
    f, rad = r.radial()
    assert f[0] == 0 and f[-1] == pytest.approx(0.5)
    assert np.all(np.abs(rad[3:] - 1.0) < 0.1)
    assert r.horizontal.shape == r.vertical.shape == (64,)


def test_mask_selects_patches():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1.0, (256, 256))
    x[:, 128:] += rng.normal(0, 5.0, (256, 128))         # moitié droite bruitée
    mask = np.zeros_like(x, dtype=bool)
    mask[:, :128] = True
    r = noise_power_spectrum(x, mask=mask)
    assert r.variance == pytest.approx(1.0, rel=0.05)
    assert r.n_patch_cols == 3


def test_perdecomp_option():
    """Option Moisan (eoqual-core) : disponible ; variance à 3 % près (voir la référence)."""
    pytest.importorskip("eoqual.core.fourier")
    rng = np.random.default_rng(5)
    x = rng.normal(0, 2.0, (512, 512))
    r = noise_power_spectrum(x, window="perdecomp")
    assert r.variance == pytest.approx(4.0, rel=0.03)


def test_nps_input_checks():
    x = np.zeros((64, 64))
    with pytest.raises(ValueError):
        noise_power_spectrum(x, patch_size=4)
    with pytest.raises(ValueError):
        noise_power_spectrum(x, mode="difference")
    with pytest.raises(ValueError):
        noise_power_spectrum(x, window="kaiser")
    with pytest.raises(ValueError):
        noise_power_spectrum(x, mask=np.zeros((10, 10), bool))


# --- Bruits structurés ------------------------------------------------------

def _sensor(**kw):
    return MatrixSensor(SensorParams(shape=(256, 256), seed=71 + SEED_OFFSET, **kw))


def _match(peaks, fx, fy):
    return [q for q in peaks if abs(q.fx - fx) < 0.02 and abs(q.fy - fy) < 0.02]


def test_temporal_pattern():
    s = _sensor(periodic=(PeriodicPattern(0.3, fx=0.125, fy=0.0625),))
    r = find_structured_noise(s.dark(n_frames=16, seed=1).frames)
    hit = _match(r.peaks, 0.125, 0.0625)
    assert len(hit) == 1 and hit[0].kind == "temporal"
    # Dispersion mesurée de l'amplitude sur 30 capteurs : 0,035 DN (8 paires de poses).
    assert hit[0].amplitude == pytest.approx(0.3, abs=5 * 0.035)
    assert hit[0].orientation == pytest.approx(np.degrees(np.arctan2(0.0625, 0.125)), abs=2)


def test_fixed_pattern():
    s = _sensor(periodic=(PeriodicPattern(0.3, fx=0.125, fy=0.0625, temporal=False),))
    r = find_structured_noise(s.dark(n_frames=16, seed=1).frames)
    hit = _match(r.peaks, 0.125, 0.0625)
    assert len(hit) == 1 and hit[0].kind == "fixed"
    assert hit[0].amplitude == pytest.approx(0.3, abs=5 * 0.003 + 0.005)


@pytest.mark.parametrize("kw", [{}, dict(dsnu_col=2.0), dict(dsnu_row=2.0),
                                dict(row_noise=1.0, row_noise_beta=1.0)])
def test_no_false_alarm(kw):
    r = find_structured_noise(_sensor(**kw).dark(n_frames=16, seed=1).frames, false_alarm=1e-5)
    assert r.peaks == []


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_line_noise(beta):
    s = MatrixSensor(SensorParams(shape=(256, 256), seed=72 + SEED_OFFSET, row_noise=1.0, row_noise_beta=beta))
    acq = s.dark(n_frames=16, seed=2)
    r = find_structured_noise(acq.frames)
    truth = acq.row_offsets - acq.row_offsets.mean(axis=0)       # part temporelle (moyenne retirée)
    assert abs(r.rows.sigma - truth.std()) < 5 * r.rows.u_sigma
    assert r.rows.beta is not None and abs(r.rows.beta - beta) < 5 * r.rows.u_beta + 0.05
    assert r.cols.sigma < 5 * r.cols.u_sigma                     # pas de bruit de colonne


def test_single_image_mode():
    s = _sensor(periodic=(PeriodicPattern(1.0, fx=0.2, fy=0.1, temporal=False),))
    r = find_structured_noise(s.dark(n_frames=1, seed=3).frames[0])
    assert set(r.searches) == {"image"}
    hit = _match(r.peaks, 0.2, 0.1)
    assert len(hit) == 1 and hit[0].kind == "image"
    assert hit[0].amplitude == pytest.approx(1.0, rel=0.1)
