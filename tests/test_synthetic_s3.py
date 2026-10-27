"""Vérification des effets S3 : bruit de ligne, motifs périodiques, rayons cosmiques, traitements."""

import numpy as np
import pytest
from conftest import SEED_OFFSET

from eoqual.measure.synthetic import (
    MatrixSensor,
    PeriodicPattern,
    PushbroomSensor,
    SensorParams,
    apply_kernel,
    gaussian_kernel,
    kernel_power_transfer,
)


def _slope(x):
    p = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(x.size)
    keep = (f > 0.002) & (f < 0.4)
    return np.polyfit(np.log(f[keep]), np.log(p[keep]), 1)[0]


def test_effects_off_by_default(sensor):
    acq = sensor.dark(n_frames=2, seed=0)
    assert acq.row_offsets is None and acq.periodic is None and acq.cosmic_mask is None
    assert sensor.truth.correlated_variance == 0.0


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_row_noise(beta):
    k = 32
    s = MatrixSensor(SensorParams(seed=60 + SEED_OFFSET, row_noise=2.0, row_noise_beta=beta))
    t = s.truth
    acq = s.dark(n_frames=k, seed=1)
    h, w = t.params.shape
    assert acq.row_offsets.shape == (k, h)
    assert acq.row_offsets.std() == pytest.approx(2.0)
    resid = (acq.frames - t.offset_map).mean(axis=2)                 # moyennes de lignes, par pose
    err = resid - acq.row_offsets
    assert err.std() == pytest.approx(np.sqrt(t.b / w), rel=0.05)
    # Variance temporelle par pixel = b + variance, d'une pose à l'autre, de l'offset de sa ligne
    # (exacte pour la réalisation ; en 1/f, plus petite que σ_ligne² : la part basse fréquence
    # varie peu entre poses voisines).
    var = acq.frames.astype(float).var(axis=0, ddof=1).mean()
    expected = t.b + acq.row_offsets.var(axis=0, ddof=1).mean()
    assert var == pytest.approx(expected, rel=5 * np.sqrt(2 / ((k - 1) * h * w)) + 0.01)
    if beta == 0:
        assert expected == pytest.approx(t.b + t.correlated_variance, rel=5 * np.sqrt(2 / (k * h)))


def test_row_noise_spectrum():
    s = MatrixSensor(SensorParams(shape=(512, 64), seed=61 + SEED_OFFSET, row_noise=1.0, row_noise_beta=1.0))
    seq = s.dark(n_frames=16, seed=2).row_offsets.ravel()            # séquence continue K·H
    assert _slope(seq) == pytest.approx(-1.0, abs=0.15)


def test_periodic_temporal_and_fixed():
    pats = (PeriodicPattern(3.0, fx=0.125, fy=0.0, temporal=True),
            PeriodicPattern(1.5, fx=0.0, fy=0.25, temporal=False, phase=0.3))
    s = MatrixSensor(SensorParams(shape=(128, 128), seed=62 + SEED_OFFSET, periodic=pats))
    t = s.truth
    acq = s.dark(n_frames=8, seed=3)
    i, j = np.mgrid[0:128, 0:128]
    for f in range(8):
        x = acq.frames[f] - t.offset_map
        # Projection sur le motif 1 : amplitude 3 quelle que soit la phase.
        c = np.mean(x * np.cos(2 * np.pi * 0.125 * j)) * 2
        sn = np.mean(x * np.sin(2 * np.pi * 0.125 * j)) * 2
        assert np.hypot(c, sn) == pytest.approx(3.0, abs=5 * np.sqrt(2 * t.b / x.size) * 2)
    phases = [np.angle(np.sum(acq.periodic[f] * np.exp(-2j * np.pi * 0.125 * j))) for f in range(8)]
    assert np.std(np.unwrap(phases)) > 0.3                           # phase temporelle
    # Motif fixe : sa projection est identique d'une pose à l'autre (le motif temporel, orthogonal
    # sur un nombre entier de périodes, n'y contribue pas).
    part2 = np.array([np.sum(acq.periodic[f] * np.cos(2 * np.pi * 0.25 * i + 0.3)) for f in range(8)])
    assert np.ptp(part2) < 1e-6 * abs(part2[0])
    assert t.correlated_variance == pytest.approx(3.0**2 / 2)


def test_cosmic_rays():
    s = MatrixSensor(SensorParams(shape=(128, 128), seed=63 + SEED_OFFSET, cosmic_rate=5.0,
                                  cosmic_charge=3000.0, read_noise=0.0))
    t = s.truth
    acq = s.dark(n_frames=200, seed=4)
    mask = acq.cosmic_mask
    hit = acq.frames[mask].astype(float) - np.broadcast_to(t.offset_map, mask.shape)[mask]
    assert np.all(np.abs(hit - 3000.0 / t.gain) < 1.0)                # charge/K, à l'arrondi près
    # Nombre de pixels touchés par pose : traces de 1 à 8 px, ~5 traces par pose.
    per_frame = mask.reshape(200, -1).sum(axis=1)
    assert 5 < per_frame.mean() < 5 * 9
    clipped = MatrixSensor(SensorParams(shape=(64, 64), seed=1, cosmic_rate=20.0, cosmic_charge=1e6))
    a = clipped.dark(n_frames=2, seed=5)
    peak = a.frames[a.cosmic_mask].astype(float) - np.broadcast_to(clipped.truth.offset_map, a.cosmic_mask.shape)[a.cosmic_mask]
    assert np.all(peak <= clipped.params.full_well / clipped.truth.gain + 1)   # écrêtage au puits


def test_pushbroom_line_noise():
    pb = PushbroomSensor(SensorParams(shape=(1, 512), seed=64 + SEED_OFFSET, row_noise=1.5,
                                      dsnu_pixel=0.0, prnu_pixel=0.0))
    t = pb.truth
    acq = pb.dark(n_lines=2000, seed=6)
    assert acq.line_offsets.shape == (2000,)
    err = (acq.image - t.detector_offset).mean(axis=1) - acq.line_offsets
    assert err.std() == pytest.approx(np.sqrt(t.b / 512), rel=0.05)


def test_apply_kernel_nps():
    """Bruit blanc filtré : NPS moyen = σ²·|H|² ; variance = σ²·Σh²."""
    rng = np.random.default_rng(7 + SEED_OFFSET)
    k = gaussian_kernel(1.0)
    stack = rng.normal(0, 2.0, (200, 64, 64))
    y = apply_kernel(stack, k)
    nps = np.mean(np.abs(np.fft.fft2(y)) ** 2, axis=0) / (64 * 64)
    h2 = kernel_power_transfer(k, (64, 64))
    keep = h2 > 1e-3
    np.testing.assert_allclose(nps[keep] / (4.0 * h2[keep]), 1.0, atol=5 / np.sqrt(200))
    assert y.var() == pytest.approx(4.0 * np.sum(k**2), rel=0.02)
    assert h2.mean() == pytest.approx(np.sum(k**2))
    # Noyau centré : pas de décalage de l'image.
    delta = np.zeros((32, 32)); delta[10, 12] = 1
    assert np.unravel_index(np.argmax(apply_kernel(delta, k)), (32, 32)) == (10, 12)


def test_s3_params_validation():
    with pytest.raises(ValueError):
        PeriodicPattern(1.0, fx=0.7, fy=0.0)
    with pytest.raises(ValueError):
        SensorParams(periodic=("x",))
    with pytest.raises(ValueError):
        SensorParams(row_noise=-1)
