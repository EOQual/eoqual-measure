"""
Variance du bruit par la pépite du variogramme (`variogram_noise`).

Scènes simulées à bruit connu ; tolérances de 5 u. Voir
REFERENCE_TECHNIQUE.md, partie IV, chapitre 14, et annexe B, §B.14.
"""

import os

import numpy as np
import pytest
from scipy import ndimage

from eoqual.measure.noise import variogram, variogram_noise
from eoqual.measure.radiometry import snr_at_levels
from eoqual.measure.synthetic.scene import natural_scene

SEED = int(os.environ.get("EOQUAL_TEST_SEED", "0"))
SIGMA = 2.0


def smooth_scene(seed=1, shape=(512, 512), contrast=0.005, beta=4.0):
    """Scène lisse (fBm β = 4) : sa contribution à γ(1) est négligeable."""
    return natural_scene(shape, 1000.0, np.random.default_rng(seed + SEED),
                         contrast=contrast, beta=beta).photons


def noisy(scene, sigma=SIGMA, seed=0):
    return scene + np.random.default_rng(100 + seed + SEED).normal(0.0, sigma, scene.shape)


def test_white_noise_recovered():
    r = variogram_noise(noisy(smooth_scene()))
    assert abs(r.variance - SIGMA**2) < 5 * r.u_variance
    assert r.reliable and r.nugget_fraction > 0.95
    assert abs(r.anisotropy) < 5
    assert abs(r.snr - r.mean / SIGMA) < 5 * r.u_snr


@pytest.mark.parametrize("model", ["linear", "quadratic"])
def test_no_selection_bias_on_pure_noise(model):
    img = 500.0 + np.random.default_rng(7 + SEED).normal(0.0, SIGMA, (512, 512))
    r = variogram_noise(img, model=model, keep_fraction=0.25)
    assert abs(r.variance - SIGMA**2) < 5 * r.u_variance


def test_uncertainty_matches_monte_carlo():
    """u(σ²) annoncé = dispersion réelle des estimations (K tirages du bruit)."""
    scene = smooth_scene(shape=(256, 256))
    k = 100
    est, u = [], []
    for i in range(k):
        r = variogram_noise(noisy(scene, seed=1000 + i), tile=32)
        est.append(r.variance)
        u.append(r.u_variance)
    ratio = np.std(est, ddof=1) / np.mean(u)
    # incertitude relative d'un écart-type estimé sur K tirages : 1/√(2(K − 1))
    assert abs(ratio - 1.0) < 5 / np.sqrt(2 * (k - 1))


def test_variogram_of_white_noise_is_flat():
    """Sans sélection, le variogramme corrigé du retrait du plan est plat.

    (La sélection des tuiles lisses creuse la courbe aux grandes distances,
    −1,7 % à h = 8 : voir le chapitre 14.)
    """
    img = 500.0 + np.random.default_rng(3 + SEED).normal(0.0, SIGMA, (512, 512))
    v = variogram(img, max_lag=8, keep_fraction=1.0)
    assert np.all(np.abs(v.gamma_rows - SIGMA**2) < 5 * v.u_gamma_rows)
    assert np.all(np.abs(v.gamma_cols - SIGMA**2) < 5 * v.u_gamma_cols)


def test_signal_dependent_noise_model_and_snr():
    """Tuiles uniformes à des niveaux différents : modèle σ² = a + b·μ retrouvé."""
    a_true, b_true, t = 4.0, 0.02, 64
    levels = np.random.default_rng(5 + SEED).uniform(100.0, 1500.0, (12, 12))
    scene = np.kron(levels, np.ones((t, t)))
    rng = np.random.default_rng(6 + SEED)
    img = scene + rng.normal(0.0, 1.0, scene.shape) * np.sqrt(a_true + b_true * scene)
    r = variogram_noise(img, tile=t, keep_fraction=1.0)
    u_a, u_b = np.sqrt(np.diag(r.cov))
    assert abs(r.a - a_true) < 5 * u_a
    assert abs(r.b - b_true) < 5 * u_b
    s = snr_at_levels(r, [200.0, 1000.0])
    expected = np.array([200.0, 1000.0]) / np.sqrt(a_true + b_true * np.array([200.0, 1000.0]))
    assert np.all(np.abs(s.snr - expected) < 5 * s.u_snr)


def test_correlated_noise_is_flagged():
    """Bruit filtré (image traitée) : pas de pépite, mesure déclarée non fiable."""
    img = ndimage.gaussian_filter(noisy(smooth_scene()), 0.8)
    r = variogram_noise(img)
    assert not r.reliable
    assert r.nugget_fraction < 0.2


def test_column_stripes_raise_anisotropy():
    img = noisy(smooth_scene())
    stripes = np.random.default_rng(9 + SEED).normal(0.0, 1.5, img.shape[1])
    r = variogram_noise(img + stripes[None, :])
    assert r.anisotropy > 5 and not r.reliable
    # le long des colonnes : bruit seul ; le long des lignes : bruit + pépite
    # des rayures, calculée sur le tirage lui-même, tuile par tuile (les
    # tuiles d'une même colonne partagent les mêmes rayures)
    def stripe_nugget(c):
        s = stripes[c:c + r.tile]
        g1, g2 = (0.5 * np.mean((s[h:] - s[:-h]) ** 2) for h in (1, 2))
        return 2 * g1 - g2

    expected = SIGMA**2 + np.mean([stripe_nugget(c) for _, c in r.tile_corner])
    assert abs(r.variance_cols - SIGMA**2) < 5 * r.u_variance_cols
    assert abs(r.variance_rows - expected) < 5 * r.u_variance_rows


def test_saturated_tiles_excluded():
    img = noisy(smooth_scene())
    img[:64, :64] = 4095.0
    r = variogram_noise(img, saturation=4095.0, keep_fraction=1.0)
    total = (img.shape[0] // r.tile) * (img.shape[1] // r.tile)
    excluded = (64 // r.tile) ** 2
    assert r.n_valid == total - excluded and r.n_tiles == r.n_valid


def test_errors():
    with pytest.raises(ValueError):
        variogram_noise(np.zeros((8, 8)))
    with pytest.raises(ValueError):
        variogram_noise(np.zeros((128, 128, 3)))
    with pytest.raises(ValueError):
        variogram_noise(np.zeros((128, 128)), model="cubic")
    with pytest.raises(ValueError):
        variogram_noise(np.zeros((128, 128)), keep_fraction=0.0)
    with pytest.raises(ValueError):
        variogram(np.zeros((128, 128)), max_lag=40)
