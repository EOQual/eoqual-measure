"""Tests de la mesure de bruit, sur images synthétiques à bruit injecté de variance connue."""
import numpy as np
import pytest

from eoqual.measure import estimate_noise
from eoqual.measure.noise import NoiseEstimate, NoiseVariance, list_noise_methods
from eoqual.measure.synthetic import add_noise, generate_clean_image

TRUE_VARIANCE = 25.0

# Tolérances relatives : les méthodes par sélection de patchs/segments
# sous-estiment de 15 à 35 % sur scène lisse (biais documenté dans
# REFERENCE_TECHNIQUE.md, « Biais résiduel ») ; les estimateurs sans
# sélection sont attendus à ~5 %.
TOLERANCES = {
    "wavelets": 0.05,
    "skimage_estimate_sigma": 0.08,
    "ponomarenko": 0.05,
    "flat_patches": 0.35,
    "adaptive_segmentation": 0.25,
    "pca_estimate": 0.35,
}


@pytest.fixture(scope="module")
def noisy():
    return add_noise(generate_clean_image((256, 256)), a=0.0, b=TRUE_VARIANCE, seed=2)


def test_registry_lists_all_methods():
    assert set(list_noise_methods()) == set(TOLERANCES)


@pytest.mark.parametrize("method", sorted(TOLERANCES))
def test_variance_recovered(noisy, method):
    est = estimate_noise(noisy, method)
    assert isinstance(est, NoiseEstimate)
    variance = float(np.ravel(est.variance)[0])
    assert variance == pytest.approx(TRUE_VARIANCE, rel=TOLERANCES[method])
    assert float(np.ravel(est.sigma)[0]) == pytest.approx(np.sqrt(variance))


def test_ponomarenko_noise_curve_slope():
    image = add_noise(generate_clean_image((256, 256)), a=0.5, b=5.0, seed=2)
    est = estimate_noise(image, "ponomarenko", num_bins=5, overlap=4)
    assert len(est.variance) == len(est.mean) == len(est.samples) == 5
    assert np.all(np.diff(est.mean) > 0)
    slope, _ = np.polyfit(est.mean, est.variance, 1)
    assert slope > 0.2  # bruit dépendant du signal détecté (a = 0.5 injecté)


def test_split_by_luminance_equal_counts():
    parts = NoiseVariance._split_by_luminance(np.arange(10.0)[::-1], 3)
    assert [p.size for p in parts] == [4, 3, 3]
    assert parts[0].tolist() == [9, 8, 7, 6]  # luminances les plus faibles d'abord


def test_bad_input():
    with pytest.raises(ValueError):
        estimate_noise(np.zeros((4, 4, 3)), "wavelets")
    with pytest.raises(ValueError):
        estimate_noise(np.zeros((32, 32)), "nope")
