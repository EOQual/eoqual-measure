"""Tests de la mesure de MTF par bord incliné, sur mires synthétiques à FTM connue."""
import numpy as np
import pytest

from eoqual.measure import measure_mtf
from eoqual.core.bayer import BAYER_SAMPLING_FACTOR, extract_bayer_planes
from eoqual.measure.mtf import SLANTED_EDGE_ESTIMATORS
from eoqual.measure.mtf.slanted_edge import (
    EdgeOrientationComparator,
    EsfLsfMTFEstimator,
    MTFComparator,
    SlantedEdgePreprocessor,
    bootstrap_mtf,
)
from eoqual.measure.synthetic import SyntheticSlantedEdge


@pytest.fixture(scope="module")
def edge():
    gen = SyntheticSlantedEdge(nx=128, ny=128, angle_deg=5.0, sigma_psf=0.8, noise_level=0.002, seed=0)
    return gen, gen.generate_image().astype(np.float64)


@pytest.mark.parametrize("method", sorted(SLANTED_EDGE_ESTIMATORS))
def test_every_estimator_runs_and_is_physical(edge, method):
    _, image = edge
    res = measure_mtf(image, method)
    mtf = np.abs(res.mtf)
    assert res.frequency.shape == mtf.shape and mtf.size > 10
    assert res.value_at(0.0) == pytest.approx(1.0, abs=0.05)
    assert 0.0 < res.value_at(0.5) < res.value_at(0.25) < 1.0
    # Angle d'inclinaison retrouvé à mieux que 0.5°
    assert abs(abs(np.degrees(res.theta_rad)) - 5.0) < 0.5


def test_spatial_matches_theoretical_mtf(edge):
    gen, image = edge
    theo_quarter, _ = gen.mtf_at_key_frequencies()
    assert measure_mtf(image, "spatial").value_at(0.25) == pytest.approx(theo_quarter, abs=0.05)


def test_frequency_at_inverts_value_at(edge):
    _, image = edge
    res = measure_mtf(image, "spatial")
    f50 = res.frequency_at(0.5)
    assert 0.0 < f50 < 0.5
    assert res.value_at(f50) == pytest.approx(0.5, abs=0.02)
    assert np.isnan(res.frequency_at(-1.0))


def test_unknown_method_raises(edge):
    with pytest.raises(ValueError):
        measure_mtf(edge[1], "nope")


@pytest.mark.parametrize("orientation, expected", [("vertical", 85.0), ("horizontal", 5.0)])
def test_orientation_detectors_agree(orientation, expected):
    gen = SyntheticSlantedEdge(nx=128, ny=128, angle_deg=5.0, orientation=orientation, sigma_psf=0.8,
                               noise_level=0.002, seed=0)
    results = EdgeOrientationComparator().compare(gen.generate_image().astype(np.float64))
    for name in ("gradient", "fourier"):
        assert "error" not in results[name].details
        assert abs(results[name].angle_deg) == pytest.approx(expected, abs=0.3)


def test_comparator_and_esf_models(edge):
    _, image = edge
    frame = SlantedEdgePreprocessor().preprocess(image)
    results = MTFComparator().compare(frame)
    assert set(results) == {"spatial", "spectral", "esf_lsf", "spectral_correlation"}
    assert all("error" not in r.details for r in results.values())
    for model in ("numeric", "sigmoid", "pchip", "tikhonov"):
        res = EsfLsfMTFEstimator(esf_model=model).estimate(frame)
        assert np.isfinite(res.details["fwhm"]) and np.isfinite(res.details["rer"])


def test_bootstrap(edge):
    _, image = edge
    frame = SlantedEdgePreprocessor().preprocess(image)
    boot = bootstrap_mtf(EsfLsfMTFEstimator(), frame, n_bootstrap=4, seed=0)
    assert boot.n_successful == 4
    # NaN là où la grille commune sort de la plage couverte par les répliques.
    finite = np.isfinite(boot.mtf_ci_low) & np.isfinite(boot.mtf_ci_high)
    assert finite.sum() > 0.5 * finite.size
    assert np.all(boot.mtf_ci_low[finite] <= boot.mtf_ci_high[finite] + 1e-12)


def test_bayer_planes_and_pixel_pitch(edge):
    _, image = edge
    planes = extract_bayer_planes(image)
    assert set(planes) == {"R", "G1", "G2", "B", "G"}
    assert planes["G"].shape == (64, 64)
    full = measure_mtf(image, "spatial")
    half = measure_mtf(planes["G"], "spatial", pixel_pitch=BAYER_SAMPLING_FACTOR)
    assert half.frequency.max() == pytest.approx(full.frequency.max() * BAYER_SAMPLING_FACTOR, rel=0.05)
