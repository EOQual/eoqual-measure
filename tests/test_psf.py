"""Tests de la mesure de Strehl et d'énergie encerclée, sur PSF synthétiques à Strehl exact."""
import numpy as np
import pytest

from eoqual.measure import measure_psf
from eoqual.measure.synthetic import SyntheticPSF

CASES = [
    {},
    {"offset": (0.37, -1.4)},
    {"aberrations_rms_waves": {"defocus": 0.05}},
    {"aberrations_rms_waves": {"coma": 0.07, "astigmatism": 0.04}, "offset": (0.5, 0.5)},
    {"aberrations_rms_waves": {"spherical": 0.06}, "sampling": 2.0},
    {"aberrations_rms_waves": {"defocus": 0.15}},
    {"aberrations_rms_waves": {"defocus": 0.05}, "photons": 1e6, "background": 20, "read_noise": 5},
]


@pytest.mark.parametrize("case", CASES)
def test_strehl_matches_truth(case):
    gen = SyntheticPSF(**case)
    res = measure_psf(gen.generate(), sampling=gen.sampling)
    assert res.strehl == pytest.approx(gen.strehl_true, rel=0.005)


def test_marechal_consistency():
    gen = SyntheticPSF(aberrations_rms_waves={"defocus": 0.03})
    assert gen.strehl_true == pytest.approx(gen.strehl_marechal, rel=0.002)


def test_encircled_energy_and_radii():
    gen = SyntheticPSF(shape=(257, 257))
    res = measure_psf(gen.generate(), sampling=gen.sampling)
    # 0.838 au premier anneau noir pour une pupille parfaite ; l'énergie hors
    # champ (normalisation à l'énergie de l'imagette) relève légèrement la valeur.
    assert res.ee_first_dark_ring == pytest.approx(0.838, abs=0.015)
    # Quasi croissante : après retrait du fond, les pixels lointains peuvent être
    # légèrement négatifs (fluctuations de ~1e-5).
    assert np.all(np.diff(res.encircled_energy) >= -1e-4)
    assert 0 < res.r50 < res.r80
    worse = measure_psf(SyntheticPSF(shape=(257, 257), aberrations_rms_waves={"coma": 0.1}).generate(), sampling=4.0)
    assert worse.r80 > res.r80 and worse.ee_first_dark_ring < res.ee_first_dark_ring


def test_centroid_and_background():
    gen = SyntheticPSF(offset=(2.3, -1.7), photons=1e6, background=30.0, seed=1)
    res = measure_psf(gen.generate(), sampling=gen.sampling)
    assert res.centroid == pytest.approx((64 + 2.3, 64 - 1.7), abs=0.05)
    assert res.background == pytest.approx(30.0, abs=1.0)
    assert res.psf.sum() == pytest.approx(1.0)


def test_without_sampling_and_errors():
    res = measure_psf(SyntheticPSF().generate())
    assert np.isnan(res.strehl) and np.isnan(res.ee_first_dark_ring) and res.reference is None
    with pytest.warns(UserWarning):
        measure_psf(SyntheticPSF(sampling=1.5, pupil_pixels=128).generate(), sampling=1.5)
    with pytest.raises(ValueError):
        measure_psf(np.zeros((32, 32, 3)))
    with pytest.raises(ValueError):
        SyntheticPSF(shape=(129, 129), sampling=0.5).generate()
