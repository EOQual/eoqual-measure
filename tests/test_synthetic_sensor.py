"""
Vérification du simulateur S0 contre sa propre vérité.

Les tolérances sont fixées à partir de l'incertitude statistique de
chaque estimateur (en général 4 à 5 σ), pas au jugé.
"""

import numpy as np
import pytest
from conftest import SEED_OFFSET

from eoqual.measure.synthetic import MatrixSensor, SensorParams


# --- Reproductibilité et séparation motifs fixes / bruit temporel ----------

def test_same_seed_same_sensor():
    a = MatrixSensor(SensorParams(seed=3))
    b = MatrixSensor(SensorParams(seed=3))
    np.testing.assert_array_equal(a.truth.offset_map, b.truth.offset_map)
    np.testing.assert_array_equal(a.truth.gain_map, b.truth.gain_map)


def test_other_seed_other_sensor():
    a = MatrixSensor(SensorParams(seed=3))
    b = MatrixSensor(SensorParams(seed=4))
    assert not np.array_equal(a.truth.offset_map, b.truth.offset_map)


def test_temporal_seed(sensor):
    x1 = sensor.dark(n_frames=2, seed=10).frames
    x2 = sensor.dark(n_frames=2, seed=10).frames
    x3 = sensor.dark(n_frames=2, seed=11).frames
    np.testing.assert_array_equal(x1, x2)
    assert not np.array_equal(x1, x3)


def test_fixed_pattern_does_not_depend_on_other_patterns():
    """Changer la PRNU ne doit pas changer la DSNU (flux aléatoires séparés)."""
    a = MatrixSensor(SensorParams(seed=5, prnu_pixel=0.01))
    b = MatrixSensor(SensorParams(seed=5, prnu_pixel=0.02))
    np.testing.assert_array_equal(a.truth.offset_map, b.truth.offset_map)


# --- Motifs fixes ----------------------------------------------------------

def test_maps_are_centred(structured_sensor):
    t = structured_sensor.truth
    assert t.offset_map.mean() == pytest.approx(t.params.offset, abs=1e-9)
    assert t.gain_map.mean() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("which", ["dsnu", "prnu"])
def test_component_amplitudes(structured_sensor, which):
    t = structured_sensor.truth
    p = t.params
    comp = getattr(t, which)
    h, w = p.shape
    for name, n in (("pixel", h * w), ("row", h), ("col", w)):
        expected = getattr(p, f"{which}_{name}")
        rel = 5.0 / np.sqrt(2 * (n - 1))   # 5 σ sur un écart-type estimé
        assert getattr(comp, name).std() == pytest.approx(expected, rel=rel)


# --- Statistiques temporelles ---------------------------------------------

def test_dark_stack_mean_and_variance(sensor):
    t = sensor.truth
    k = 64
    frames = sensor.dark(n_frames=k, seed=1).frames.astype(np.float64)
    n = frames[0].size
    # Moyenne temporelle par pixel = carte d'offset, à √(b/K) près.
    resid = frames.mean(axis=0) - t.offset_map
    sigma_mean = np.sqrt(t.b / k)
    assert abs(resid.mean()) < 5 * sigma_mean / np.sqrt(n)
    assert resid.std() == pytest.approx(sigma_mean, rel=0.02)
    # Variance temporelle (1/(K − 1)) = b.
    var = frames.var(axis=0, ddof=1).mean()
    assert var == pytest.approx(t.b, rel=5 * np.sqrt(2 / ((k - 1) * n)) + 0.005)


@pytest.mark.parametrize("signal_dn", [50.0, 500.0, 2500.0])
def test_flat_mean_and_variance(sensor, signal_dn):
    t = sensor.truth
    k = 32
    photons = t.photons_for_signal(signal_dn)
    frames = sensor.expose(photons, n_frames=k, seed=2).frames.astype(np.float64)
    n = frames[0].size
    mean_map = frames.mean(axis=0)
    expected_mean = t.expected_mean(photons)
    expected_var = t.expected_variance(photons).mean()
    assert expected_var == pytest.approx(t.a * signal_dn + t.b, rel=1e-6)
    # Écart moyen nul à 5 σ près, σ = √(σ²_temp / (K·N)).
    sigma = np.sqrt(expected_var / (k * n))
    assert abs((mean_map - expected_mean).mean()) < 5 * sigma
    var = frames.var(axis=0, ddof=1).mean()
    assert var == pytest.approx(expected_var, rel=5 * np.sqrt(2 / ((k - 1) * n)) + 0.005)


def test_photon_transfer_slope_is_one_over_gain(sensor):
    """Pente de σ²(μ) par paires de différences = 1/K (la PTC « vraie »)."""
    t = sensor.truth
    mus, variances = [], []
    for i, s in enumerate([100.0, 400.0, 1000.0, 2000.0, 3000.0]):
        x = sensor.expose(t.photons_for_signal(s), n_frames=2, seed=100 + i).frames.astype(np.float64)
        mus.append(x.mean() - t.params.offset)
        variances.append(np.var(x[0] - x[1]) / 2)
    slope, _ = np.polyfit(mus, variances, 1)
    # L'ordonnée à l'origine (b) est trop mal contrainte par ces niveaux ;
    # elle est vérifiée sur l'obscurité (test_dark_stack_mean_and_variance).
    assert slope == pytest.approx(t.a, rel=0.02)


# --- Écrêtages et format ---------------------------------------------------

def test_output_format(sensor):
    frames = sensor.expose(1000.0, n_frames=3, seed=0).frames
    assert frames.dtype == np.uint16
    assert frames.shape == (3, *sensor.params.shape)


def test_adc_saturation(sensor):
    t = sensor.truth
    frames = sensor.expose(1e7, n_frames=2, seed=0).frames
    assert np.all(frames == t.saturation_dn)


def test_full_well_clipping():
    """Puits plein avant le convertisseur : plateau à offset + FW/K, sans bruit de photons."""
    p = SensorParams(seed=1 + SEED_OFFSET, full_well=6000.0, prnu_pixel=0.0, prnu_col=0.0)
    s = MatrixSensor(p)
    t = s.truth
    frames = s.expose(1e6, n_frames=32, seed=3).frames.astype(np.float64)
    plateau = t.offset_map + p.full_well / p.conversion_gain
    assert (frames.mean(axis=0) - plateau).mean() == pytest.approx(0.0, abs=0.01)
    assert frames.var(axis=0, ddof=1).mean() == pytest.approx(t.b, rel=0.03)


def test_scene_map_input(sensor):
    h, w = sensor.params.shape
    scene = np.linspace(0, 2000, w)[None, :].repeat(h, axis=0)
    acq = sensor.expose(scene, n_frames=1, seed=0)
    assert acq.frames.shape == (1, h, w)
    with pytest.raises(ValueError):
        sensor.expose(np.ones((3, 3)), seed=0)
    with pytest.raises(ValueError):
        sensor.expose(-1.0, seed=0)
