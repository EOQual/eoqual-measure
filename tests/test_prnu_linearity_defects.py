"""Validation de `measure_prnu`, `measure_linearity` et `map_defective_pixels`."""

import numpy as np
import pytest
from conftest import SEED_OFFSET

from eoqual.measure.radiometry import (
    map_defective_pixels,
    measure_dark_current,
    measure_dark_frame,
    measure_linearity,
    measure_prnu,
)
from eoqual.measure.radiometry.dark import _u_sigma
from eoqual.measure.radiometry.nonuniformity import _legendre_surface
from eoqual.measure.synthetic import MatrixSensor, SensorParams


# --- PRNU -------------------------------------------------------------------

@pytest.fixture(scope="module")
def prnu_case():
    s = MatrixSensor(SensorParams(seed=110 + SEED_OFFSET, prnu_pixel=0.01, prnu_row=0.002,
                                  prnu_col=0.004, vignetting=0.1))
    t = s.truth
    dark = measure_dark_frame(s.dark(n_frames=32, seed=1).frames)
    flats = [s.expose(t.photons_for_signal(lvl), n_frames=16, seed=10 + i).frames
             for i, lvl in enumerate([500.0, 1500.0, 3000.0])]
    true_resp = t.gain_map * t.transmission.total
    return t, dark, flats, true_resp / true_resp.mean()


def test_prnu_total(prnu_case):
    t, dark, flats, truth = prnu_case
    r = measure_prnu(flats, dark)
    assert abs(r.prnu - truth.std()) < 5 * r.u_prnu
    assert r.prnu_emva == pytest.approx(truth.std(), rel=0.01)
    np.testing.assert_allclose(r.prnu_per_level, truth.std(), rtol=0.01)
    err = r.response_map - truth
    assert np.nanstd(err) == pytest.approx(np.sqrt(r.noise_variance), rel=0.05)


def test_prnu_low_and_high_frequency(prnu_case):
    t, dark, flats, truth = prnu_case
    r = measure_prnu(flats, dark, smooth_degree=4)
    surface = _legendre_surface(truth, np.ones_like(truth, bool), 4)
    assert r.low_frequency == pytest.approx(surface.std() / surface.mean(), rel=0.005)
    assert abs(r.high_frequency - (truth / surface).std()) < 5 * r.u_high_frequency
    # Décomposition de la PRNU haute fréquence : la surface de degré 4 absorbe
    # ≈ 5/256 de la variance des profils aléatoires de lignes et de colonnes.
    d = r.decomposition
    assert d.pixel == pytest.approx(t.prnu.pixel.std(), rel=0.01)
    assert d.col == pytest.approx(t.prnu.col.std(), rel=0.08)
    assert d.row == pytest.approx(t.prnu.row.std(), rel=0.15)


def test_prnu_exclusion_and_checks(prnu_case):
    t, dark, flats, _ = prnu_case
    mask = np.zeros(t.params.shape, bool)
    mask[:10] = True
    r = measure_prnu(flats[2], dark, exclude=mask)
    assert np.isnan(r.response_map[:10]).all() and r.prnu_emva is None
    with pytest.raises(ValueError):
        measure_prnu(flats[0], dark, exclude=np.zeros((3, 3), bool))


# --- Linéarité --------------------------------------------------------------

def _lin_sensor(beta, disp):
    s = MatrixSensor(SensorParams(seed=111 + SEED_OFFSET, nonlinearity=beta, nonlinearity_dispersion=disp,
                                  full_well=12000.0))
    t = s.truth
    dark = measure_dark_frame(s.dark(n_frames=16, seed=1).frames)
    expo = np.linspace(0.05, 1.1, 14)
    photons = expo * t.photons_for_signal(3800.0)
    stacks = [s.expose(p, n_frames=8, seed=10 + i).frames for i, p in enumerate(photons)]
    return s, t, measure_linearity(stacks, expo, dark, saturation_dn=t.saturation_dn)


@pytest.mark.parametrize("beta, disp", [(0.1, 0.02), (-0.05, 0.01)])
def test_linearity_curvature(beta, disp):
    s, t, r = _lin_sensor(beta, disp)
    kappa_true = -beta * t.gain / s.params.full_well
    assert abs(r.curvature - kappa_true) < 5 * r.u_curvature + 1e-3 * abs(kappa_true)
    disp_true = t.nonlinearity_map.std() * t.gain / s.params.full_well
    assert abs(r.curvature_dispersion - disp_true) < 5 * r.u_curvature_dispersion
    assert r.linearity_error > 1.0                       # non-linéarité de quelques %
    assert not r.used[-1]                                # dernière exposition saturée, exclue


def test_linear_sensor():
    _, _, r = _lin_sensor(0.0, 0.0)
    assert r.linearity_error < 0.05
    assert abs(r.curvature) < 5 * r.u_curvature
    assert r.curvature_dispersion < 5 * r.u_curvature_dispersion


def test_relative_equalization_misses_common_nonlinearity():
    """
    Non-linéarité commune e' = e(1 − βe/FW), c = βK/FW (DN⁻¹), x signal linéaire (DN) :

    - l'égalisation relative (gains calculés à un niveau, appliqués à un autre)
      n'en garde qu'un résidu du second ordre, ≈ PRNU·c·Δx, alors que l'erreur
      de linéarité absolue atteint plusieurs % ;
    - la PRNU apparente est comprimée d'un facteur (1 − 2cx)/(1 − cx).
    """
    beta, fw = 0.1, 12000.0
    s = MatrixSensor(SensorParams(seed=112 + SEED_OFFSET, nonlinearity=beta, full_well=fw))
    t = s.truth
    c = beta * t.gain / fw
    dark = measure_dark_frame(s.dark(n_frames=16, seed=1).frames)
    x_lo, x_hi = 500.0, 3500.0
    lo = measure_prnu(s.expose(t.photons_for_signal(x_lo), n_frames=16, seed=2).frames, dark)
    hi = measure_prnu(s.expose(t.photons_for_signal(x_hi), n_frames=16, seed=3).frames, dark)
    f = lambda x: (1 - 2 * c * x) / (1 - c * x)
    assert hi.prnu / lo.prnu == pytest.approx(f(x_hi) / f(x_lo), abs=5 * np.hypot(hi.u_prnu / hi.prnu, lo.u_prnu / lo.prnu))
    # Égalisation : réponse au fort niveau divisée par la réponse au faible niveau.
    residual = hi.response_map / lo.response_map
    v = hi.noise_variance + lo.noise_variance
    s2 = residual.var() - v
    expected = lo.prnu * c * (x_hi - x_lo)                             # second ordre
    assert np.sqrt(max(s2, 0.0)) < 2 * expected + 5 * _u_sigma(s2, v, residual.size, np.inf)
    assert expected < 1e-3 and c * x_hi > 0.08                          # résidu < 0,1 % pour 8,75 % de compression


def test_linearity_checks():
    s = MatrixSensor(SensorParams(shape=(32, 32), seed=1))
    d = s.dark(n_frames=4, seed=0).frames
    st = [s.expose(100.0 * i, n_frames=2, seed=i).frames for i in (1, 2)]
    with pytest.raises(ValueError):
        measure_linearity(st, [1.0, 2.0], d)


# --- Pixels défectueux --------------------------------------------------------

@pytest.fixture(scope="module")
def defect_case():
    s = MatrixSensor(SensorParams(seed=113 + SEED_OFFSET, dark_current=2.0, dark_nonuniformity=0.3,
                                  hot_pixel_fraction=1e-3, hot_pixel_range=(30.0, 1000.0),
                                  dead_pixel_fraction=1e-3, stuck_pixel_fraction=5e-4,
                                  vignetting=0.15, n_dust=4, prnu_pixel=0.01))
    t = s.truth
    dark = s.dark(n_frames=16, seed=1, exposure_time=30.0).frames.astype(float)
    flats = [s.expose(t.photons_for_signal(lvl), n_frames=16, seed=10 + i, exposure_time=30.0).frames.astype(float)
             for i, lvl in enumerate([1000.0, 3000.0])]
    dc = measure_dark_current([s.dark(n_frames=16, seed=20 + i, exposure_time=te).frames
                               for i, te in enumerate([1.0, 30.0, 100.0])], [1.0, 30.0, 100.0],
                              saturation_dn=t.saturation_dn)
    return s, t, dark, flats, dc


def test_defect_classes(defect_case):
    _, t, dark, flats, dc = defect_case
    # p_FA = 10⁻⁸ par pixel : ≈ 7·10⁻⁴ fausse alarme attendue par carte de 65 536 pixels
    # (au défaut 10⁻⁶, 5 fausses alarmes mesurées sur 40 cartes pour 2,6 nominales).
    m = map_defective_pixels(dark, flats, dark_current=dc, dark_exposure_time=30.0, false_alarm=1e-8)
    for name, truth in (("stuck", t.defects.stuck), ("dead", t.defects.dead), ("hot", t.dark_current.hot)):
        est = getattr(m, name)
        assert not np.any(est & ~truth), name                     # aucune fausse détection
        assert (est & truth).sum() >= 0.97 * truth.sum(), name
    assert m.noisy.sum() == 0 and m.response.sum() == 0
    assert set(m.counts()) == {"stuck", "dead", "hot", "noisy", "response"}
    assert np.array_equal(m.any, m.codes > 0)


def test_injected_noisy_and_response_outliers(defect_case):
    s, t, dark, flats, dc = defect_case
    rng = np.random.default_rng(5)
    clean = ~(t.defects.stuck | t.defects.dead | t.dark_current.hot)
    idx = rng.choice(np.flatnonzero(clean), size=40, replace=False)
    noisy_px, resp_px = idx[:20], idx[20:]
    d2 = dark.copy()
    d2.reshape(16, -1)[:, noisy_px] += rng.normal(0, 15.0, (16, 20))       # bruit télégraphique, ≫ σ
    f2 = [f.copy() for f in flats]
    for f in f2:
        sig = f.reshape(16, -1)[:, resp_px] - dark.reshape(16, -1)[:, resp_px].mean(axis=0)
        f.reshape(16, -1)[:, resp_px] -= 0.3 * sig                          # réponse −30 %
    m = map_defective_pixels(d2, f2, dark_current=dc, dark_exposure_time=30.0, false_alarm=1e-8)
    assert set(np.flatnonzero(m.noisy)) == set(noisy_px)
    assert set(np.flatnonzero(m.response)) == set(resp_px)


def test_defect_checks(defect_case):
    _, _, dark, flats, _ = defect_case
    with pytest.raises(ValueError):
        map_defective_pixels(dark[:2], flats)
    with pytest.raises(ValueError):
        map_defective_pixels(dark, [])
