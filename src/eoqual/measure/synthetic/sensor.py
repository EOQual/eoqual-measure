"""
Simulateurs de capteurs à vérité connue : matrice (S0), barrette push
broom (S2), effets temporels structurés (S3).

Chaîne d'un pixel (modèle linéaire EMVA 1288, unités entre crochets) :

    photons [ph] ──× QE × PRNU──▶ signal moyen μ_e [e⁻]
      → Poisson                         (bruit de photons, en électrons)
      → écrêtage à la capacité du puits [e⁻]
      → (+ rayons cosmiques [e⁻], S3)
      → + bruit de lecture gaussien σ_lect [e⁻]
      → ÷ K (e⁻/DN) + offset + DSNU [DN]
      → (+ bruit de ligne, motifs périodiques [DN], S3)
      → quantification (arrondi) et écrêtage sur [0, 2ⁿ − 1] [DN]

Deux graines distinctes :

- `SensorParams.seed` fixe l'**identité du capteur** : cartes DSNU et PRNU
  (et, aux étapes suivantes, pixels chauds, colonnes défectueuses…),
  identiques d'une acquisition à l'autre ;
- la graine passée à `MatrixSensor.expose` fixe le **bruit temporel** de
  la pile acquise.

Notation du bruit (CATALOGUE.md, §0) : σ²(μ) = a·μ + b en DN, avec
a = 1/K et b = (σ_lect/K)² + 1/12 (quantification).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eoqual.measure.synthetic.dark import DarkCurrentMap, dark_current_map, temperature_factor
from eoqual.measure.synthetic.patterns import StructuredMap, pattern_rng, structured_map
from eoqual.measure.synthetic.response import (
    ADCTransfer,
    Defects,
    Transmission,
    adc_transfer,
    defect_maps,
    transmission_map,
)
from eoqual.measure.synthetic.temporal_effects import (
    COSMIC_TAG,
    PERIODIC_TAG,
    ROW_TAG,
    RTS_STATE_TAG,
    PeriodicPattern,
    RTSMap,
    cosmic_rays,
    periodic_field,
    row_offsets,
    rts_map,
    rts_states,
)

# Étiquette du flux de bruit temporel, distincte de celles des motifs fixes.
_TEMPORAL_TAG = zlib.crc32(b"temporal")
_BIAS_TAG = zlib.crc32(b"bias")
_OVERSCAN_TAG = zlib.crc32(b"overscan")


@dataclass(slots=True, frozen=True)
class SensorParams:
    """
    Paramètres d'un capteur matriciel (S0).

    Attributes
    ----------
    shape :
        Taille de la matrice (lignes, colonnes).
    qe :
        Rendement quantique moyen (électrons par photon).
    conversion_gain :
        Gain de conversion K (e⁻/DN).
    read_noise :
        Bruit de lecture σ_lect (e⁻ RMS).
    full_well :
        Capacité du puits (e⁻) ; le signal en électrons y est écrêté.
    bit_depth :
        Nombre de bits du convertisseur ; sortie écrêtée sur [0, 2ⁿ − 1].
    offset :
        Offset moyen (DN).
    dsnu_pixel, dsnu_row, dsnu_col :
        Écarts-types des composantes pixel, ligne et colonne de la DSNU (DN).
    prnu_pixel, prnu_row, prnu_col :
        Écarts-types relatifs des composantes de la PRNU (sans unité,
        0.01 = 1 %).
    seed :
        Graine des motifs fixes (identité du capteur).
    row_noise, row_noise_beta :
        Bruit de ligne temporel (S3) : offset commun aux pixels d'une ligne,
        écart-type en DN, spectre 1/f^β le long de la séquence de lecture
        (0 : blanc, 1 : rose).
    periodic :
        Motifs périodiques additifs (S3), `PeriodicPattern`.
    cosmic_rate, cosmic_charge, cosmic_max_length :
        Rayons cosmiques (S3) : nombre moyen de traces par pose (par image
        en push broom), charge déposée par pixel (e⁻), longueur maximale
        (pixels).
    dark_current, dark_nonuniformity :
        Courant d'obscurité moyen des pixels ordinaires (e⁻/s) à
        `temperature_ref`, et écart-type de son logarithme (DSNU du
        courant, loi log-normale) (S1).
    hot_pixel_fraction, hot_pixel_range :
        Fraction de pixels chauds et plage (min, max) de leur facteur sur
        le courant, tiré log-uniformément (S1).
    temperature_ref, bandgap :
        Température de référence (°C) et énergie de bande interdite (eV)
        de la loi I_d ∝ T³·exp(−E_g/2kT) (S1).
    glow_rate, glow_scale :
        Lueur d'amplificateur au coin (0, 0) (e⁻/s) et longueur de
        décroissance (pixels) (S1).
    bias_jitter, bias_drift :
        Variation de l'offset commune à toute une pose : écart-type d'un
        tirage par pose (DN) et dérive linéaire (DN par pose ; par ligne en
        push broom) (S1).
    overscan_columns :
        Nombre de colonnes d'overscan (matrice seulement) : non éclairées,
        elles portent offset, bruit de lecture, courant d'obscurité moyen,
        bruit de ligne et variations d'offset (S1).
    vignetting, n_dust, dust_depth, dust_radius :
        Transmission optique (S4) : baisse relative aux coins, nombre de
        poussières, profondeur de leur anneau d'atténuation, plage de rayons
        (pixels).
    nonlinearity, nonlinearity_dispersion :
        Compression e' = e·(1 − β·e/FW) : β commun et écart-type de sa
        dispersion par pixel (S4).
    dead_pixel_fraction, stuck_pixel_fraction :
        Fractions de pixels morts (aucune réponse à la lumière) et bloqués
        (code constant) (S4).
    rts_fraction, rts_amplitude, rts_rate :
        Pixels télégraphiques (S4) : fraction, plage de l'amplitude du saut
        (DN), plage des probabilités de bascule par pose (montée et
        descente tirées séparément).
    adc_dnl, adc_missing_codes :
        Convertisseur à pas inégaux (S4) : écart-type des seuils autour de
        leur position idéale (LSB) et nombre de codes manquants.
    """

    shape: tuple[int, int] = (256, 256)
    qe: float = 0.7
    conversion_gain: float = 3.0
    read_noise: float = 5.0
    full_well: float = 15000.0
    bit_depth: int = 12
    offset: float = 100.0
    dsnu_pixel: float = 0.5
    dsnu_row: float = 0.0
    dsnu_col: float = 0.3
    prnu_pixel: float = 0.005
    prnu_row: float = 0.0
    prnu_col: float = 0.002
    seed: int = 0
    row_noise: float = 0.0
    row_noise_beta: float = 0.0
    periodic: tuple[PeriodicPattern, ...] = ()
    cosmic_rate: float = 0.0
    cosmic_charge: float = 2000.0
    cosmic_max_length: float = 8.0
    dark_current: float = 0.0
    dark_nonuniformity: float = 0.0
    hot_pixel_fraction: float = 0.0
    hot_pixel_range: tuple[float, float] = (10.0, 1000.0)
    temperature_ref: float = 20.0
    bandgap: float = 1.12
    glow_rate: float = 0.0
    glow_scale: float = 20.0
    bias_jitter: float = 0.0
    bias_drift: float = 0.0
    overscan_columns: int = 0
    vignetting: float = 0.0
    n_dust: int = 0
    dust_depth: float = 0.05
    dust_radius: tuple[float, float] = (4.0, 12.0)
    nonlinearity: float = 0.0
    nonlinearity_dispersion: float = 0.0
    dead_pixel_fraction: float = 0.0
    stuck_pixel_fraction: float = 0.0
    rts_fraction: float = 0.0
    rts_amplitude: tuple[float, float] = (3.0, 10.0)
    rts_rate: tuple[float, float] = (0.02, 0.2)
    adc_dnl: float = 0.0
    adc_missing_codes: int = 0

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or min(self.shape) < 1:
            raise ValueError(f"shape doit être (lignes, colonnes) positifs, reçu {self.shape}")
        for name in ("qe", "conversion_gain", "full_well"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} doit être strictement positif")
        for name in ("read_noise", "dsnu_pixel", "dsnu_row", "dsnu_col",
                     "prnu_pixel", "prnu_row", "prnu_col", "row_noise", "cosmic_rate",
                     "cosmic_charge", "dark_current", "dark_nonuniformity", "glow_rate",
                     "bias_jitter", "overscan_columns"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} doit être positif ou nul")
        if not 1 <= self.bit_depth <= 16:
            raise ValueError("bit_depth doit être entre 1 et 16")
        if self.cosmic_max_length < 1:
            raise ValueError("cosmic_max_length doit être >= 1")
        if not all(isinstance(x, PeriodicPattern) for x in self.periodic):
            raise ValueError("periodic doit être un tuple de PeriodicPattern")
        if not 0 <= self.hot_pixel_fraction < 1:
            raise ValueError("hot_pixel_fraction doit être dans [0, 1[")
        lo, hi = self.hot_pixel_range
        if not 1 <= lo <= hi:
            raise ValueError("hot_pixel_range doit vérifier 1 <= min <= max")
        if self.glow_scale <= 0 or self.bandgap <= 0:
            raise ValueError("glow_scale et bandgap doivent être strictement positifs")
        if not 0 <= self.vignetting < 1 or not 0 <= self.dust_depth < 1 or self.n_dust < 0:
            raise ValueError("vignetting et dust_depth dans [0, 1[, n_dust >= 0")
        if self.nonlinearity_dispersion < 0 or not -0.5 < self.nonlinearity < 0.5:
            raise ValueError("nonlinearity dans ]−0,5 ; 0,5[, dispersion >= 0")
        if not 0 <= self.dead_pixel_fraction + self.stuck_pixel_fraction < 1:
            raise ValueError("fractions de pixels morts et bloqués dans [0, 1[")
        if not 0 <= self.rts_fraction < 1 or not 0 < self.rts_rate[0] <= self.rts_rate[1] <= 1:
            raise ValueError("rts_fraction dans [0, 1[, rts_rate dans ]0, 1]")
        if self.adc_dnl < 0 or self.adc_missing_codes < 0:
            raise ValueError("adc_dnl et adc_missing_codes doivent être positifs ou nuls")


@dataclass(slots=True)
class SensorTruth:
    """
    Vérité d'un capteur simulé : ce que les mesures doivent retrouver.

    Attributes
    ----------
    params :
        Paramètres du capteur.
    offset_map :
        Offset par pixel, offset + DSNU (DN), forme (H, W).
    dsnu :
        Décomposition pixel / ligne / colonne de la DSNU (DN).
    gain_map :
        Gain relatif par pixel, 1 + PRNU (moyenne exactement 1).
    prnu :
        Décomposition pixel / ligne / colonne de la PRNU (relative).
    a :
        Pente du modèle de bruit temporel, 1/K (DN par DN).
    b :
        Variance temporelle à signal nul, (σ_lect/K)² + 1/12 (DN²).
    saturation_dn :
        Code maximal du convertisseur, 2ⁿ − 1 (DN).
    tdi_stages :
        Nombre d'étages TDI (push broom) ; 1 pour un capteur matriciel.
    dark_current :
        Carte de courant d'obscurité (`DarkCurrentMap`, e⁻/s à
        `temperature_ref`), ou `None` sans courant d'obscurité (S1).
    hot_pixels :
        Masque des pixels chauds, ou `None` (S1).
    overscan_offset :
        Offset des colonnes d'overscan (H, n) en DN, ou `None`.
    transmission :
        Transmission optique (`Transmission`), ou `None` (S4).
    nonlinearity_map :
        Coefficient β de compression par pixel, ou `None` (S4).
    defects :
        Pixels morts et bloqués (`Defects`), ou `None` (S4).
    rts :
        Pixels télégraphiques (`RTSMap`), ou `None` (S4).
    adc :
        Convertisseur à pas inégaux (`ADCTransfer`), ou `None` : arrondi
        idéal (S4).
    """

    params: SensorParams
    offset_map: NDArray[np.float64]
    dsnu: StructuredMap
    gain_map: NDArray[np.float64]
    prnu: StructuredMap
    a: float
    b: float
    saturation_dn: int
    tdi_stages: int = 1
    dark_current: DarkCurrentMap | None = None
    hot_pixels: NDArray[np.bool_] | None = None
    overscan_offset: NDArray[np.float64] | None = None
    transmission: Transmission | None = None
    nonlinearity_map: NDArray[np.float64] | None = None
    defects: Defects | None = None
    rts: RTSMap | None = None
    adc: ADCTransfer | None = None

    @property
    def gain(self) -> float:
        """Gain de conversion K (e⁻/DN)."""
        return self.params.conversion_gain

    @property
    def read_noise_dn(self) -> float:
        """Bruit de lecture en DN, σ_lect / K."""
        return self.params.read_noise / self.params.conversion_gain

    @property
    def dsnu_rms(self) -> float:
        """DSNU totale : écart-type spatial de l'offset (DN)."""
        return float(self.dsnu.total.std())

    @property
    def prnu_rms(self) -> float:
        """PRNU totale : écart-type spatial du gain relatif."""
        return float(self.prnu.total.std())

    def signal_electrons(self, photons: ArrayLike) -> NDArray[np.float64]:
        """
        Signal moyen μ_e (e⁻) par pixel pour un nombre moyen de photons, avant
        non-linéarité.

        μ_e = N·QE·Φ·T·g (N étages TDI, T transmission optique, g gain
        relatif) ; nul pour les pixels morts.
        """
        e = self.tdi_stages * self.params.qe * np.asarray(photons, dtype=np.float64) * self.gain_map
        if self.transmission is not None:
            e = e * self.transmission.total
        if self.defects is not None:
            e = np.where(self.defects.dead, 0.0, e)
        return e

    def dark_electrons(self, exposure_time: float = 1.0, temperature: float | None = None) -> NDArray[np.float64]:
        """
        Charge d'obscurité moyenne par pixel (e⁻) : (I_d(T)·g + lueur)·t.

        `temperature` en °C (défaut : `temperature_ref`). La lueur
        d'amplificateur ne suit pas la loi en température.
        """
        p = self.params
        if self.dark_current is None:
            return np.zeros(p.shape)
        temp = p.temperature_ref if temperature is None else temperature
        f = temperature_factor(temp, p.temperature_ref, p.bandgap)
        return (self.dark_current.current * f + self.dark_current.glow) * exposure_time

    def expected_mean(self, photons: ArrayLike, exposure_time: float = 1.0,
                      temperature: float | None = None) -> NDArray[np.float64]:
        """
        Moyenne temporelle attendue par pixel (DN), hors écrêtages et hors
        variations d'offset par pose.

        offset_map + (μ_e + charge d'obscurité) / K.
        """
        e = self.signal_electrons(photons) + self.dark_electrons(exposure_time, temperature)
        if self.nonlinearity_map is not None:
            # E[e(1 − βe/FW)] = μ − β(μ² + μ)/FW pour e poissonnien.
            e = e - self.nonlinearity_map * (e**2 + e) / self.params.full_well
        mean = self.offset_map + e / self.params.conversion_gain
        if self.defects is not None:
            mean = np.where(self.defects.stuck, self.defects.stuck_value, mean)
        return mean

    def expected_variance(self, photons: ArrayLike, exposure_time: float = 1.0,
                          temperature: float | None = None) -> NDArray[np.float64]:
        """
        Variance temporelle attendue par pixel (DN²), hors écrêtages, part
        **non corrélée** entre pixels.

        a·μ + b, avec μ = (μ_e + charge d'obscurité)/K le signal corrigé de
        l'offset (DN) ; avec non-linéarité, la part poissonnienne est
        comprimée par (1 − 2βμ_e/FW)² (approximation au premier ordre).
        Nulle pour un pixel bloqué. Le bruit de ligne et les motifs
        temporels (S3) s'y ajoutent : voir `correlated_variance`.
        """
        e = self.signal_electrons(photons) + self.dark_electrons(exposure_time, temperature)
        if self.nonlinearity_map is not None:
            e = e * (1.0 - 2.0 * self.nonlinearity_map * e / self.params.full_well) ** 2
        var = self.a * e / self.params.conversion_gain + self.b
        if self.defects is not None:
            var = np.where(self.defects.stuck, 0.0, var)
        return var

    @property
    def correlated_variance(self) -> float:
        """
        Variance temporelle par pixel due aux effets corrélés (S3, DN²) :
        bruit de ligne σ_ligne² et motifs temporels Σ A²/2.
        """
        p = self.params
        return p.row_noise**2 + sum(x.rms**2 for x in p.periodic if x.temporal)

    def photons_for_signal(self, signal_dn: float) -> float:
        """Nombre moyen de photons (par étage TDI) donnant un signal moyen de `signal_dn` DN au-dessus de l'offset."""
        return signal_dn * self.params.conversion_gain / (self.params.qe * self.tdi_stages)

    @property
    def detector_offset(self) -> NDArray[np.float64]:
        """Offset par colonne (détecteur push broom), moyenne de `offset_map` sur les lignes (W,)."""
        return self.offset_map.mean(axis=0)

    @property
    def detector_gain(self) -> NDArray[np.float64]:
        """Gain relatif par colonne (détecteur push broom), (W,)."""
        return self.gain_map.mean(axis=0)


@dataclass(slots=True)
class Acquisition:
    """
    Pile d'images simulées et ce qui les a produites.

    Attributes
    ----------
    frames :
        Images en DN, entiers non signés, forme (K, H, W).
    photons :
        Nombre moyen de photons par pixel (scalaire ou carte).
    seed :
        Graine du bruit temporel.
    truth :
        Vérité du capteur.
    row_offsets, periodic, cosmic_mask :
        Réalisations des effets S3 (`None` s'ils sont absents) : offsets de
        ligne (K, H) en DN, champ périodique (K, H, W) en DN, pixels touchés
        par un rayon cosmique (K, H, W).
    exposure_time, temperature :
        Temps de pose (s) et température (°C, `None` : référence) (S1).
    bias_offsets :
        Variation d'offset commune à chaque pose (K,) en DN, ou `None` (S1).
    overscan :
        Colonnes d'overscan (K, H, n) en DN, ou `None` (S1).
    rts_states :
        États (K, n_rts) des pixels télégraphiques (vrai : niveau haut), ou
        `None` (S4).
    """

    frames: NDArray[np.uint16]
    photons: float | NDArray[np.float64]
    seed: int
    truth: SensorTruth = field(repr=False)
    row_offsets: NDArray[np.float64] | None = field(default=None, repr=False)
    periodic: NDArray[np.float64] | None = field(default=None, repr=False)
    cosmic_mask: NDArray[np.bool_] | None = field(default=None, repr=False)
    exposure_time: float = 1.0
    temperature: float | None = None
    bias_offsets: NDArray[np.float64] | None = field(default=None, repr=False)
    overscan: NDArray[np.uint16] | None = field(default=None, repr=False)
    rts_states: NDArray[np.bool_] | None = field(default=None, repr=False)


def _build_truth(p: SensorParams, tdi_stages: int = 1) -> SensorTruth:
    """Tire les motifs fixes d'un capteur et construit sa vérité."""
    dsnu = structured_map(p.shape, p.dsnu_pixel, p.dsnu_row, p.dsnu_col,
                          pattern_rng(p.seed, "dsnu"))
    prnu = structured_map(p.shape, p.prnu_pixel, p.prnu_row, p.prnu_col,
                          pattern_rng(p.seed, "prnu"))
    if np.any(prnu.total <= -1.0):
        raise ValueError("PRNU trop forte : gain relatif négatif ou nul sur certains pixels")
    k = p.conversion_gain
    dark = None
    if p.dark_current > 0 or p.glow_rate > 0:
        dark = dark_current_map(p.shape, p.dark_current, p.dark_nonuniformity, p.hot_pixel_fraction,
                                p.hot_pixel_range, p.glow_rate, p.glow_scale,
                                pattern_rng(p.seed, "dark_current"), pattern_rng(p.seed, "hot_pixels"))
    transmission = None
    if p.vignetting > 0 or p.n_dust > 0:
        transmission = transmission_map(p.shape, p.vignetting, p.n_dust, p.dust_depth, p.dust_radius,
                                        pattern_rng(p.seed, "transmission"))
    nl_map = None
    if p.nonlinearity != 0 or p.nonlinearity_dispersion > 0:
        rng_nl = pattern_rng(p.seed, "nonlinearity")
        nl_map = p.nonlinearity + p.nonlinearity_dispersion * rng_nl.standard_normal(p.shape)
    defects = None
    if p.dead_pixel_fraction > 0 or p.stuck_pixel_fraction > 0:
        defects = defect_maps(p.shape, p.dead_pixel_fraction, p.stuck_pixel_fraction,
                              2**p.bit_depth - 1, pattern_rng(p.seed, "defects"))
    rts = None
    if p.rts_fraction > 0:
        rts = rts_map(p.shape, p.rts_fraction, p.rts_amplitude, p.rts_rate, pattern_rng(p.seed, "rts"))
    adc = None
    if p.adc_dnl > 0 or p.adc_missing_codes > 0:
        adc = adc_transfer(p.bit_depth, p.adc_dnl, p.adc_missing_codes, pattern_rng(p.seed, "adc"))
    overscan_offset = None
    if p.overscan_columns > 0:
        # L'overscan partage l'électronique de ligne : offset + composante ligne de la DSNU.
        overscan_offset = np.repeat((p.offset + dsnu.row)[:, None], p.overscan_columns, axis=1)
    return SensorTruth(
        params=p,
        offset_map=p.offset + dsnu.total,
        dsnu=dsnu,
        gain_map=1.0 + prnu.total,
        prnu=prnu,
        a=1.0 / k,
        b=(p.read_noise / k) ** 2 + 1.0 / 12.0,
        saturation_dn=2**p.bit_depth - 1,
        tdi_stages=tdi_stages,
        dark_current=dark,
        hot_pixels=None if dark is None else dark.hot,
        overscan_offset=overscan_offset,
        transmission=transmission,
        nonlinearity_map=nl_map,
        defects=defects,
        rts=rts,
        adc=adc,
    )


def _digitize(
    mu_e: NDArray[np.float64],
    truth: SensorTruth,
    rng: np.random.Generator,
    *,
    extra_electrons: NDArray[np.float64] | None = None,
    extra_dn: NDArray[np.float64] | None = None,
) -> NDArray[np.uint16]:
    """
    Chaîne temporelle d'un tableau de signaux moyens μ_e (e⁻), de forme finale.

    Poisson (+ charges ajoutées) → puits → bruit de lecture → ÷K + offset
    (+ DN ajoutés) → convertisseur. `truth.offset_map` est diffusé sur les
    dernières dimensions.
    """
    p = truth.params
    electrons = rng.poisson(mu_e).astype(np.float64)
    if extra_electrons is not None:
        electrons += extra_electrons
    np.minimum(electrons, p.full_well, out=electrons)
    if truth.nonlinearity_map is not None:
        electrons *= 1.0 - truth.nonlinearity_map * electrons / p.full_well
    if p.read_noise > 0:
        electrons += rng.normal(0.0, p.read_noise, size=electrons.shape)
    analog = electrons / p.conversion_gain + truth.offset_map
    if extra_dn is not None:
        analog += extra_dn
    if truth.adc is not None:
        dn = truth.adc.quantize(analog)                     # seuils inégaux (S4)
    else:
        dn = np.floor(analog + 0.5)
    dn = np.clip(dn, 0, truth.saturation_dn)
    if truth.defects is not None and truth.defects.stuck.any():
        dn = np.where(truth.defects.stuck, truth.defects.stuck_value, dn)
    return dn.astype(np.uint16)


def _s3_effects(p: SensorParams, seed: int, frames: int, rows: int, width: int):
    """
    Effets S3 d'une acquisition de `frames` poses de `rows` lignes : offsets
    de ligne (frames, rows), champ périodique (frames, rows, width), charges
    et masque cosmiques (frames, rows, width). `None` pour un effet absent.
    """
    rows_dn = None
    if p.row_noise > 0:
        rng = np.random.default_rng([int(p.seed), ROW_TAG, int(seed)])
        rows_dn = row_offsets(frames * rows, p.row_noise, p.row_noise_beta, rng).reshape(frames, rows)
    field_dn = None
    if p.periodic:
        rng = np.random.default_rng([int(p.seed), PERIODIC_TAG, int(seed)])
        field_dn = periodic_field(p.periodic, (rows, width), frames, rng)
    cosmic_e, cosmic_mask = None, None
    if p.cosmic_rate > 0:
        rng = np.random.default_rng([int(p.seed), COSMIC_TAG, int(seed)])
        cosmic_e, cosmic_mask = cosmic_rays((frames, rows, width), p.cosmic_rate, p.cosmic_charge,
                                            p.cosmic_max_length, rng)
    return rows_dn, field_dn, cosmic_e, cosmic_mask


def _bias_offsets(p: SensorParams, seed: int, n: int) -> NDArray[np.float64] | None:
    """Variation d'offset commune par pose (ou par ligne en push broom) : gigue + dérive."""
    if p.bias_jitter == 0 and p.bias_drift == 0:
        return None
    rng = np.random.default_rng([int(p.seed), _BIAS_TAG, int(seed)])
    jitter = rng.normal(0.0, p.bias_jitter, n) if p.bias_jitter > 0 else np.zeros(n)
    return jitter + p.bias_drift * np.arange(n)


def _overscan(truth: SensorTruth, seed: int, n_frames: int, exposure_time: float,
              temperature: float | None, rows_dn, bias) -> NDArray[np.uint16]:
    """
    Colonnes d'overscan : courant d'obscurité moyen (sans dispersion ni
    pixels chauds), bruit de lecture, offset de ligne, bruit de ligne et
    variations d'offset de la pose.
    """
    p = truth.params
    h, n = p.shape[0], p.overscan_columns
    rng = np.random.default_rng([int(p.seed), _OVERSCAN_TAG, int(seed)])
    dark_e = 0.0
    if truth.dark_current is not None:
        f = temperature_factor(p.temperature_ref if temperature is None else temperature,
                               p.temperature_ref, p.bandgap)
        dark_e = p.dark_current * f * exposure_time
    mu = np.full((n_frames, h, n), dark_e)
    extra = None
    if rows_dn is not None:
        extra = np.broadcast_to(rows_dn[..., None], (n_frames, h, n)).copy()
    if bias is not None:
        extra = bias[:, None, None] if extra is None else extra + bias[:, None, None]
    # Pixels virtuels : ni non-linéarité ni défauts des pixels actifs.
    shim = replace(truth, offset_map=truth.overscan_offset, nonlinearity_map=None, defects=None)
    return _digitize(mu, shim, rng, extra_dn=extra)


def _extra_dn(rows_dn, field_dn):
    if rows_dn is None and field_dn is None:
        return None
    out = 0.0
    if rows_dn is not None:
        out = out + rows_dn[..., None]
    if field_dn is not None:
        out = out + field_dn
    return out


def _check_photons(photons: ArrayLike, shape: tuple[int, ...] | None) -> NDArray[np.float64]:
    arr = np.asarray(photons, dtype=np.float64)
    if arr.ndim != 0 and (shape is None or arr.shape != shape):
        raise ValueError(f"photons doit être un scalaire ou une carte {shape}")
    if np.any(arr < 0) or not np.all(np.isfinite(arr)):
        raise ValueError("photons doit être positif ou nul et fini")
    return arr


class MatrixSensor:
    """
    Capteur matriciel simulé (S0).

    Les motifs fixes sont tirés à la construction à partir de
    `params.seed` ; chaque appel à `expose` tire un nouveau bruit temporel
    à partir de sa propre graine.

    Examples
    --------
    >>> sensor = MatrixSensor(SensorParams(shape=(64, 64)))
    >>> dark = sensor.dark(n_frames=16, seed=1)
    >>> flat = sensor.expose(sensor.truth.photons_for_signal(1000), n_frames=2, seed=2)
    """

    def __init__(self, params: SensorParams | None = None) -> None:
        self.params = params if params is not None else SensorParams()
        self.truth = _build_truth(self.params)

    def expose(self, photons: ArrayLike, *, n_frames: int = 1, seed: int,
               exposure_time: float = 1.0, temperature: float | None = None) -> Acquisition:
        """
        Acquiert une pile de `n_frames` images sous un éclairement constant.

        Parameters
        ----------
        photons :
            Nombre moyen de photons par pixel pendant la pose : scalaire
            (champ uniforme) ou carte de forme (H, W) (scène). Le temps de
            pose ne le modifie pas : c'est la quantité intégrée.
        n_frames :
            Nombre d'images de la pile.
        seed :
            Graine du bruit temporel (obligatoire : simulations
            reproductibles).
        exposure_time :
            Temps de pose (s) : intègre le courant d'obscurité et la lueur.
        temperature :
            Température du plan focal (°C) ; défaut : `temperature_ref`.
        """
        if n_frames < 1:
            raise ValueError("n_frames doit être >= 1")
        if exposure_time < 0:
            raise ValueError("exposure_time doit être positif ou nul")
        p, t = self.params, self.truth
        photons_arr = _check_photons(photons, p.shape)
        rng = np.random.default_rng([int(p.seed), _TEMPORAL_TAG, int(seed)])
        mean_e = t.signal_electrons(photons_arr) + t.dark_electrons(exposure_time, temperature)
        mu_e = np.broadcast_to(mean_e, (n_frames, *p.shape))
        rows_dn, field_dn, cosmic_e, cosmic_mask = _s3_effects(p, seed, n_frames, *p.shape)
        bias = _bias_offsets(p, seed, n_frames)
        extra = _extra_dn(rows_dn, field_dn)
        if bias is not None:
            extra = bias[:, None, None] if extra is None else extra + bias[:, None, None]
        states = None
        if t.rts is not None and t.rts.index.size:
            states = rts_states(t.rts, n_frames, np.random.default_rng([int(p.seed), RTS_STATE_TAG, int(seed)]))
            jump = np.zeros((n_frames, p.shape[0] * p.shape[1]))
            jump[:, t.rts.index] = states * t.rts.amplitude
            jump = jump.reshape(n_frames, *p.shape)
            extra = jump if extra is None else extra + jump
        dn = _digitize(mu_e, t, rng, extra_electrons=cosmic_e, extra_dn=extra)
        overscan = None
        if p.overscan_columns > 0:
            overscan = _overscan(t, seed, n_frames, exposure_time, temperature, rows_dn, bias)
        stored = float(photons_arr) if photons_arr.ndim == 0 else photons_arr
        return Acquisition(frames=dn, photons=stored, seed=int(seed), truth=t,
                           row_offsets=rows_dn, periodic=field_dn, cosmic_mask=cosmic_mask,
                           exposure_time=float(exposure_time), temperature=temperature,
                           bias_offsets=bias, overscan=overscan, rts_states=states)

    def dark(self, *, n_frames: int = 1, seed: int, exposure_time: float = 1.0,
             temperature: float | None = None) -> Acquisition:
        """Pile d'images d'obscurité (aucun photon)."""
        return self.expose(0.0, n_frames=n_frames, seed=seed, exposure_time=exposure_time,
                           temperature=temperature)


@dataclass(slots=True)
class PushbroomAcquisition:
    """
    Image push broom simulée.

    Attributes
    ----------
    image :
        Image en DN, forme (M, W) : M lignes acquises successivement
        (le long de la trace), W détecteurs (en travers).
    photons :
        Photons moyens par pixel et par étage TDI (scalaire ou scène (M, W)).
    seed :
        Graine du bruit temporel.
    truth :
        Vérité du capteur ; `truth.detector_offset` et `truth.detector_gain`
        donnent l'offset et le gain de chaque détecteur.
    line_offsets, periodic, cosmic_mask :
        Réalisations des effets S3 : bruit de ligne (M,) en DN, champ
        périodique (M, W), pixels touchés par un rayon cosmique (M, W).
    exposure_time, temperature :
        Temps d'intégration d'une ligne (s) et température (°C) (S1).
    bias_offsets :
        Variation d'offset par ligne (M,) en DN : gigue tirée une fois par
        image, dérive par ligne (S1).
    rts_states :
        États (M, n_rts) des pixels télégraphiques, ou `None` (S4).
    """

    image: NDArray[np.uint16]
    photons: float | NDArray[np.float64]
    seed: int
    truth: SensorTruth = field(repr=False)
    line_offsets: NDArray[np.float64] | None = field(default=None, repr=False)
    periodic: NDArray[np.float64] | None = field(default=None, repr=False)
    cosmic_mask: NDArray[np.bool_] | None = field(default=None, repr=False)
    exposure_time: float = 1.0
    temperature: float | None = None
    bias_offsets: NDArray[np.float64] | None = field(default=None, repr=False)
    rts_states: NDArray[np.bool_] | None = field(default=None, repr=False)


class PushbroomSensor:
    """
    Barrette push broom simulée (S2) : W détecteurs, une ligne par instant.

    Chaque détecteur a son offset (DSNU) et son gain (PRNU) ; chaque ligne
    de l'image est une nouvelle acquisition (bruit temporel indépendant).
    Avec N étages TDI, la charge de N lignes de pixels est accumulée avant
    une lecture unique : signal ×N, bruit de lecture inchangé.

    Parameters
    ----------
    params :
        Paramètres de la barrette ; `shape` doit valoir (1, W). Utiliser
        les composantes `*_col` ou `*_pixel` pour la dispersion entre
        détecteurs (la composante ligne est nulle par construction).
    tdi_stages :
        Nombre d'étages TDI N (1 sans TDI).

    Notes
    -----
    En TDI, le gain d'un détecteur représente la moyenne des N étages de sa
    colonne ; la dispersion propre des étages (moyennée par le TDI) n'est
    pas simulée séparément.
    """

    def __init__(self, params: SensorParams | None = None, *, tdi_stages: int = 1) -> None:
        self.params = params if params is not None else SensorParams(shape=(1, 1024), dsnu_pixel=0.0, prnu_pixel=0.0)
        if self.params.shape[0] != 1:
            raise ValueError(f"une barrette push broom a shape=(1, W), reçu {self.params.shape}")
        if tdi_stages < 1:
            raise ValueError("tdi_stages doit être >= 1")
        if self.params.overscan_columns > 0:
            raise ValueError("overscan_columns n'est pas simulé pour une barrette push broom")
        self.tdi_stages = int(tdi_stages)
        self.truth = _build_truth(self.params, tdi_stages=self.tdi_stages)

    @property
    def n_detectors(self) -> int:
        """Nombre de détecteurs W."""
        return self.params.shape[1]

    def expose(self, photons: ArrayLike, *, n_lines: int | None = None, seed: int,
               exposure_time: float = 1.0, temperature: float | None = None) -> PushbroomAcquisition:
        """
        Acquiert une image de `n_lines` lignes.

        Parameters
        ----------
        photons :
            Photons moyens par pixel et par étage : scalaire (scène
            uniforme, `n_lines` obligatoire) ou scène (M, W).
        n_lines :
            Nombre de lignes M si `photons` est scalaire.
        seed :
            Graine du bruit temporel.
        exposure_time, temperature :
            Temps d'intégration d'une ligne (s) et température (°C) (S1).
        """
        w = self.n_detectors
        arr = np.asarray(photons, dtype=np.float64)
        if arr.ndim == 0:
            if n_lines is None or n_lines < 1:
                raise ValueError("n_lines (>= 1) est obligatoire pour une scène uniforme")
            arr = _check_photons(arr, None)
            shape = (int(n_lines), w)
        else:
            if arr.ndim != 2 or arr.shape[1] != w:
                raise ValueError(f"la scène doit être de forme (M, {w}), reçu {arr.shape}")
            if n_lines is not None and n_lines != arr.shape[0]:
                raise ValueError("n_lines incompatible avec la scène")
            arr = _check_photons(arr, arr.shape)
            shape = arr.shape
        p, t = self.params, self.truth
        rng = np.random.default_rng([int(p.seed), _TEMPORAL_TAG, int(seed)])
        mean_e = t.signal_electrons(arr) + t.dark_electrons(exposure_time, temperature)
        mu_e = np.broadcast_to(mean_e, shape)                    # gain (1, W) diffusé sur les lignes
        # Effets S3 : une seule « pose » de M lignes (le temps défile le long de la trace).
        rows_dn, field_dn, cosmic_e, cosmic_mask = _s3_effects(p, seed, 1, *shape)
        extra = None if (rows_dn is None and field_dn is None) else _extra_dn(rows_dn, field_dn)[0]
        bias = None
        if p.bias_jitter > 0 or p.bias_drift > 0:
            jitter = _bias_offsets(replace(p, bias_drift=0.0), seed, 1)
            bias = (0.0 if jitter is None else jitter[0]) + p.bias_drift * np.arange(shape[0])
            extra = bias[:, None] if extra is None else extra + bias[:, None]
        states = None
        if t.rts is not None and t.rts.index.size:
            # Push broom : la chaîne de Markov avance d'une ligne à l'autre.
            states = rts_states(t.rts, shape[0], np.random.default_rng([int(p.seed), RTS_STATE_TAG, int(seed)]))
            jump = np.zeros(shape)
            jump[:, t.rts.index] = states * t.rts.amplitude
            extra = jump if extra is None else extra + jump
        image = _digitize(mu_e, t, rng,
                          extra_electrons=None if cosmic_e is None else cosmic_e[0],
                          extra_dn=extra)
        stored = float(arr) if arr.ndim == 0 else arr
        return PushbroomAcquisition(image=image, photons=stored, seed=int(seed), truth=t,
                                    line_offsets=None if rows_dn is None else rows_dn[0],
                                    periodic=None if field_dn is None else field_dn[0],
                                    cosmic_mask=None if cosmic_mask is None else cosmic_mask[0],
                                    exposure_time=float(exposure_time), temperature=temperature,
                                    bias_offsets=bias, rts_states=states)

    def dark(self, *, n_lines: int, seed: int, exposure_time: float = 1.0,
             temperature: float | None = None) -> PushbroomAcquisition:
        """Bloc de lignes d'obscurité."""
        return self.expose(0.0, n_lines=n_lines, seed=seed, exposure_time=exposure_time,
                           temperature=temperature)


@dataclass(slots=True)
class WhiskbroomAcquisition:
    """
    Image whisk broom simulée.

    Attributes
    ----------
    image :
        Image en DN, forme (n_scans·k, W) : chaque balayage produit k lignes
        (une par détecteur), W pixels le long du balayage.
    detector :
        Indice du détecteur de chaque ligne (n_scans·k,).
    backward :
        Vrai pour les lignes d'un balayage retour (n_scans·k,).
    expected_profile :
        Niveau moyen attendu de chaque ligne (DN), pour la vérité.
    truth :
        Vérité de la barrette de k détecteurs.
    """

    image: NDArray[np.uint16]
    detector: NDArray[np.int64]
    backward: NDArray[np.bool_]
    expected_profile: NDArray[np.float64]
    truth: SensorTruth = field(repr=False)


class WhiskbroomSensor:
    """
    Imageur whisk broom simulé (S4) : k détecteurs balaient ensemble k lignes,
    en alternant balayages aller et retour.

    Chaque détecteur a son offset et son gain : l'image présente des bandes
    de période k lignes. Un balayage retour peut avoir un offset et un gain
    propres (`scan_offset`, `scan_gain`, hystérésis de l'électronique ou
    du mécanisme), d'où une structure de période 2k.

    Parameters
    ----------
    params :
        Paramètres de la barrette ; `shape` = (1, k). Dispersion entre
        détecteurs par `*_col` ou `*_pixel`.
    scan_offset :
        Offset ajouté aux balayages retour (DN).
    scan_gain :
        Écart relatif de gain des balayages retour.

    Notes
    -----
    Simplifications : pas d'effets S3 (bruit de ligne, motifs, cosmiques)
    ni de pixels télégraphiques ; l'ordre temporel des pixels d'un balayage
    retour (inversé) n'intervient pas, aucun effet simulé ne dépendant du
    temps à l'intérieur d'un balayage.
    """

    def __init__(self, params: SensorParams, *, scan_offset: float = 0.0, scan_gain: float = 0.0) -> None:
        if params.shape[0] != 1:
            raise ValueError(f"une barrette whisk broom a shape=(1, k), reçu {params.shape}")
        if params.overscan_columns > 0 or params.rts_fraction > 0:
            raise ValueError("overscan et pixels télégraphiques ne sont pas simulés en whisk broom")
        self.params = params
        self.scan_offset = float(scan_offset)
        self.scan_gain = float(scan_gain)
        self.truth = _build_truth(params)

    @property
    def n_detectors(self) -> int:
        """Nombre de détecteurs k."""
        return self.params.shape[1]

    def expose(self, photons: ArrayLike, *, n_scans: int | None = None, width: int | None = None,
               seed: int, exposure_time: float = 1.0) -> WhiskbroomAcquisition:
        """
        Acquiert une image de `n_scans` balayages.

        Parameters
        ----------
        photons :
            Photons moyens par pixel : scalaire (scène uniforme, `n_scans` et
            `width` obligatoires) ou scène (n_scans·k, W).
        seed :
            Graine du bruit temporel.
        exposure_time :
            Temps d'intégration d'un pixel (s).
        """
        k = self.n_detectors
        arr = np.asarray(photons, dtype=np.float64)
        if arr.ndim == 0:
            if not n_scans or not width:
                raise ValueError("n_scans et width sont obligatoires pour une scène uniforme")
            shape = (n_scans * k, width)
        else:
            if arr.ndim != 2 or arr.shape[0] % k:
                raise ValueError(f"la scène doit être de forme (n_scans·{k}, W)")
            shape = arr.shape
        arr = _check_photons(arr, arr.shape if arr.ndim else None)
        p, t = self.params, self.truth
        rows = shape[0]
        det = np.arange(rows) % k
        back = (np.arange(rows) // k) % 2 == 1
        g = t.gain_map[0, det] * (1.0 + self.scan_gain * back)                   # (rows,)
        dark = t.dark_electrons(exposure_time)[0, det]
        mean_e = p.qe * np.broadcast_to(arr, shape) * g[:, None] + dark[:, None]
        offsets = t.offset_map[0, det] + self.scan_offset * back
        nl = None if t.nonlinearity_map is None else t.nonlinearity_map[0, det][:, None]
        shim = replace(t, offset_map=offsets[:, None], nonlinearity_map=nl, defects=None, rts=None)
        rng = np.random.default_rng([int(p.seed), _TEMPORAL_TAG, int(seed)])
        image = _digitize(mean_e, shim, rng)
        expected = offsets + (mean_e.mean(axis=1)) / p.conversion_gain
        return WhiskbroomAcquisition(image=image, detector=det.astype(np.int64), backward=back,
                                     expected_profile=expected, truth=t)
