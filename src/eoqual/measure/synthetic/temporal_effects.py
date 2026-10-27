"""
Effets temporels structurés (S3) : bruit de ligne, motifs périodiques,
rayons cosmiques.

Ils s'ajoutent à la chaîne S0 sans la modifier : chacun tire dans son
propre flux aléatoire, et tous sont nuls par défaut. Voir
REFERENCE_TECHNIQUE.md, partie V, §2.5.

Conventions de coordonnées : i est l'indice de ligne (matriciel : ligne
de la matrice, lue après la précédente ; push broom : ligne acquise, donc
le temps), j l'indice de colonne. Les fréquences sont en cycles/pixel.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from eoqual.measure.synthetic.patterns import power_law_noise

ROW_TAG = zlib.crc32(b"row_noise")
PERIODIC_TAG = zlib.crc32(b"periodic")
COSMIC_TAG = zlib.crc32(b"cosmic")


@dataclass(slots=True, frozen=True)
class PeriodicPattern:
    """
    Motif sinusoïdal additif A·cos(2π(f_x·j + f_y·i) + φ) (DN).

    Attributes
    ----------
    amplitude :
        Amplitude crête A (DN) ; valeur efficace A/√2.
    fx, fy :
        Fréquences en travers des colonnes (j) et le long des lignes (i),
        en cycles/pixel, dans ]−0,5, 0,5].
    temporal :
        Vrai : phase φ tirée à chaque pose (interférence non synchronisée,
        bruit temporel). Faux : phase fixe `phase` (couplage synchronisé,
        bruit spatial fixe).
    phase :
        Phase fixe (radians) si `temporal` est faux.
    """

    amplitude: float
    fx: float
    fy: float
    temporal: bool = True
    phase: float = 0.0

    def __post_init__(self) -> None:
        if self.amplitude < 0:
            raise ValueError("amplitude doit être positive ou nulle")
        for f in (self.fx, self.fy):
            if not -0.5 < f <= 0.5:
                raise ValueError("fx et fy doivent être dans ]−0,5 ; 0,5] cycle/pixel")

    @property
    def rms(self) -> float:
        """Valeur efficace A/√2 (DN)."""
        return self.amplitude / np.sqrt(2.0)


def row_offsets(n_rows: int, sigma: float, beta: float, rng: np.random.Generator) -> NDArray[np.float64]:
    """
    Offsets de ligne (DN) le long de la séquence de lecture, de spectre 1/f^β.

    La séquence est continue d'une pose à l'autre (bruit en 1/f d'une
    électronique) : pour une pile (K, H), on tire K·H valeurs d'un seul
    tenant. Écart-type réalisé exactement égal à `sigma`.
    """
    if sigma == 0:
        return np.zeros(n_rows)
    return sigma * power_law_noise(n_rows, beta, rng)


def periodic_field(
    patterns: tuple[PeriodicPattern, ...],
    shape: tuple[int, int],
    n_frames: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Somme des motifs, forme (K, H, W) ; phases temporelles tirées par pose."""
    h, w = shape
    i = np.arange(h)[:, None]
    j = np.arange(w)[None, :]
    out = np.zeros((n_frames, h, w))
    for pat in patterns:
        arg = 2 * np.pi * (pat.fx * j + pat.fy * i)
        phases = rng.uniform(0, 2 * np.pi, n_frames) if pat.temporal else np.full(n_frames, pat.phase)
        out += pat.amplitude * np.cos(arg[None] + phases[:, None, None])
    return out


def cosmic_rays(
    shape: tuple[int, ...],
    rate: float,
    charge: float,
    max_length: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """
    Traces de rayons cosmiques : charge (e⁻) déposée et masque, forme `shape`.

    `shape` vaut (K, H, W) (une trace par pose) ou (M, W) (push broom, une
    seule « pose »). Nombre de traces par pose : Poisson(rate). Chaque trace
    est un segment de longueur uniforme dans [1, max_length] pixels,
    d'orientation uniforme, déposant `charge` e⁻ par pixel traversé.
    """
    frames = shape if len(shape) == 3 else (1, *shape)
    k, h, w = frames
    dep = np.zeros(frames)
    for f in range(k):
        for _ in range(rng.poisson(rate)):
            y0, x0 = rng.uniform(0, h), rng.uniform(0, w)
            length = rng.uniform(1.0, max_length)
            theta = rng.uniform(0, np.pi)
            n = max(int(np.ceil(length)) * 2, 2)
            t = np.linspace(0, length, n)
            ys = np.clip((y0 + t * np.sin(theta)).astype(int), 0, h - 1)
            xs = np.clip((x0 + t * np.cos(theta)).astype(int), 0, w - 1)
            dep[f, ys, xs] = charge
    dep = dep.reshape(shape)
    return dep, dep > 0


RTS_STATE_TAG = zlib.crc32(b"rts_state")


@dataclass(slots=True)
class RTSMap:
    """
    Pixels télégraphiques (RTS, S4) : bruit à deux niveaux.

    Un piège de charge près du transistor de lecture capture et libère un
    électron au hasard : la sortie du pixel bascule entre deux niveaux
    séparés de `amplitude`. L'état suit une chaîne de Markov d'une pose à
    l'autre (d'une ligne à l'autre en push broom) : probabilité de passer
    de l'état bas à l'état haut `p_up`, du haut au bas `p_down` ; durées
    moyennes de séjour 1/p_up (bas) et 1/p_down (haut).

    Attributes
    ----------
    mask :
        Pixels télégraphiques, forme (H, W).
    index :
        Indices aplatis de ces pixels.
    amplitude, p_up, p_down :
        Amplitude du saut (DN) et probabilités de bascule par pose, par
        pixel télégraphique (même ordre que `index`).
    """

    mask: NDArray[np.bool_]
    index: NDArray[np.int64]
    amplitude: NDArray[np.float64]
    p_up: NDArray[np.float64]
    p_down: NDArray[np.float64]

    @property
    def high_fraction(self) -> NDArray[np.float64]:
        """Probabilité stationnaire de l'état haut, p_up / (p_up + p_down)."""
        return self.p_up / (self.p_up + self.p_down)


def rts_map(shape: tuple[int, int], fraction: float, amplitude: tuple[float, float],
            rate: tuple[float, float], rng: np.random.Generator) -> RTSMap:
    """Tire les pixels télégraphiques : amplitude et probabilités uniformes dans leurs plages."""
    n = shape[0] * shape[1]
    k = int(round(fraction * n))
    idx = np.sort(rng.choice(n, size=k, replace=False)) if k > 0 else np.zeros(0, dtype=np.int64)
    mask = np.zeros(shape, dtype=bool)
    mask.flat[idx] = True
    return RTSMap(mask=mask, index=idx.astype(np.int64),
                  amplitude=rng.uniform(*amplitude, size=k),
                  p_up=rng.uniform(*rate, size=k), p_down=rng.uniform(*rate, size=k))


def rts_states(rts: RTSMap, n_steps: int, rng: np.random.Generator) -> NDArray[np.bool_]:
    """
    États (n_steps, n_rts) des pixels télégraphiques : état initial tiré de la
    loi stationnaire, puis chaîne de Markov.
    """
    k = rts.index.size
    states = np.zeros((n_steps, k), dtype=bool)
    if k == 0 or n_steps == 0:
        return states
    s = rng.random(k) < rts.high_fraction
    for t in range(n_steps):
        states[t] = s
        u = rng.random(k)
        s = np.where(s, u >= rts.p_down, u < rts.p_up)
    return states
