"""
`temporal_noise` — bruit temporel d'un capteur matriciel.

Entrée : paire ou pile de champs uniformes de même niveau. Le bruit spatial
fixe (DSNU, PRNU) est éliminé par différence (paire) ou par variance
temporelle pixel à pixel (pile). Voir REFERENCE_TECHNIQUE.md, partie IV, chapitre 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eoqual.measure.radiometry._stack import as_stack
from eoqual.measure.radiometry.dark import DarkFrameResult


@dataclass(slots=True)
class TemporalNoiseResult:
    """
    Résultat de `measure_temporal_noise` (unités : DN).

    Attributes
    ----------
    method :
        ``"pair"`` (différence de deux poses) ou ``"stack"`` (variance
        temporelle par pixel).
    n_frames :
        Nombre de poses utilisées.
    mean :
        Niveau moyen brut (moyenne spatio-temporelle).
    dark_level :
        Niveau d'obscurité retiré (offset moyen), ou `None`.
    signal :
        Signal moyen corrigé de l'obscurité, mean − dark_level, ou `None`
        sans obscurité fournie.
    temporal_variance, u_temporal_variance :
        Variance temporelle σ²_temp et son incertitude-type (hypothèse
        gaussienne, pixels indépendants).
    temporal_noise :
        √σ²_temp.
    variance_map :
        Variance temporelle par pixel (1/(K − 1)) pour ``"stack"`` ;
        `None` pour ``"pair"``.
    """

    method: str
    n_frames: int
    mean: float
    dark_level: float | None
    signal: float | None
    temporal_variance: float
    u_temporal_variance: float
    temporal_noise: float
    variance_map: NDArray[np.float64] | None


def _dark_level(dark: float | ArrayLike | DarkFrameResult | None) -> float | None:
    if dark is None:
        return None
    if isinstance(dark, DarkFrameResult):
        return dark.mean_offset
    arr = np.asarray(dark, dtype=np.float64)
    if arr.ndim == 3:
        raise ValueError("dark : fournir un niveau, une carte d'offset ou un DarkFrameResult, pas une pile")
    return float(arr.mean())


def measure_temporal_noise(
    frames: ArrayLike,
    *,
    dark: float | ArrayLike | DarkFrameResult | None = None,
    method: Literal["auto", "pair", "stack"] = "auto",
) -> TemporalNoiseResult:
    """
    Mesure le bruit temporel sur une paire ou une pile de champs uniformes.

    Parameters
    ----------
    frames :
        Pile (K, H, W), K ≥ 2, de poses au même niveau d'éclairement.
    dark :
        Obscurité à retirer du niveau moyen : niveau scalaire, carte
        d'offset (H, W) ou résultat de `measure_dark_frame`. N'affecte pas
        la variance.
    method :
        ``"pair"`` : σ² = var(x₁ − x₂)/2 sur les deux premières poses ;
        ``"stack"`` : moyenne des variances temporelles par pixel ;
        ``"auto"`` : ``"pair"`` si K = 2, ``"stack"`` sinon.

    Notes
    -----
    La différence de paire retire sa propre moyenne : un écart de niveau
    global entre les deux poses (dérive d'offset, fluctuation de la source)
    ne biaise pas la variance. Une fluctuation de source *proportionnelle*
    au signal, en revanche, laisse un résidu de PRNU (voir la référence).
    """
    stack = as_stack(frames, min_frames=2)
    k = stack.shape[0]
    n = stack[0].size
    if method == "auto":
        method = "pair" if k == 2 else "stack"

    if method == "pair":
        diff = stack[0] - stack[1]
        var = float(diff.var(ddof=1) / 2.0)
        u_var = var * np.sqrt(2.0 / (n - 1))
        var_map = None
        used = 2
    elif method == "stack":
        if k < 3:
            raise ValueError("method='stack' demande au moins 3 poses (utiliser 'pair' pour 2)")
        var_map = stack.var(axis=0, ddof=1)
        var = float(var_map.mean())
        u_var = var * np.sqrt(2.0 / ((k - 1) * n))
        used = k
    else:
        raise ValueError(f"method inconnue : {method!r}")

    mean = float(stack[:used].mean())
    dark_level = _dark_level(dark)
    signal = None if dark_level is None else mean - dark_level
    return TemporalNoiseResult(
        method=method,
        n_frames=used,
        mean=mean,
        dark_level=dark_level,
        signal=signal,
        temporal_variance=var,
        u_temporal_variance=float(u_var),
        temporal_noise=float(np.sqrt(var)),
        variance_map=var_map,
    )
