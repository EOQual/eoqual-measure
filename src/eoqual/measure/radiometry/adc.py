"""
`adc_histogram` — codes manquants et non-linéarités différentielle et
intégrale du convertisseur (méthode de l'histogramme, IEEE Std 1241).

Entrée : images dont la tension d'entrée a une densité lisse à l'échelle
du pas de quantification : fond lisse ou rampe couvrant la plage de
codes, avec un bruit de plusieurs pas. Le nombre d'occurrences d'un code
est alors proportionnel à sa largeur : DNL_c = n_c / n̂_c − 1, où n̂_c est
la densité attendue, estimée en lissant l'histogramme. Voir
REFERENCE_TECHNIQUE.md, partie IV, chapitre 9.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d


@dataclass(slots=True)
class ADCHistogramResult:
    """
    Résultat de `adc_histogram`.

    Attributes
    ----------
    codes :
        Codes analysés (densité attendue ≥ `min_counts`), croissants.
    counts, expected :
        Occurrences observées et attendues (histogramme lissé) par code.
    dnl, u_dnl :
        Non-linéarité différentielle (LSB) et son incertitude (Poisson,
        1/√n̂).
    inl :
        Non-linéarité intégrale (LSB) : somme cumulée de la DNL sur les
        codes analysés, rapportée à la droite des extrémités et centrée.
        Exacte en mode ``"uniform"`` ; limitée à l'échelle du lissage en
        mode ``"local"``.
    missing :
        Codes manquants : aucune occurrence pour une densité attendue
        ≥ `min_counts` (probabilité d'un zéro fortuit e^{−n̂}).
    dnl_rms, dnl_max :
        Écart-type de la DNL corrigé du bruit d'estimation, et |DNL| maximal.
    """

    codes: NDArray[np.int64]
    counts: NDArray[np.float64]
    expected: NDArray[np.float64]
    dnl: NDArray[np.float64]
    u_dnl: NDArray[np.float64]
    inl: NDArray[np.float64]
    missing: NDArray[np.int64]
    dnl_rms: float
    dnl_max: float


def adc_histogram(
    images: ArrayLike,
    *,
    bit_depth: int | None = None,
    density: str = "local",
    smoothing: float = 16.0,
    min_counts: float = 50.0,
    plateau_tolerance: float = 0.1,
    exclusion: int = 1,
) -> ADCHistogramResult:
    """
    DNL, INL et codes manquants par la méthode de l'histogramme.

    Parameters
    ----------
    images :
        Images en codes entiers (toute forme).
    bit_depth :
        Nombre de bits (défaut : déduit du code maximal).
    density :
        Densité attendue de la tension d'entrée :

        - ``"uniform"`` : entrée uniformément répartie (rampe beaucoup plus
          large que le bruit) ; on analyse le plateau de l'histogramme, où
          la densité est constante, et l'INL est exacte ;
        - ``"local"`` : densité lisse quelconque (fond lisse bruité),
          estimée par régression locale quadratique de
          l'histogramme ; les codes où la densité est trop courbée pour
          cette régression (test de cohérence entre deux largeurs) sont
          écartés de la DNL, mais pas de la recherche des codes manquants.
          La DNL est juste,
          mais l'INL n'est connue qu'à l'échelle du lissage : ses variations
          plus lentes sont absorbées par l'estimation de la densité.
    smoothing :
        Écart-type (codes) des poids de la régression locale (mode
        ``"local"``).
    min_counts :
        Densité attendue minimale d'un code analysé.
    plateau_tolerance :
        Mode ``"uniform"`` : un code appartient au plateau si sa densité
        lissée est à moins de cette fraction de la médiane du plateau.
    exclusion :
        Mode ``"local"`` : demi-largeur (codes) de la fenêtre exclue autour
        du code dans l'estimation de sa densité.
    """
    x = np.asarray(images)
    if not np.issubdtype(x.dtype, np.integer):
        if not np.all(np.mod(x, 1) == 0):
            raise ValueError("images doit contenir des codes entiers")
        x = x.astype(np.int64)
    if x.min() < 0:
        raise ValueError("codes négatifs")
    n_codes = 2**bit_depth if bit_depth else int(x.max()) + 1
    hist = np.bincount(x.ravel(), minlength=n_codes).astype(np.float64)[:n_codes]
    local = gaussian_filter1d(hist, smoothing, mode="nearest")

    if density == "uniform":
        ref = float(np.percentile(local[1:-1], 90))              # insensible au pic d'écrêtage
        top = np.median(local[local >= 0.5 * ref])
        use = np.abs(local / top - 1) <= plateau_tolerance
        use[0] = use[-1] = False
        codes = np.flatnonzero(use)
        if codes.size < 10:
            raise ValueError("plateau trop court : élargir la rampe")
        # Les bords du plateau sont encore dans la transition (bruit qui
        # adoucit les extrémités de la rampe) : on rogne la largeur
        # mesurée de chaque transition (10 % → 90 % du plateau), une fois.
        above10 = np.flatnonzero(local >= 0.1 * top)
        above90 = np.flatnonzero(local >= 0.9 * top)
        left = above90[0] + (above90[0] - above10[0])
        right = above90[-1] - (above10[-1] - above90[-1])
        lo_c, hi_c = max(codes.min(), left), min(codes.max(), right)
        if hi_c - lo_c < 10:
            raise ValueError("plateau trop court après retrait des transitions : élargir la rampe")
        codes = np.arange(lo_c, hi_c + 1)                          # plateau contigu
        n_c = hist[codes]
        # Densité constante : Σ largeurs = nombre de codes sur le plateau.
        e_c = np.full(codes.size, n_c.sum() / codes.size)
        u = 1.0 / np.sqrt(e_c)
        missing = codes[(n_c == 0) & (e_c >= min_counts)]
    elif density == "local":
        # Densité attendue : régression locale quadratique pondérée
        # (gaussienne d'écart-type σ = `smoothing`), sans le code ni ses
        # voisins immédiats ; noyau équivalent L = e₁ᵀ(XᵀWX)⁻¹XᵀW, constant
        # car les codes sont régulièrement espacés. Une moyenne pondérée
        # simple serait biaisée de ≈ σ²·f″/2 là où la densité est courbée.
        def quad_kernel(sig):
            half = 4 * int(np.ceil(sig))
            d = np.arange(-half, half + 1, dtype=np.float64)
            w = np.exp(-0.5 * (d / sig) ** 2)
            w[np.abs(d) <= exclusion] = 0.0
            x = np.column_stack([np.ones_like(d), d, d**2])
            return half, (np.linalg.inv(x.T @ (x * w[:, None])) @ (x * w[:, None]).T)[0]

        def estimate(sig):
            half, k = quad_kernel(sig)
            return np.correlate(np.pad(hist, half, mode="edge"), k, mode="valid"), float(np.sum(k**2))

        expected, k2 = estimate(smoothing)
        # Cohérence : la même régression à σ/2 ne doit différer que par son
        # bruit. Sinon (bords d'une rampe, queues), la densité est trop courbée
        # à l'échelle σ et le code est écarté de l'analyse de DNL.
        narrow, k2n = estimate(max(smoothing / 2, exclusion + 2.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            consistent = np.abs(expected - narrow) < 3.0 * np.sqrt((k2 + k2n) * np.maximum(expected, 1.0))
        populated = expected >= min_counts
        populated[0] = populated[-1] = False
        # Écrêtage : un code extrême anormalement peuplé (pic de saturation)
        # fausse la régression de ses voisins ; on écarte ce bord.
        margin = 4 * int(np.ceil(smoothing))
        for end, inner in ((0, slice(1, 1 + margin)), (hist.size - 1, slice(hist.size - 1 - margin, hist.size - 1))):
            ref = np.median(hist[inner]) if hist[inner].size else 0.0
            if hist[end] > 3.0 * max(ref, 1.0):
                lo_, hi_ = (0, margin + 1) if end == 0 else (hist.size - margin - 1, hist.size)
                populated[lo_:hi_] = False
        missing = np.flatnonzero(populated & (hist == 0))
        use = populated & consistent
        codes = np.flatnonzero(use)
        if codes.size < 10:
            raise ValueError("trop peu de codes peuplés : élargir la plage ou augmenter les données")
        n_c = hist[codes]
        e_c = expected[codes]
        # Incertitude : Poisson du code et des voisins pondérés par le noyau.
        # La DNL des voisins n'y contribue presque pas : la somme des largeurs
        # de codes contigus se télescope (Σ w = t_fin − t_début).
        u = np.sqrt((1.0 + k2) / e_c)
    else:
        raise ValueError(f"density inconnue : {density!r}")

    dnl = n_c / e_c - 1.0
    inl = np.cumsum(dnl)
    # INL relative à la droite des extrémités de la plage analysée (IEEE 1241).
    inl = inl - np.linspace(inl[0], inl[-1], inl.size)
    inl -= inl.mean()
    s2 = float(np.var(dnl) - np.mean(u**2))
    return ADCHistogramResult(
        codes=codes.astype(np.int64), counts=n_c, expected=e_c, dnl=dnl, u_dnl=u, inl=inl,
        missing=missing.astype(np.int64),
        dnl_rms=float(np.sqrt(max(s2, 0.0))), dnl_max=float(np.max(np.abs(dnl))),
    )
