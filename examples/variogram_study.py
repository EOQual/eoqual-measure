"""
Étude : biais de la pépite du variogramme selon la scène et le bruit.

Produit les tableaux 1 à 5 de docs/technique/bruit-variogramme/ (validation.md et comparaison.md) :

1. scènes fractales (fBm, β = 2,5 à 4), avec ou sans flou optique, à deux
   contrastes : biais relatif de la pépite (modèles linéaire et
   quadratique), fraction de pépite F et taux de mesures déclarées
   fiables ;
2. scène urbaine (aplats à bords francs) : effet de la sélection des tuiles ;
3. bruit corrélé (bruit blanc filtré, comme après un traitement) : la
   pépite ne mesure plus le bruit, F le signale ;
4. bruit blanc seul : biais propre de l'estimateur et incertitude ;
5. comparaison avec les autres estimateurs sur scène de `estimate_noise`
   (Ponomarenko, ondelettes, patchs plats, PCA) sur les mêmes images.

Bruit blanc σ = 2 DN sur une scène de niveau moyen 1000 DN, 512 × 512
pixels, tuiles de 16 pixels (défaut) ; R réalisations (scène et bruit tirés à
nouveau) par configuration. Usage : python examples/variogram_study.py
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from eoqual.measure.noise import estimate_noise, variogram_noise
from eoqual.measure.synthetic.scene import natural_scene

SIGMA = 2.0
SHAPE = (512, 512)
R = 8


def urban_scene(rng: np.random.Generator) -> np.ndarray:
    """Aplats rectangulaires superposés (toits, chaussées), bords francs, léger flou."""
    img = np.full(SHAPE, 1000.0)
    for _ in range(60):
        y, x = rng.integers(0, SHAPE[0], 2)
        h, w = rng.integers(20, 120, 2)
        img[y:y + h, x:x + w] += rng.normal(0.0, 40.0)
    return ndimage.gaussian_filter(img, 0.7)


def summary(values: list[float]) -> tuple[float, float]:
    v = np.asarray(values)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def fmt(m: float, u: float, pct: bool = True) -> str:
    return f"{100 * m:+.1f} ± {100 * u:.1f} %" if pct else f"{m:.2f}"


def table_fbm() -> None:
    print("\n### Tableau 1 — scènes fractales, bruit blanc σ = 2 DN (σ² = 4 DN²)\n")
    print("| β | Flou optique σ_PSF | Écart-type de scène (DN) | Biais, modèle linéaire | "
          "Biais, modèle quadratique | F | Fiable |")
    print("|---|---|---|---|---|---|---|")
    for beta in (2.5, 3.0, 3.5, 4.0):
        for psf in (0.0, 0.6):
            for std in (5.0, 20.0):
                lin, quad, frac, rel = [], [], [], []
                for k in range(R):
                    rng = np.random.default_rng(1000 * k + 17)
                    sc = natural_scene(SHAPE, 1000.0, rng, contrast=std / 1000.0, beta=beta,
                                       psf_sigma=psf).photons
                    img = sc + rng.normal(0.0, SIGMA, SHAPE)
                    r = variogram_noise(img)
                    q = variogram_noise(img, model="quadratic")
                    lin.append(r.variance / SIGMA**2 - 1)
                    quad.append(q.variance / SIGMA**2 - 1)
                    frac.append(r.nugget_fraction)
                    rel.append(r.reliable)
                print(f"| {beta} | {psf} | {std:.0f} | {fmt(*summary(lin))} | {fmt(*summary(quad))} | "
                      f"{np.mean(frac):.2f} | {np.mean(rel):.0%} |")


def table_urban() -> None:
    print("\n### Tableau 2 — scène urbaine (aplats à bords francs), bruit blanc σ = 2 DN\n")
    print("| Taille des tuiles | Fraction retenue | Biais, modèle linéaire | F | Fiable |")
    print("|---|---|---|---|---|")
    for tile in (64, 32, 16):
        for keep in (1.0, 0.5, 0.25):
            lin, frac, rel = [], [], []
            for k in range(R):
                rng = np.random.default_rng(1000 * k + 23)
                img = urban_scene(rng) + rng.normal(0.0, SIGMA, SHAPE)
                r = variogram_noise(img, tile=tile, keep_fraction=keep)
                lin.append(r.variance / SIGMA**2 - 1)
                frac.append(r.nugget_fraction)
                rel.append(r.reliable)
            print(f"| {tile} | {keep} | {fmt(*summary(lin))} | {np.mean(frac):.2f} | {np.mean(rel):.0%} |")


def table_pure_noise() -> None:
    print("\n### Tableau 4 — bruit blanc seul : biais propre de l'estimateur (60 tirages)\n")
    print("| Taille des tuiles | Fraction retenue | Biais | Incertitude d'une mesure (u/σ²) |")
    print("|---|---|---|---|")
    for tile in (16, 64):
        for keep in (1.0, 0.5, 0.25):
            v, u = [], []
            for k in range(60):
                img = 1000.0 + np.random.default_rng(k).normal(0.0, SIGMA, SHAPE)
                r = variogram_noise(img, tile=tile, keep_fraction=keep)
                v.append(r.variance / SIGMA**2 - 1)
                u.append(r.u_variance / SIGMA**2)
            m, um = summary(v)
            print(f"| {tile} | {keep} | {fmt(m, um)} | {100 * np.mean(u):.2f} % |")


def table_correlated() -> None:
    print("\n### Tableau 3 — bruit corrélé (bruit blanc σ = 2 DN filtré par une gaussienne)\n")
    print("| Filtre σ_f (pixels) | Variance du bruit par pixel (DN²) | Pépite (DN²) | F | Fiable |")
    print("|---|---|---|---|---|")
    for sf in (0.0, 0.3, 0.5, 0.8, 1.2):
        nug, var, frac, rel = [], [], [], []
        for k in range(R):
            rng = np.random.default_rng(1000 * k + 31)
            sc = natural_scene(SHAPE, 1000.0, rng, contrast=0.005, beta=4.0).photons
            noise = rng.normal(0.0, SIGMA, SHAPE)
            if sf > 0:
                noise = ndimage.gaussian_filter(noise, sf, mode="wrap")
            r = variogram_noise(sc + noise)
            nug.append(r.variance)
            var.append(noise.var())
            frac.append(r.nugget_fraction)
            rel.append(r.reliable)
        print(f"| {sf} | {np.mean(var):.2f} | {np.mean(nug):.2f} | {np.mean(frac):.2f} | {np.mean(rel):.0%} |")


def table_comparison() -> None:
    print("\n### Tableau 5 — comparaison des estimateurs sur scène (biais relatif de σ², σ = 2 DN)\n")
    cases = {
        "lisse (β = 4)": lambda rng: natural_scene(SHAPE, 1000.0, rng, contrast=0.02, beta=4.0).photons
        + rng.normal(0.0, SIGMA, SHAPE),
        "texturée (β = 3, 20 DN)": lambda rng: natural_scene(SHAPE, 1000.0, rng, contrast=0.02, beta=3.0).photons
        + rng.normal(0.0, SIGMA, SHAPE),
        "urbaine": lambda rng: urban_scene(rng) + rng.normal(0.0, SIGMA, SHAPE),
        "bruit filtré (σ_f = 0,5)": lambda rng: natural_scene(SHAPE, 1000.0, rng, contrast=0.005, beta=4.0).photons
        + ndimage.gaussian_filter(rng.normal(0.0, SIGMA, SHAPE), 0.5, mode="wrap") * SIGMA
        / ndimage.gaussian_filter(rng.normal(0.0, SIGMA, SHAPE), 0.5, mode="wrap").std(),
        "rayures de colonnes (σ = 1,5)": lambda rng: natural_scene(SHAPE, 1000.0, rng, contrast=0.005,
                                                                   beta=4.0).photons
        + rng.normal(0.0, SIGMA, SHAPE) + rng.normal(0.0, 1.5, SHAPE[1])[None, :],
    }
    methods = ("variogram", "ponomarenko", "wavelets", "flat_patches", "pca_estimate")
    print("| Scène | " + " | ".join(methods) + " | F (variogramme) |")
    print("|---|" + "---|" * (len(methods) + 1))
    for name, make in cases.items():
        res = {m: [] for m in methods}
        frac = []
        for k in range(R):
            img = make(np.random.default_rng(1000 * k + 41))
            for m in methods:
                v = estimate_noise(img, m).variance
                res[m].append(float(np.ravel(v)[0]) / SIGMA**2 - 1)
            frac.append(variogram_noise(img).nugget_fraction)
        cells = " | ".join(f"{100 * np.mean(res[m]):+.0f} %" for m in methods)
        print(f"| {name} | {cells} | {np.mean(frac):.2f} |")


if __name__ == "__main__":
    table_fbm()
    table_urban()
    table_correlated()
    table_pure_noise()
    table_comparison()
