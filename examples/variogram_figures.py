"""
Figures du chapitre 14 de la partie IV (variogramme) : principe, chaîne de
calcul, extrapolation à l'origine, biais en fonction de la fraction de
pépite, pièges. Données simulées, graines fixées.

Usage : python examples/variogram_figures.py  →  resources/variogram_*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from eoqual.measure.noise import variogram, variogram_noise  # noqa: E402
from eoqual.measure.synthetic.scene import natural_scene  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "resources"
SIGMA = 2.0
SHAPE = (512, 512)


def gamma_1d(z: np.ndarray, hmax: int) -> np.ndarray:
    return np.array([0.5 * np.mean((z[h:] - z[:-h]) ** 2) for h in range(1, hmax + 1)])


# --------------------------------------------------------------------------
# 1. Principe
# --------------------------------------------------------------------------

def figure_principe() -> None:
    rng = np.random.default_rng(3)
    n, rho, sig = 20000, 0.95, 4.0
    s = np.zeros(n)
    for i in range(1, n):  # signal de scène corrélé (AR(1), longueur ~20 px)
        s[i] = rho * s[i - 1] + rng.normal(0.0, 20.0 * np.sqrt(1 - rho**2))
    s += 100.0
    z = s + rng.normal(0.0, sig, n)
    hs = np.arange(1, 101)
    g_s, g_z = gamma_1d(s, 100), gamma_1d(z, 100)
    sill = g_z[70:].mean()

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    x = np.arange(160)
    ax[0].plot(x, s[:160], "k", lw=2, label="scène S")
    ax[0].plot(x, z[:160], ".", ms=4, color="tab:red", label=f"mesure Z = S + N (σ = {sig:g})")
    top = z[:160].max()
    for i0, h, c, ytxt in ((30, 1, "tab:blue", top + 6), (95, 15, "tab:green", top + 14)):
        ax[0].annotate("", xy=(i0 + h, z[i0 + h]), xytext=(i0, z[i0]),
                       arrowprops=dict(arrowstyle="<->", color=c, lw=2))
        ax[0].annotate(f"paire distante de h = {h}", xy=(i0 + h / 2, max(z[i0], z[i0 + h])),
                       xytext=(i0 - 5, ytxt), color=c, arrowprops=dict(arrowstyle="-", color=c, lw=0.6))
    ax[0].set_ylim(z[:160].min() - 15, top + 22)
    ax[0].set(title="① On compare les pixels deux à deux, à distance h", xlabel="pixel", ylabel="DN")
    ax[0].legend(loc="lower left", fontsize=9)

    ax[1].plot(hs, g_z, "o", ms=3, color="tab:red", label="γ de la mesure Z")
    ax[1].plot(hs, g_s, "k-", lw=1.5, label="γ de la scène seule")
    ax[1].axhline(sill, color="gray", ls="--")
    ax[1].text(52, sill * 1.03, "palier = var(S) + σ²", color="gray")
    ax[1].annotate("", xy=(70, 0), xytext=(70, sill), arrowprops=dict(arrowstyle="<->", color="gray"))
    ax[1].axvline(45, color="tab:purple", ls=":")
    ax[1].text(46, sill * 0.55, "portée\n(taille des objets)", color="tab:purple")
    ax[1].set(title="② γ(h) = ½·moyenne[(Z(x+h) − Z(x))²]", xlabel="h (pixels)", ylabel="γ (DN²)",
              xlim=(0, 100), ylim=(0, sill * 1.2))
    ax[1].legend(loc="lower right", fontsize=9)

    hh = np.linspace(0, 6, 30)
    coef = np.polyfit(hs[:2], g_z[:2], 1)
    ax[2].plot(hs[:10], g_z[:10], "o", color="tab:red", label="γ de la mesure Z")
    ax[2].plot(hs[:10], g_s[:10], "k-", lw=1.5, label="γ de la scène seule (→ 0)")
    ax[2].plot(hh, np.polyval(coef, hh), "b--", label="droite par h = 1, 2")
    ax[2].plot(0, coef[1], "bs", ms=9)
    ax[2].axhline(sig**2, color="tab:orange", ls=":", label=f"σ² vrai = {sig**2:g}")
    ax[2].annotate(f"pépite = 2γ(1) − γ(2) = {coef[1]:.1f}\n→ σ ≈ {np.sqrt(coef[1]):.2f}",
                   xy=(0, coef[1]), xytext=(4.5, 25), arrowprops=dict(arrowstyle="->"),
                   bbox=dict(fc="white", ec="none", alpha=0.9))
    ax[2].set(title="③ Extrapolation en h = 0 : la pépite = σ²", xlabel="h (pixels)", ylabel="γ (DN²)",
              xlim=(-0.3, 10), ylim=(0, g_z[9] * 1.1))
    ax[2].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "variogram_principe.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# 2. Chaîne de calcul
# --------------------------------------------------------------------------

def figure_chaine() -> None:
    steps = [
        ("Image brute\n(DN)", "avant dématriçage,\nrééchantillonnage,\ncompression"),
        ("Tuiles 16 × 16", "sans recouvrement ;\nexclusion des tuiles\nsaturées"),
        ("Retrait d'un plan", "par tuile ;\neffet sur γ corrigé\nexactement"),
        ("Sélection", "50 % des tuiles les\nplus lisses (gradient\nde la tuile lissée σ = 3)"),
        ("γ(1), γ(2), γ(3)", "le long des lignes\net des colonnes"),
        ("Pépite par tuile", "linéaire : 2γ(1) − γ(2)\nquadratique :\n3γ(1) − 3γ(2) + γ(3)"),
    ]
    outs = ["σ² ± u (dispersion\nentre tuiles)", "F = σ²/γ(1)\n→ fiable ?", "anisotropie\nlignes/colonnes",
            "SNR = μ/σ", "a + b·μ\n(tuiles en luminance)"]
    fig, ax = plt.subplots(figsize=(17, 5.2))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    w, h, y0 = 2.45, 1.75, 2.75
    for k, (title, sub) in enumerate(steps):
        x0 = 0.2 + k * 2.8
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.05", fc="#dbe9f6", ec="#1f4e79"))
        ax.text(x0 + w / 2, y0 + h - 0.3, title, ha="center", va="center", fontsize=10.5, weight="bold")
        ax.text(x0 + w / 2, y0 + 0.62, sub, ha="center", va="center", fontsize=8.5, color="#333333")
        if k < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x0 + w + 0.02, y0 + h / 2), (x0 + 2.8 - 0.02, y0 + h / 2),
                                         arrowstyle="-|>", mutation_scale=15, color="#1f4e79"))
    xm = 0.2 + 5 * 2.8 + w / 2
    for k, txt in enumerate(outs):
        x0 = 0.6 + k * 3.3
        ax.add_patch(FancyBboxPatch((x0, 0.25), 2.7, 0.95, boxstyle="round,pad=0.05", fc="#e2f0d9", ec="#38761d"))
        ax.text(x0 + 1.35, 0.72, txt, ha="center", va="center", fontsize=9)
        ax.add_patch(FancyArrowPatch((xm, y0 - 0.08), (x0 + 1.35, 1.25), arrowstyle="-|>", mutation_scale=12,
                                     color="#38761d", connectionstyle="arc3,rad=0.0", lw=0.8))
    ax.text(8.5, 5.0, "Chaîne de calcul de variogram_noise", ha="center", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "variogram_chaine.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. Extrapolation selon la scène
# --------------------------------------------------------------------------

def figure_extrapolation() -> None:
    cases = [("scène lisse (β = 4, 20 DN)", 4.0, 0.0, 0.02),
             ("scène texturée (β = 3, 20 DN)", 3.0, 0.0, 0.02),
             ("texture fine floutée (β = 2,5, σ_PSF = 0,6, 20 DN)", 2.5, 0.6, 0.02)]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, (title, beta, psf, c) in zip(axs, cases):
        rng = np.random.default_rng(17)
        sc = natural_scene(SHAPE, 1000.0, rng, contrast=c, beta=beta, psf_sigma=psf).photons
        img = sc + rng.normal(0.0, SIGMA, SHAPE)
        v = variogram(img, max_lag=6)
        vs = variogram(sc, max_lag=6)
        r = variogram_noise(img)
        q = variogram_noise(img, model="quadratic")
        h = np.r_[0, v.lags]
        ax.plot(v.lags, v.gamma, "o", color="tab:red", label="γ de l'image")
        ax.plot(vs.lags, vs.gamma, "k-", lw=1.2, label="γ de la scène seule")
        hh = np.linspace(0, 3, 20)
        ax.plot(hh, np.polyval(np.polyfit([1, 2], v.gamma[:2], 1), hh), "b--", lw=1.2,
                label=f"linéaire : σ² = {r.variance:.2f}")
        ax.plot(hh, np.polyval(np.polyfit([1, 2, 3], v.gamma[:3], 2), hh), "g:", lw=1.5,
                label=f"quadratique : σ² = {q.variance:.2f}")
        ax.axhline(SIGMA**2, color="tab:orange", ls="--", lw=1, label="σ² vrai = 4")
        ax.set(xlim=(-0.2, 6.3), ylim=(min(0, r.variance) - 1, v.gamma[-1] * 1.1), xlabel="h (pixels)",
               ylabel="γ (DN²)", title=f"{title}\nF = {r.nugget_fraction:.2f} → "
                                        f"{'fiable' if r.reliable else 'non fiable'}")
        ax.legend(fontsize=8, loc="lower right" if r.reliable else "upper left")
        del h
    fig.tight_layout()
    fig.savefig(OUT / "variogram_extrapolation.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. Biais en fonction de F
# --------------------------------------------------------------------------

def figure_biais_f() -> None:
    pts = []
    for beta in (2.5, 3.0, 3.5, 4.0):
        for psf in (0.0, 0.6):
            for c in (0.005, 0.01, 0.02, 0.04):
                for k in range(3):
                    rng = np.random.default_rng(1000 * k + int(100 * beta) + int(10 * psf))
                    try:
                        sc = natural_scene(SHAPE, 1000.0, rng, contrast=c, beta=beta, psf_sigma=psf).photons
                    except ValueError:
                        continue
                    r = variogram_noise(sc + rng.normal(0.0, SIGMA, SHAPE))
                    pts.append((r.nugget_fraction, r.variance / SIGMA**2 - 1, beta))
    p = np.array(pts)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for beta, col in zip((2.5, 3.0, 3.5, 4.0), ("tab:red", "tab:orange", "tab:green", "tab:blue")):
        m = p[:, 2] == beta
        ax.plot(p[m, 0], 100 * p[m, 1], "o", ms=5, color=col, alpha=0.8, label=f"β = {beta}")
    ax.axvspan(0.8, 1.05, color="tab:green", alpha=0.08)
    ax.axvline(0.8, color="tab:green", ls="--")
    ax.text(0.81, -95, "fiable\n(F ≥ 0,8)", color="tab:green")
    ax.axhspan(-9, 9, color="gray", alpha=0.15)
    ax.text(-0.45, 12, "±9 %", color="gray")
    ax.set(xlim=(-0.5, 1.05), ylim=(-140, 60), xlabel="fraction de pépite F = σ² / γ(1)",
           ylabel="biais de σ² (%)",
           title="Le biais est faible quand le bruit domine l'écart entre voisins")
    ax.legend(title="texture de la scène", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "variogram_biais_F.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. Pièges
# --------------------------------------------------------------------------

def urban_scene(rng: np.random.Generator) -> np.ndarray:
    img = np.full(SHAPE, 1000.0)
    for _ in range(60):
        y, x = rng.integers(0, SHAPE[0], 2)
        hh, ww = rng.integers(20, 120, 2)
        img[y:y + hh, x:x + ww] += rng.normal(0.0, 40.0)
    return ndimage.gaussian_filter(img, 0.7)


def figure_pieges() -> None:
    rng = np.random.default_rng(5)
    smooth = natural_scene(SHAPE, 1000.0, rng, contrast=0.005, beta=4.0).photons
    noise = rng.normal(0.0, SIGMA, SHAPE)
    stripes = rng.normal(0.0, 1.5, SHAPE[1])[None, :]
    cases = [
        ("A. bruit blanc", smooth + noise, {}),
        ("B. bruit filtré (σ_f = 0,8) :\nimage traitée", smooth + ndimage.gaussian_filter(noise, 0.8), {}),
        ("C. scène urbaine, toutes les tuiles", urban_scene(np.random.default_rng(23)) + noise,
         dict(keep_fraction=1.0)),
        ("C'. même scène, 50 % des tuiles\nles plus lisses (défaut)", urban_scene(np.random.default_rng(23)) + noise,
         {}),
        ("D. rayures de colonnes (σ = 1,5)", smooth + noise + stripes, {}),
    ]
    fig, axs = plt.subplots(2, 5, figsize=(21, 8))
    for k, (title, img, kw) in enumerate(cases):
        axs[0, k].imshow(img[:256, :256], cmap="gray")
        axs[0, k].set_title(title, fontsize=10)
        axs[0, k].axis("off")
        v = variogram(img, max_lag=8, **kw)
        r = variogram_noise(img, **kw)
        a = axs[1, k]
        a.plot(v.lags, v.gamma_rows, "o-", ms=3, color="tab:blue", label="le long des lignes")
        a.plot(v.lags, v.gamma_cols, "s-", ms=3, color="tab:green", label="le long des colonnes")
        a.axhline(SIGMA**2, color="tab:orange", ls="--", label="σ² vrai = 4")
        top = max(v.gamma_rows.max(), v.gamma_cols.max(), 4.0)
        a.set(xlim=(0, 8.5), ylim=(min(0, r.variance) - 0.5, top * 1.15), xlabel="h (pixels)")
        a.set_title(f"σ² = {r.variance:.2f} ± {r.u_variance:.2f}   F = {r.nugget_fraction:.2f}\n"
                    f"anisotropie {r.anisotropy:+.1f} u   {'fiable' if r.reliable else 'NON fiable'}",
                    fontsize=9.5)
    axs[1, 0].set_ylabel("γ (DN²)")
    axs[1, 0].legend(fontsize=8)
    fig.suptitle("Cas favorable et pièges (σ vrai = 2 DN, tuiles de 16 pixels)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "variogram_pieges.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    figure_principe()
    figure_chaine()
    figure_extrapolation()
    figure_biais_f()
    figure_pieges()
    print("figures écrites dans", OUT)
