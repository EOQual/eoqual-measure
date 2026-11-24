# Variogramme : limites, formulaire, pistes et références

## 7. Limites

- **Bruit blanc requis** : image brute, avant tout traitement qui corrèle
  les pixels voisins ([§4.3](validation.md#43-bruit-corrélé)). Sur un produit traité, utiliser le NPS
  ([chapitre 12](../radiometrie/12-nps-bruits-structures.md)).
- **Scène lisse à l'échelle du pixel** : la texture fine biaise la pépite
  ([§4.1](validation.md#41-scènes-fractales)) ; F le signale, sans corriger.
- **Bruit fixe et bruit temporel confondus** : sur une seule image, la DSNU
  et la PRNU pixel à pixel sont blanches et entrent dans la pépite. Pour les
  séparer : piles temporelles et [mesures radiométriques](../radiometrie/index.md), chapitres 3, 5 et 11.
- **Bruit structuré** : détecté par l'anisotropie, pas mesuré ; utiliser
  `structured_noise`.
- **Sélection et forme de la courbe** : retenir les tuiles lisses creuse
  légèrement la courbe aux grandes distances (−1,7 % à h = 8 sur un bruit
  blanc, sans effet sur la pépite) ; pour tracer la forme exacte, utiliser
  `variogram(..., keep_fraction=1)`.
- **Écrêtage et quantification** : exclure les tuiles saturées
  (`saturation`) ; un bruit inférieur au pas de quantification
  (σ < 0,3 DN) n'est pas mesurable.
- **Pas de masque** : les zones à exclure (nuages, eau, ombres) se traitent
  en amont, en découpant l'image.

## 8. Formulaire

| Grandeur | Expression |
|---|---|
| Variogramme | γ(h) = ½·E[(Z(x + h) − Z(x))²] = C(0) − C(h) |
| Scène + bruit blanc | γ_Z(h) = γ_S(h) + σ², h ≥ 1 |
| Pépite, modèle linéaire | σ² = 2γ(1) − γ(2) |
| Pépite, modèle quadratique | σ² = 3γ(1) − 3γ(2) + γ(3) |
| Scène fractale 1/f^β | γ_S(h) ∝ h^(β−2) près de l'origine |
| Retrait du plan | γ(h) = σ²·c(h), c(h) = moyenne[1 − ½(H_aa + H_bb) + H_ab], H = D(DᵀD)⁻¹Dᵀ |
| Fraction de pépite | F = σ²/γ(1) ; fiable si F ≥ 0,8, σ² > 3u, \|anisotropie\| ≤ 5 |
| Incertitude | u(σ²) = écart-type des pépites des tuiles / √n |
| SNR | μ/σ ; u(SNR) = SNR·u(σ²)/(2σ²) |

## 9. Pistes

- Masque de zones à exclure (nuages, eau, ombres) en entrée.
- Mode Bayer : pépite par plan de la mosaïque brute.
- Correction du biais de scène à partir de la forme de γ(h) aux distances
  2 à 6 (modèle en h^α ajusté), au lieu de seulement le signaler par F.
- Modèle a + b·μ pondéré par l'incertitude propre de chaque classe de
  luminance (regroupement des tuiles en classes).

## Références

- P. J. Curran, J. L. Dungan, « Estimation of signal-to-noise: a new
  procedure applied to AVIRIS data », *IEEE Transactions on Geoscience and
  Remote Sensing* 27(5), 620-628, 1989 (SNR par la pépite du variogramme).
- G. Matheron, « Principles of geostatistics », *Economic Geology* 58(8),
  1246-1266, 1963 (variogramme, effet de pépite).
