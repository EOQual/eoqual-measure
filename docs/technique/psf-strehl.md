# Strehl et énergie encerclée

Module `eoqual.measure.psf`, fonction `measure_psf`.

## 1. Objet

Qualité d'une PSF observée sur une source ponctuelle isolée (étoile,
collimateur de banc) : rapport de Strehl (pic de la PSF rapporté à celui
de la tache d'Airy de même pupille et même échantillonnage) et énergie
encerclée EE(r) (fraction de l'énergie dans un disque de rayon r).

## 2. Chaîne de calcul (`measure_psf`)

1. **Fond** : médiane d'une couronne au bord de l'imagette (largeur 1/10
   du petit côté, au moins 2 px), ou valeur fournie, ou aucun retrait.
2. **Centroïde** : barycentre des valeurs positives dans un disque de
   rayon 3 λ/D (5 px sans échantillonnage connu) autour du maximum,
   itéré.
3. **Recentrage** : translation sous-pixel de Fourier
   (`eoqual.core.fourier.translate`) qui amène le centroïde au centre
   géométrique `(n - 1) / 2`, là où est centrée la référence d'Airy.
   Sans recentrage, le pic échantillonné dépend de la position du centre
   dans le pixel.
4. **Normalisation** à la somme 1 (énergie contenue dans l'imagette).
5. **Énergie encerclée** (`eoqual.core.psf.encircled_energy`), r50, r80.
6. **Strehl** si `sampling` (pixels par λ/D) est fourni : référence
   `eoqual.core.psf.airy_psf` sur la même grille, **soumise au même retrait
   de fond et à la même normalisation que la PSF** ; puis rapport des
   pics. EE au premier anneau noir (1.22 λ/D).

## 3. Pourquoi traiter la référence comme la PSF

Le fond estimé sur la couronne contient aussi les ailes de la PSF ; le
retirer ôte de l'énergie à la PSF, que la normalisation reporte sur le
pic. Avec une référence d'Airy normalisée sans ce traitement, le Strehl
serait surestimé de +0.5 à +1 % sur des PSF synthétiques sans bruit. En
appliquant à la référence la même estimation de fond (même couronne) et
la même normalisation, l'écart retombe à ±0.2 %.

## 4. Validation

`eoqual.measure.synthetic.SyntheticPSF` : PSF de Fraunhofer d'une pupille
circulaire portant des aberrations de Zernike connues (défocalisation,
astigmatisme, coma, sphérique ; vagues rms, normalisation de Noll),
décalage sous-pixel, bruit de photon, fond, bruit de lecture. Le Strehl
exact est le rapport des pics de la même simulation avec et sans
aberration ; il suit l'approximation de Maréchal `exp(-(2πσ)²)` pour
σ ≲ 0.1 vague.

| Cas (4 px par λ/D sauf mention) | Strehl exact | Mesuré | Écart |
|---|---|---|---|
| parfait | 1.0000 | 1.0015 | +0.15 % |
| parfait, décalé (0.37, −1.4) px | 1.0000 | 1.0017 | +0.17 % |
| défocalisation 0.05 vague | 0.9048 | 0.9061 | +0.15 % |
| coma 0.07 + astigmatisme 0.04, décalé | 0.7725 | 0.7719 | −0.07 % |
| défocalisation 0.05, 10⁶ photons, fond 20, lecture 5 | 0.9048 | 0.9047 | −0.01 % |
| défocalisation 0.05, 10⁵ photons, fond 50, lecture 10 | 0.9048 | 0.9124 | +0.84 % |
| sphérique 0.06, 2 px par λ/D (Nyquist) | 0.8654 | 0.8664 | +0.12 % |
| défocalisation 0.15 vague | 0.3723 | 0.3730 | +0.19 % |

Le résidu de +0.15 % sans bruit vient de la discrétisation de la pupille
simulée (128 px) face à l'Airy analytique ; à faible flux, le bruit sur le
pic domine.

## 5. Limites

- **Échantillonnage** : sous 2 px par λ/D, le pic est sous-échantillonné
  et le Strehl n'a plus de sens (avertissement émis).
- **Intégration sur le pixel** non modélisée : la référence est une Airy
  échantillonnée ponctuellement. Négligeable à 4 px par λ/D, de l'ordre
  du pourcent à Nyquist.
- **Champ fini** : EE est normalisée à l'énergie de l'imagette ; l'énergie
  hors champ la relève légèrement (0.853 au lieu de 0.838 au premier
  anneau noir pour 129 px à 4 px par λ/D). Prendre une imagette d'au
  moins une trentaine de λ/D de côté pour une EE absolue.
- **Pupille** circulaire non obturée pour la référence ; une obturation
  centrale ou des araignées demandent une référence adaptée (à ajouter à
  `eoqual.core.psf`).
- **Bande spectrale** : la référence est monochromatique ; pour une
  bande large, `sampling` doit être pris à la longueur d'onde effective.

## 6. Références

- V. N. Mahajan, « Strehl ratio for primary aberrations: some analytical
  results for circular and annular pupils », *JOSA* 72(9), 1982,
  p. 1258-1266.
- R. J. Noll, « Zernike polynomials and atmospheric turbulence », *JOSA*
  66(3), 1976, p. 207-211.
- M. Born, E. Wolf, *Principles of Optics*, Cambridge University Press
  (tache d'Airy, énergie encerclée).
