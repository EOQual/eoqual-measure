<p align="center">
  <img src="resources/eoqual_rounded.png" alt="EOQual" width="280">
</p>

# eoqual-measure

Mesures physiques **contrôlées** de la qualité image d'un système
d'observation de la Terre : la cible (mire, bord choisi, champ uniforme,
source ponctuelle) ou le protocole est connu, et le résultat est une
grandeur physique du système — FTM, variance de bruit, gain — et non un
score de qualité.

Voir **[REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md)** pour le détail
de chaque mesure (principe, architecture, limites, historique des
corrections, bibliographie) et
**[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)** pour la provenance
du code et l'audit des licences.

## Sommaire

1. [Installation](#installation)
2. [Démarrage rapide](#démarrage-rapide)
3. [Catalogue des mesures](#catalogue-des-mesures)
4. [Architecture](#architecture)
5. [Périmètre : eoqual-measure et eoqual-metrics](#périmètre--eoqual-measure-et-eoqual-metrics)
6. [Licence](#licence)
7. [Contribuer](#contribuer)
8. [Credits](#credits)

---

## Installation

### 1. Socle

```bash
pip install git+https://github.com/EOQual/eoqual-measure.git
```

eoqual-core (briques numériques communes à EOQual), numpy, scipy,
opencv-python, scikit-image, PyWavelets, imageio, rich (voir
`pyproject.toml`). Suffit pour toutes les mesures en mode
programmatique.

### 2. Extras optionnels

| Extra | Ajoute | Pour |
|---|---|---|
| `plot` | plotly | Visualisations HTML interactives, CLI `eoqual-measure-mtf`, `SyntheticSlantedEdge.plot` |
| `study` | plotly, pandas | Études de sensibilité (`mtf.slanted_edge.sensitivity`, `examples/mtf_sensitivity_study.py`) |

```bash
pip install "eoqual-measure[plot] @ git+https://github.com/EOQual/eoqual-measure.git"
```

## Démarrage rapide

```python
from eoqual.measure import measure_mtf, estimate_noise, list_measures

list_measures(include_planned=True)          # registre des mesures

# MTF sur une zone contenant un bord incliné (orientation quelconque)
res = measure_mtf(edge_zone)                 # ESF/LSF, modèle PCHIP
res.value_at(0.5), res.frequency_at(0.5)     # MTF@Nyquist, MTF50 (cycles/pixel)
res.details["fwhm"], res.details["rer"]

# Plan Bayer : fréquences ramenées au pixel physique du capteur
from eoqual.core.bayer import extract_bayer_planes, BAYER_SAMPLING_FACTOR
g = extract_bayer_planes(raw, "RGGB")["G"]      # RGGB, BGGR, GRBG ou GBRG
measure_mtf(g, "spatial", pixel_pitch=BAYER_SAMPLING_FACTOR)

# Variance du bruit, par classe de luminance (bruit dépendant du signal)
est = estimate_noise(image, "ponomarenko", num_bins=8)
est.mean, est.variance, est.sigma

# Strehl et énergie encerclée d'une étoile (4 pixels par λ/D)
from eoqual.measure import measure_psf
psf = measure_psf(star, sampling=4.0)
psf.strehl, psf.r80, psf.ee_first_dark_ring
```

Comparer plusieurs algorithmes sur la même zone :

```python
from eoqual.measure.mtf.slanted_edge import (
    EdgeOrientationComparator, MTFComparator, SlantedEdgePreprocessor,
)
angles = EdgeOrientationComparator().compare(edge_zone)       # gradient, PCA, Fourier
frame = SlantedEdgePreprocessor().preprocess(edge_zone)
mtfs = MTFComparator().compare(frame, oversample_factor=4)    # 4 estimateurs
```

Ligne de commande (extra `plot`) :

```bash
eoqual-measure-mtf mire_H.png mire_V.png --zone-labels H V
eoqual-measure-mtf mire_raw.tiff --image-format bayer --bayer-pattern GRBG   # + aberration chromatique latérale
```

Exemples complets dans `examples/` : intercomparaison des estimateurs de
bruit (validation synthétique ou images réelles, mode Bayer), étude de
sensibilité MTF, génération de mires synthétiques.

## Catalogue des mesures

| Mesure | Statut | Point d'entrée | Grandeurs | Algorithmes |
|---|---|---|---|---|
| `mtf_slanted_edge` | ✅ | `eoqual.measure.mtf.measure_mtf` | MTF(f), MTF@Nyquist, MTF50, FWHM, RER, angle, aberration chromatique latérale (Bayer) | Orientation : gradient, PCA (+RANSAC), phase de Fourier. MTF : spatial (dé-entrelacement), spectral (moindres carrés multi-répliques), ESF/LSF (numérique, sigmoïde, PCHIP, Tikhonov), corrélation spectrale. Bootstrap des lignes. |
| `noise_variance` | ✅ | `eoqual.measure.noise.estimate_noise` | Variance (globale ou par classe de luminance), échantillons intra-image | `flat_patches`, `wavelets` (Donoho-Johnstone), `skimage_estimate_sigma`, `ponomarenko` (DCT), `adaptive_segmentation` (SLIC), `pca_estimate` |
| `photon_transfer_curve` | 🔜 | `eoqual.measure.radiometry.ptc` | Gain (e⁻/DN), bruit de lecture, saturation, linéarité | s'appuie sur `noise_variance` |
| `equalization_noise` | 🔜 | `eoqual.measure.radiometry.equalization` | Bruit d'égalisation résiduel (ligne/colonne/pixel) | s'appuie sur `noise_variance` |
| `cnr_target` | 🔜 | `eoqual.measure.contrast.cnr` | Contraste sur bruit sur mire | s'appuie sur `noise_variance` |
| `strehl_encircled_energy` | ✅ | `eoqual.measure.psf.measure_psf` | Rapport de Strehl, énergie encerclée EE(r), r50, r80, EE au premier anneau noir | Fond par couronne, centroïde, recentrage sous-pixel (Fourier), référence d'Airy traitée comme la PSF |

Les mesures prévues (🔜) sont déclarées dans `config.py::MEASURES_CONFIGS`
avec leur emplacement et leur contrat, pour que leur ajout ne demande
aucune réorganisation.

## Architecture

```
src/eoqual/measure/
├── config.py            registre MEASURES_CONFIGS (disponibles + prévues)
├── runner.py            list_measures()
├── synthetic/           scènes à vérité terrain connue : bord incliné à
│                        FTM analytique, bruit injecté de variance connue,
│                        PSF aberrée (Zernike) à Strehl exact
├── mtf/                 measure_mtf(), MTFResult
│   └── slanted_edge/    preprocess, orientation/, estimators/, bootstrap,
│                        chromatic, sensitivity, visualization, cli
├── noise/               estimate_noise(), NoiseEstimate, NoiseVariance
└── psf/                 measure_psf(), PSFMeasurement
```

Principes :

- **une famille = un sous-paquet** (`mtf`, `noise`, puis `radiometry`,
  `contrast`, `psf`), chacun avec une fonction d'entrée simple et un
  résultat typé (`MTFResult`, `NoiseEstimate`) ;
- **comparer avant de choisir** : chaque étape d'une mesure a une classe
  de base et plusieurs implémentations interchangeables, exécutables côte
  à côte sur la même donnée (comparateurs) ;
- **mesures composées** : une mesure peut s'appuyer sur une autre via son
  API publique (la courbe de transfert photonique consommera
  `estimate_noise`, le Strehl consommera la MTF), déclaré dans
  `builds_on` ;
- **validation par simulation** : chaque mesure a son générateur dans
  `synthetic/`, et ses tests vérifient la vérité terrain ;
- **dépendances lourdes en extras** : plotly et pandas ne sont jamais
  importés par le socle ;
- **briques communes dans eoqual-core** : chaîne ESF → LSF → MTF, motifs
  de Bayer, détrend, tuiles, et PSF (Airy, Strehl, énergie encerclée)
  pour les mesures à venir.

## Périmètre : eoqual-measure et eoqual-metrics

| | eoqual-measure | eoqual-metrics |
|---|---|---|
| Entrée | cible ou protocole contrôlé (mire, bord choisi, champ uniforme) | scène quelconque, avec ou sans référence |
| Sortie | grandeur physique du système, avec incertitude | indicateur de qualité |
| MTF | bord incliné ISO 12233 sur zone fournie, 4 estimateurs comparés | MTF@Nyquist sur arêtes naturelles détectées (`aem`, `sasbem`, `mtf`) |

Le calcul ESF → LSF → MTF est partagé via `eoqual-core`
(`eoqual.core.edge`), sans dépendance directe entre les deux modules.

## Licence

MIT — voir `license.txt`. Provenance du code et licences des dépendances :
`THIRD_PARTY_LICENSES.md`.

## Contribuer

Voir `CONTRIBUTING.md` — en particulier la procédure pour ajouter une
nouvelle mesure.

## Credits

Voir `AUTHORS.md`.
