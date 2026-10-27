<p align="center">
  <img src="resources/eoqual_rounded.png" alt="EOQual" width="280">
</p>

# eoqual-measure

> **Projet en cours de développement.** Les mesures sont validées sur
> données simulées (vérité connue) mais ne sont pas encore qualifiées sur
> données réelles ; l'API peut évoluer et des bugs ou des anomalies sont
> possibles. Merci de les signaler par une issue. Plan de qualification
> sur données réelles : `examples/PLAN_TESTS_DONNEES_REELLES.md`.

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

# Radiométrie : obscurité, puis courbe de transfert photonique
from eoqual.measure.radiometry import measure_dark_frame, measure_ptc, snr_at_levels
dark = measure_dark_frame(dark_stack)            # (K, H, W) : offset, DSNU, b
ptc = measure_ptc(flat_pairs, dark)              # une paire par niveau
ptc.gain, ptc.read_noise_e, ptc.saturation_e     # e⁻/DN, e⁻, e⁻
snr_at_levels(ptc, [100.0, 1000.0]).snr
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
sensibilité MTF, génération de mires synthétiques, études de domaine de
validité des mesures radiométriques (`pushbroom_texture_bias_study.py`,
`structured_noise_detection_study.py`) et plan de tests sur données
réelles.

## Catalogue des mesures

| Mesure | Statut | Point d'entrée | Grandeurs | Algorithmes |
|---|---|---|---|---|
| `mtf_slanted_edge` | ✅ | `eoqual.measure.mtf.measure_mtf` | MTF(f), MTF@Nyquist, MTF50, FWHM, RER, angle, aberration chromatique latérale (Bayer) | Orientation : gradient, PCA (+RANSAC), phase de Fourier. MTF : spatial (dé-entrelacement), spectral (moindres carrés multi-répliques), ESF/LSF (numérique, sigmoïde, PCHIP, Tikhonov), corrélation spectrale. Bootstrap des lignes. |
| `noise_variance` | ✅ | `eoqual.measure.noise.estimate_noise` | Variance (globale ou par classe de luminance), échantillons intra-image | `flat_patches`, `wavelets` (Donoho-Johnstone), `skimage_estimate_sigma`, `ponomarenko` (DCT), `adaptive_segmentation` (SLIC), `pca_estimate` |
| `dark_frame` | ✅ | `eoqual.measure.radiometry.measure_dark_frame` | Offset, bruit à signal nul b, DSNU (ligne/colonne/pixel), stationnarité | Moyenne et variance temporelles, décomposition sans biais ; pile matricielle ou bloc push broom |
| `dark_current` | ✅ | `eoqual.measure.radiometry.measure_dark_current` | Courant par pixel, DSNU du courant, pixels chauds, gain poissonnien, loi en température | Régression masquée par pixel, loi T³·exp(−E_g/2kT) |
| `temporal_noise` | ✅ | `eoqual.measure.radiometry.measure_temporal_noise` | Variance temporelle, signal corrigé de l'obscurité | Différence de paire ou variance de pile |
| `photon_transfer_curve` | ✅ | `eoqual.measure.radiometry.measure_ptc` | Gain (e⁻/DN), bruit de lecture, saturation (EMVA), linéarité, χ² | Moindres carrés pondérés σ² = a·μ + b ; s'appuie sur `temporal_noise` |
| `prnu` | ✅ | `eoqual.measure.radiometry.measure_prnu` | Carte de réponse, PRNU totale et EMVA, vignettage (BF) / PRNU (HF) | Surface de Legendre, décomposition de la HF |
| `linearity` | ✅ | `eoqual.measure.radiometry.measure_linearity` | Erreur de linéarité EMVA, courbure commune et par pixel | Ajustement quadratique, méthode delta |
| `defective_pixels` | ✅ | `eoqual.measure.radiometry.map_defective_pixels` | Pixels bloqués, morts, chauds, bruyants, réponse aberrante | Tests du χ², médiane locale |
| `rts_pixels` | ✅ | `eoqual.measure.radiometry.find_rts_pixels` | Pixels télégraphiques : niveaux, saut, bascules | Mélange à deux niveaux (EM, BIC), Viterbi |
| `adc_histogram` | ✅ | `eoqual.measure.radiometry.adc_histogram` | DNL, INL, codes manquants | Méthode de l'histogramme (IEEE 1241) |
| `pushbroom_temporal_noise` | ✅ | `eoqual.measure.radiometry.measure_pushbroom_noise` | Modèle σ²(μ) d'un push broom en vol, diagnostic de texture | Filtre de différence séparable, moindres carrés pondérés |
| `snr_model` | ✅ | `eoqual.measure.radiometry.snr_at_levels` | SNR et incertitude aux niveaux demandés | Propagation de la covariance (a, b) |
| `spatial_noise` | ✅ | `eoqual.measure.radiometry.measure_spatial_noise` | Bruit spatial fixe, bandes, aller/retour (whisk broom) | Pile, push broom, whisk broom |
| `equalization_noise` | ✅ | `eoqual.measure.radiometry.measure_equalization_noise` | FPN résiduel, BF, HF, spectre de la ligne moyenne | Lissage gaussien, bruit filtré corrigé |
| `noise_power_spectrum` | ✅ | `eoqual.measure.noise.noise_power_spectrum` | NPS 2D, coupes, moyenne radiale | Welch, fenêtre de Hann (ou `perdecomp`) |
| `structured_noise` | ✅ | `eoqual.measure.noise.find_structured_noise` | Motifs périodiques (fixes / temporels), bruit de ligne et de colonne | Seuils Gamma par région, bornés sur les axes |
| `cnr_target` | ✅ | `eoqual.measure.contrast.measure_cnr` | Contraste sur bruit sur mire | Plages d'Otsu érodées ou fournies |
| `strehl_encircled_energy` | ✅ | `eoqual.measure.psf.measure_psf` | Rapport de Strehl, énergie encerclée EE(r), r50, r80, EE au premier anneau noir | Fond par couronne, centroïde, recentrage sous-pixel (Fourier), référence d'Airy traitée comme la PSF |

Les mesures sont déclarées dans `config.py::MEASURES_CONFIGS` avec leur
emplacement et leur contrat ; une mesure prévue (🔜) y figure avant d'être
implémentée, pour que son ajout ne demande aucune réorganisation. Les
mesures radiométriques sont décrites dans `REFERENCE_TECHNIQUE.md`,
partie IV (tableau de synthèse par type de capteur, matriciel, push broom
ou whisk broom), et leur simulateur dans la partie V.

## Architecture

```
src/eoqual/measure/
├── config.py            registre MEASURES_CONFIGS (disponibles + prévues)
├── runner.py            list_measures()
├── synthetic/           scènes à vérité terrain connue : bord incliné à
│                        FTM analytique, bruit injecté de variance connue,
│                        PSF aberrée (Zernike) à Strehl exact ; simulateur
│                        de capteur (matrice, push broom, whisk broom)
├── mtf/                 measure_mtf(), MTFResult
│   └── slanted_edge/    preprocess, orientation/, estimators/, bootstrap,
│                        chromatic, sensitivity, visualization, cli
├── noise/               estimate_noise(), NoiseEstimate, NoiseVariance ;
│                        noise_power_spectrum(), find_structured_noise()
├── radiometry/          obscurité, courant d'obscurité, bruit temporel, PTC,
│                        PRNU, linéarité, défauts, RTS, convertisseur,
│                        push broom, SNR, bruit spatial et d'égalisation
├── contrast/            measure_cnr(), CNRResult
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
  API publique (la courbe de transfert photonique consomme
  `measure_temporal_noise` et `measure_dark_frame`), déclaré dans
  `builds_on` ;
- **validation par simulation** : chaque mesure a son générateur dans
  `synthetic/`, et ses tests vérifient la vérité terrain ;
- **dépendances lourdes en extras** : plotly et pandas ne sont jamais
  importés par le socle ;
- **briques communes dans eoqual-core** : chaîne ESF → LSF → MTF, motifs
  de Bayer, détrend, tuiles, PSF (Airy, Strehl, énergie encerclée), et
  décomposition périodique + lisse (option `perdecomp` du NPS).

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
