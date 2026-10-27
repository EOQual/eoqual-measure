# Licences tierces et provenance du code — inventaire et statut

> **Avertissement.** Cet inventaire est tenu au meilleur effort. Si une
> erreur y est identifiée (source mal attribuée, statut incorrect), elle
> sera corrigée dès que signalée — et si nécessaire, l'implémentation
> concernée sera retirée plutôt que sa licence « arrangée » a posteriori.
> Pour signaler un problème, voir `CONTRIBUTING.md`.

**État actuel** : aucun code tiers n'est inclus (pas de dossier
`vendor/`) ; toutes les dépendances sont permissives ; tout le code du
paquet a été écrit par l'auteur, à partir de méthodes publiées.

## 1. Légende

- 🔴 **Bloquant** — licence copyleft ou absente, atteignable depuis le socle.
- 🟢 **Compatible** — licence permissive (MIT/BSD/Apache), ou code écrit
  pour ce projet.

## 2. Dépendances

Licences relevées dans les métadonnées des paquets installés
(`importlib.metadata`, 2026-10-11).

| Paquet | Usage | Licence | Statut |
|---|---|---|---|
| eoqual-core | briques numériques communes (`edge`, `bayer`, `detrend`, `tiling`, `psf`, `fourier`) | MIT (même auteur) | 🟢 |
| numpy | partout | BSD-3-Clause (+ 0BSD, MIT, Zlib, CC0 pour des composants) | 🟢 |
| scipy | FFT, DCT, ajustements, filtres | BSD-3-Clause | 🟢 |
| opencv-python | `cv2.Sobel` (canonicalisation, détecteur PCA) | Apache-2.0 | 🟢 |
| scikit-image | `estimate_sigma`, SLIC | BSD-3-Clause | 🟢 |
| PyWavelets | ondelettes (`wavelets`) | MIT, BSD-3-Clause | 🟢 |
| imageio | lecture/écriture d'images | BSD-2-Clause | 🟢 |
| rich | tableaux console | MIT | 🟢 |
| plotly (extra `plot`) | visualisations HTML | MIT | 🟢 |
| pandas (extra `study`) | études de sensibilité | BSD-3-Clause | 🟢 |

## 3. Provenance du code

Tout le code est écrit par l'auteur. Les méthodes implémentées sont
publiées ; leurs références figurent dans `REFERENCE_TECHNIQUE.md`.

| Sous-paquet | Méthodes | Références principales | Statut |
|---|---|---|---|
| `mtf/slanted_edge/`, `synthetic/slanted_edge.py` | Bord incliné, ESF/LSF, PCHIP, Tikhonov, RER | ISO 12233:2017 ; Burns (2000) ; Kohm (2004) ; Fritsch & Carlson (1980) ; Xie et al. (2018) ; Harrington et al. (2015) | 🟢 |
| `noise/variance.py` — `ponomarenko` | DCT par blocs, sélection basse fréquence, classes de luminance | Ponomarenko et al. (2007) ; Colom & Buades (2013) | 🟢 |
| `noise/variance.py` — autres | Ondelettes, patchs plats, superpixels SLIC, PCA de patchs | Donoho & Johnstone (1994) ; Achanta et al. (2012) ; Pyatykh et al. (2013) ; Chen et al. (2015) | 🟢 |
| `psf/`, `synthetic/psf.py` | Strehl, énergie encerclée, PSF de Fraunhofer aberrée | Mahajan (1982) ; Noll (1976) ; Born & Wolf | 🟢 |
| `radiometry/` | Obscurité, PTC, courant d'obscurité, PRNU, linéarité, défauts, RTS (EM, BIC, Viterbi), histogramme du convertisseur, bruit d'un push broom (filtre de différence), bruit spatial et d'égalisation, SNR | EMVA 1288 ; Janesick (2001, 2007) ; IEEE Std 1241 ; Immerkær (1996) ; Bartlett (1946) ; Taylor (1997) | 🟢 |
| `noise/spectrum.py` | NPS de Welch, détection de raies (seuils Gamma), bruit de ligne | Welch (1967) ; ISO 15739 ; Moisan (2011), via `eoqual.core.fourier.perdecomp` | 🟢 |
| `contrast/` | Contraste sur bruit, seuil d'Otsu | Otsu (1979) | 🟢 |
| `synthetic/` — simulateur de capteur (`sensor`, `patterns`, `scene`, `dark`, `response`, `temporal_effects`, `processing`) | Modèle de pixel EMVA, synthèse spectrale 1/f^β, scènes fBm, courant d'obscurité, non-linéarité, RTS, convertisseur à pas inégaux | EMVA 1288 ; Downey, *Think DSP* (CC-BY 4.0, idée de la synthèse spectrale, aucun code repris) ; guide CCD d'Astropy (BSD-3, modèle d'image artificielle, aucun code repris) | 🟢 |
| `resources/modele_pixel_emva.{svg,png}` | Schéma du modèle de pixel | Redessiné par l'auteur d'après EMVA 1288 et une figure de K. Douglass (« Modeling noise for image simulations ») ; aucune image reprise | 🟢 |
| `config.py`, `runner.py`, `noise/estimate.py`, `mtf/__init__.py`, tests | — | — | 🟢 |
