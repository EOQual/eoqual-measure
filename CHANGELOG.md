# Change Log

## Version History

Non publié

:   -   **Nouvelle mesure `variogram_noise`** (`eoqual.measure.noise`) :
        variance du bruit et SNR d'une image brute par la pépite du
        variogramme (Curran et Dungan, 1989), aussi accessible par
        `estimate_noise(image, "variogram")`. Tuiles de 16 pixels, retrait
        d'un plan corrigé exactement, sélection des tuiles les plus lisses
        sur un critère insensible au bruit, extrapolation linéaire ou
        quadratique, pépite par direction (anisotropie : rayures),
        fraction de pépite F = σ²/γ(1) et verdict `reliable`, modèle
        σ² = a + b·μ compatible avec `snr_at_levels`. Fonction `variogram`
        pour tracer la courbe.
    -   Document `docs/technique/bruit-variogramme.md` (principe, figures,
        validation, comparaison avec Ponomarenko et les autres
        estimateurs, cas réel) ; tests `tests/test_variogram.py` (11,
        robustes sur 30 graines) ; étude `examples/variogram_study.py` ;
        figures `examples/variogram_figures.py` ; notebook
        `examples/real_scene_variogram.ipynb`.
    -   **Documentation réorganisée** : `REFERENCE_TECHNIQUE.md` est
        découpé en documents spécialisés dans `docs/technique/` (MTF,
        estimateurs de bruit, variogramme, Strehl, mesures radiométriques
        avec un fichier par chapitre, simulateur), publiés avec l'API et
        les notes de version par Sphinx et MyST (`docs/`). Les historiques
        de corrections sont retirés des documents : ce qui change d'une
        version à l'autre est dans ce fichier, les variantes de méthode
        écartées sont décrites au présent dans chaque document. Les
        figures passent de `resources/` à `docs/technique/figures/`.
    -   Exemples de la courbe bruit-luminance (estimateurs de bruit)
        remis dans la notation σ² = a + b·I de 0.4.0 : ils inversaient
        encore a et b.

0.4.0 (2026-11-18)

:   -   **Modèle de bruit noté σ² = a + b·μ** (a plancher, b pente = 1/K),
        comme dans les autres modules EOQual : un bruit stationnaire
        s'écrit a = σ², b = 0. Méthodes et résultats inchangés.
        **Changement d'API** (lettres inversées par rapport à 0.3.1) :
        -   `PTCResult`, `PushbroomNoiseResult`, `SensorTruth` : `a` est le
            plancher, `b` la pente ; `cov` est ordonnée (a, b) ; le gain
            vaut 1/b.
        -   `measure_pushbroom_noise` : le plancher s'impose par `a=` (au
            lieu de `b=`), et `b_fixed` devient `a_fixed` ; la covariance
            1 × 1 porte alors sur b.
        -   `snr_at_levels` : une covariance 1 × 1 porte sur b (a imposé).
        -   `add_noise(image, a, b)` : `a` est le plancher, `b` la pente.
        -   Exemple `noise_intercomparison.py` : `--a` plancher, `--b` pente.

0.3.1 (2026-11-08)

:   -   **Simulateur de bord intégré sur le pixel par défaut** :
        `SyntheticSlantedEdge(pixel_integration=True)` devient le défaut.
        L'échantillonnage ponctuel des versions 0.1 à 0.3 était un modèle
        irréaliste (bords fins repliés, MTF faussée près du Nyquist) ;
        `pixel_integration=False` le reproduit. **Changement de
        comportement** : les images générées et la vérité
        `mtf_theoretical` (qui inclut désormais `|sinc(f)|`) changent.

0.3.0 (2026-11-08)

:   -   **MTF du système par défaut** : l'estimateur `esf_lsf` (méthode
        par défaut de `measure_mtf`) ne retire plus la MTF du pixel. Il
        renvoie, comme les trois autres estimateurs, la MTF du système
        avant échantillonnage, pixel compris (ISO 12233). **Changement de
        comportement** : les valeurs de 0.2.0 étaient celles de l'optique
        seule, plus fortes d'un facteur `1/sinc(f)` (+11 % à 0,25
        cycle/pixel, +57 % au Nyquist).
    -   Option `remove_pixel_aperture=True` (CLI
        `--esf-lsf-remove-pixel-aperture`) pour la MTF de l'optique seule.
    -   `SyntheticSlantedEdge(pixel_integration=True, supersample=8)` :
        intégration sur le pixel, vérité incluant `|sinc(f)|`. Le défaut
        (échantillonnage ponctuel) est conservé.
    -   Tests de la MTF du système et de l'option ; analyse et justesse
        mesurée : `REFERENCE_TECHNIQUE.md`, partie I, §9.15.

0.2.0 (2026-10-27)

:   -   **Projet signalé en cours de développement** (README, docstring
        du paquet) : mesures validées sur simulation, pas encore
        qualifiées sur données réelles.
    -   **Mesures radiométriques** (`eoqual.measure.radiometry`) :
        `measure_dark_frame`, `measure_dark_current`,
        `measure_temporal_noise`, `measure_ptc` (courbe de transfert
        photonique), `measure_prnu`, `measure_linearity`,
        `map_defective_pixels`, `find_rts_pixels`, `adc_histogram`,
        `measure_pushbroom_noise`, `snr_at_levels`,
        `measure_spatial_noise` (pile, push broom, whisk broom),
        `measure_equalization_noise`, `decompose_spatial`.
    -   **Spectres de bruit** (`eoqual.measure.noise`) :
        `noise_power_spectrum`, `find_structured_noise`.
    -   **Contraste sur bruit** (`eoqual.measure.contrast`) :
        `measure_cnr`.
    -   **Simulateur de capteur** (`eoqual.measure.synthetic`) :
        `MatrixSensor`, `PushbroomSensor` (TDI), `WhiskbroomSensor`,
        `natural_scene`, `PeriodicPattern`, `apply_kernel` ; bruits de
        photons, de lecture et d'obscurité, DSNU/PRNU, courant
        d'obscurité et température, pixels chauds, morts, bloqués et
        télégraphiques, non-linéarité, transmission optique, bruits
        structurés, rayons cosmiques, convertisseur à pas inégaux.
    -   Registre `config.py` : 16 mesures ajoutées ou rendues
        disponibles ; la PTC s'appuie sur `temporal_noise` et
        `dark_frame`.
    -   `REFERENCE_TECHNIQUE.md` : parties IV (mesures radiométriques) et
        V (simulateur) ; lien du sommaire de la partie I corrigé.
    -   `examples/` : études de domaine de validité
        (`pushbroom_texture_bias_study.py`,
        `structured_noise_detection_study.py`) et plan de tests sur
        données réelles.
    -   169 tests ajoutés (validation contre la vérité du simulateur,
        robustes sur 30 graines via `EOQUAL_TEST_SEED`).
    -   Intégration continue des tests (`.github/workflows/tests.yml`).
    -   `license.txt` : titulaire des droits (Olivier Amram) au lieu du
        nom du module.

0.1.0 (2026-10-11)

:   -   Première version.
    -   **MTF par bord incliné** (`eoqual.measure.mtf`) : canonicalisation
        générique, 3 détecteurs d'orientation (gradient, PCA, phase de
        Fourier), 4 estimateurs de MTF (spatial, spectral, ESF/LSF avec
        modèles numérique/sigmoïde/PCHIP/Tikhonov, corrélation
        spectrale), comparateurs, bootstrap, aberration chromatique
        latérale sur plans Bayer (quatre motifs), études de sensibilité,
        CLI `eoqual-measure-mtf` (`--image-format bayer*`,
        `--bayer-pattern`). MTF ESF/LSF normalisée par la composante
        continue (référence technique,
        §9.14).
    -   **Variance du bruit** (`eoqual.measure.noise`) : 6 estimateurs
        (`flat_patches`, `wavelets`, `skimage_estimate_sigma`,
        `ponomarenko`, `adaptive_segmentation`, `pca_estimate`) ;
        `ponomarenko` vectorisé, DCT float64, classes de luminance à
        effectifs égaux.
    -   **Strehl et énergie encerclée** (`eoqual.measure.psf`) :
        `measure_psf` (Strehl, EE(r), r50, r80, EE au premier anneau
        noir) ; écart au Strehl exact ±0.2 % sur PSF synthétiques
        (référence technique, partie III).
    -   API de haut niveau : `measure_mtf()` → `MTFResult` (avec
        `frequency_at()` pour MTF50 & co), `estimate_noise()` →
        `NoiseEstimate`, `measure_psf()` → `PSFMeasurement`.
    -   Registre `MEASURES_CONFIGS`, avec les mesures prévues (courbe de
        transfert photonique, bruit d'égalisation, contraste sur bruit).
    -   `synthetic/` : mires à bord incliné à FTM analytique, injection
        de bruit à variance connue, PSF aberrée (Zernike) à Strehl exact.
    -   Briques numériques partagées via **eoqual-core** (`edge`, `bayer`,
        `detrend`, `tiling`, `psf`, `fourier`).
    -   Extras `plot` (plotly) et `study` (pandas).

## Pistes

-   Mesures prévues au registre.
