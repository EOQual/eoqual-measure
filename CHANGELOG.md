# Change Log

## Version History

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
