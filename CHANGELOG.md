# Change Log

## Version History

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
