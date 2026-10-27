"""
Registre central des mesures de ``eoqual.measure``.

Une mesure, au sens de ce paquet, est une **mesure physique contrôlée** :
la cible (mire, bord choisi, champ uniforme, source ponctuelle) ou le
protocole est connu, et le résultat est une grandeur physique du système
d'imagerie (FTM, variance de bruit, gain...), pas un score de qualité.
Les indicateurs sans référence calculés sur scène quelconque relèvent
d'``eoqual-metrics``.

Chaque entrée décrit : la famille, le statut (``"available"`` ou
``"planned"``), l'entrée attendue, les grandeurs produites, le point
d'entrée Python (``module:fonction``), les mesures existantes sur
lesquelles elle s'appuie, et les références. Les entrées ``"planned"``
fixent l'emplacement et le contrat visés pour les mesures à venir ;
voir ``CONTRIBUTING.md`` pour en ajouter une. Les mesures radiométriques
(famille ``radiometry``, spectres de ``noise``, ``contrast``) sont décrites
dans ``REFERENCE_TECHNIQUE.md``, partie IV.
"""

from collections import OrderedDict

MEASURES_CONFIGS: OrderedDict = OrderedDict(
    {
        "mtf_slanted_edge": {
            "family": "mtf",
            "status": "available",
            "input": "Zone 2D contenant un unique bord incliné (mire ou bord choisi)",
            "outputs": "MTF(f) brute et corrigée de l'angle, MTF@Nyquist, MTF50, FWHM, RER",
            "entry": "eoqual.measure.mtf:measure_mtf",
            "builds_on": [],
            "reference": "ISO 12233:2017 ; Burns (2000), Proc. IS&T PICS, 135-138",
        },
        "noise_variance": {
            "family": "noise",
            "status": "available",
            "input": "Image 2D ; zones homogènes détectées automatiquement",
            "outputs": "Variance du bruit (globale ou par classe de luminance), échantillons intra-image",
            "entry": "eoqual.measure.noise:estimate_noise",
            "builds_on": [],
            "reference": "Ponomarenko et al. (2007), IGARSS ; Colom & Buades (2013), IPOL 3, 173-197",
        },
        "dark_frame": {
            "family": "radiometry",
            "status": "available",
            "input": "Pile d'images d'obscurité (K, H, W), ou bloc de lignes d'obscurité d'un push broom (M, W)",
            "outputs": "Carte d'offset, bruit à signal nul b, DSNU et décomposition ligne/colonne/pixel, stationnarité (dérive)",
            "entry": "eoqual.measure.radiometry:measure_dark_frame",
            "builds_on": [],
            "reference": "EMVA Standard 1288 (DSNU, bruit à signal nul)",
        },
        "dark_current": {
            "family": "radiometry",
            "status": "available",
            "input": "Piles d'obscurité à plusieurs temps de pose (et températures)",
            "outputs": "Courant d'obscurité par pixel, moyenne et DSNU du courant, pixels chauds, gain poissonnien, loi en température (E_g, doublement)",
            "entry": "eoqual.measure.radiometry:measure_dark_current",
            "builds_on": ["dark_frame"],
            "reference": "EMVA Standard 1288 (courant d'obscurité) ; Janesick (2001), Scientific CCDs, SPIE PM83",
        },
        "temporal_noise": {
            "family": "radiometry",
            "status": "available",
            "input": "Paire ou pile de champs uniformes de même niveau (+ obscurité)",
            "outputs": "Variance temporelle, signal corrigé de l'obscurité",
            "entry": "eoqual.measure.radiometry:measure_temporal_noise",
            "builds_on": ["dark_frame"],
            "reference": "EMVA Standard 1288",
        },
        "photon_transfer_curve": {
            "family": "radiometry",
            "status": "available",
            "input": "Paires de champs uniformes à niveaux d'éclairement croissants (+ obscurité)",
            "outputs": "Gain de conversion (e-/DN), bruit de lecture, capacité à saturation, linéarité, chi2 réduit",
            "entry": "eoqual.measure.radiometry:measure_ptc",
            "builds_on": ["temporal_noise", "dark_frame"],
            "reference": "Janesick (2007), Photon Transfer, SPIE Press PM170 ; EMVA Standard 1288",
        },
        "pushbroom_temporal_noise": {
            "family": "radiometry",
            "status": "available",
            "input": "Images push broom (M, W) de scènes peu texturées à plusieurs niveaux (+ obscurité)",
            "outputs": "Modèle de bruit temporel sigma²(mu) = a·mu + b, diagnostic de texture résiduelle",
            "entry": "eoqual.measure.radiometry:measure_pushbroom_noise",
            "builds_on": ["dark_frame"],
            "reference": "Immerkær (1996), CVIU 64(2), 300-302",
        },
        "snr_model": {
            "family": "radiometry",
            "status": "available",
            "input": "Résultat portant un modèle de bruit (a, b, covariance) et des niveaux",
            "outputs": "SNR et son incertitude aux niveaux demandés",
            "entry": "eoqual.measure.radiometry:snr_at_levels",
            "builds_on": ["photon_transfer_curve", "pushbroom_temporal_noise"],
            "reference": "EMVA Standard 1288",
        },
        "spatial_noise": {
            "family": "radiometry",
            "status": "available",
            "input": "Pile de champs uniformes, zone uniforme push broom (M, W) ou whisk broom",
            "outputs": "Bruit spatial fixe, décomposition, bandes périodiques, différence aller/retour (whisk broom)",
            "entry": "eoqual.measure.radiometry:measure_spatial_noise",
            "builds_on": ["dark_frame"],
            "reference": "EMVA Standard 1288 (non-uniformités spatiales)",
        },
        "equalization_noise": {
            "family": "radiometry",
            "status": "available",
            "input": "Zone uniforme d'une image push broom égalisée (M, W)",
            "outputs": "FPN résiduel, composantes BF et HF corrigées du bruit temporel, spectre de la ligne moyenne",
            "entry": "eoqual.measure.radiometry:measure_equalization_noise",
            "builds_on": ["spatial_noise"],
            "reference": "EMVA Standard 1288 (non-uniformités spatiales DSNU/PRNU)",
        },
        "prnu": {
            "family": "radiometry",
            "status": "available",
            "input": "Piles de champs uniformes à un ou plusieurs niveaux + obscurité",
            "outputs": "Carte de réponse relative, PRNU totale et EMVA, séparation BF (vignettage) / HF, décomposition",
            "entry": "eoqual.measure.radiometry:measure_prnu",
            "builds_on": ["dark_frame"],
            "reference": "EMVA Standard 1288 (PRNU)",
        },
        "linearity": {
            "family": "radiometry",
            "status": "available",
            "input": "Piles à plusieurs expositions connues (temps de pose, source stable) + obscurité",
            "outputs": "Erreur de linéarité EMVA, courbure commune, dispersion de la courbure par pixel",
            "entry": "eoqual.measure.radiometry:measure_linearity",
            "builds_on": ["dark_frame"],
            "reference": "EMVA Standard 1288 (linearity error)",
        },
        "defective_pixels": {
            "family": "radiometry",
            "status": "available",
            "input": "Obscurité, champs uniformes, résultat de dark_current",
            "outputs": "Carte des pixels bloqués, morts, chauds, bruyants et à réponse aberrante",
            "entry": "eoqual.measure.radiometry:map_defective_pixels",
            "builds_on": ["dark_current"],
            "reference": "EMVA Standard 1288",
        },
        "rts_pixels": {
            "family": "radiometry",
            "status": "available",
            "input": "Longue pile temporelle (obscurité ou champ uniforme stable)",
            "outputs": "Pixels télégraphiques : niveaux, amplitude du saut, probabilités de bascule",
            "entry": "eoqual.measure.radiometry:find_rts_pixels",
            "builds_on": [],
            "reference": "Critère BIC (mélange à deux niveaux) ; décodage de Viterbi d'une chaîne de Markov à deux états",
        },
        "adc_histogram": {
            "family": "radiometry",
            "status": "available",
            "input": "Images à fond lisse ou rampe couvrant la plage de codes",
            "outputs": "DNL et INL par code, codes manquants",
            "entry": "eoqual.measure.radiometry:adc_histogram",
            "builds_on": [],
            "reference": "IEEE Std 1241 (essais des convertisseurs analogique-numérique)",
        },
        "noise_power_spectrum": {
            "family": "noise",
            "status": "available",
            "input": "Image ou pile de zones uniformes (brutes ou traitées)",
            "outputs": "NPS 2D, coupes, moyenne radiale, variance intégrée",
            "entry": "eoqual.measure.noise:noise_power_spectrum",
            "builds_on": [],
            "reference": "Welch (1967), IEEE Trans. Audio Electroacoust. 15(2) ; ISO 15739",
        },
        "structured_noise": {
            "family": "noise",
            "status": "available",
            "input": "Pile (ou image) d'obscurité ou de champs uniformes",
            "outputs": "Motifs périodiques (fréquence, orientation, amplitude, nature fixe/temporelle), bruit de ligne et de colonne",
            "entry": "eoqual.measure.noise:find_structured_noise",
            "builds_on": ["noise_power_spectrum"],
            "reference": "Welch (1967) ; EMVA Standard 1288 (spectrogrammes)",
        },
        "cnr_target": {
            "family": "contrast",
            "status": "available",
            "input": "Mire à deux plages, zones fournies ou détectées",
            "outputs": "Rapport contraste sur bruit et son incertitude",
            "entry": "eoqual.measure.contrast:measure_cnr",
            "builds_on": [],
            "reference": "Définition CNR = (mu_clair - mu_sombre) / sqrt(sigma²_clair + sigma²_sombre)",
        },
        "strehl_encircled_energy": {
            "family": "psf",
            "status": "available",
            "input": "Imagette d'une source ponctuelle isolée (étoile, collimateur), échantillonnée à λ/D >= 2 px",
            "outputs": "Rapport de Strehl, énergie encerclée EE(r), r50, r80, EE au premier anneau noir",
            "entry": "eoqual.measure.psf:measure_psf",
            "builds_on": [],
            "reference": "Mahajan (1982), JOSA 72(9), 1258-1266 ; Born & Wolf, Principles of Optics",
        },
    }
)
