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
voir ``CONTRIBUTING.md`` pour en ajouter une.
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
        "photon_transfer_curve": {
            "family": "radiometry",
            "status": "planned",
            "input": "Paires de champs uniformes à niveaux d'éclairement croissants (+ noirs)",
            "outputs": "Gain de conversion (e-/DN), bruit de lecture, capacité à saturation, linéarité",
            "entry": "eoqual.measure.radiometry.ptc",
            "builds_on": ["noise_variance"],
            "reference": "Janesick (2007), Photon Transfer, SPIE Press PM170 ; EMVA Standard 1288",
        },
        "equalization_noise": {
            "family": "radiometry",
            "status": "planned",
            "input": "Champ uniforme après égalisation radiométrique (correction des non-uniformités)",
            "outputs": "Bruit d'égalisation résiduel (composantes ligne/colonne/pixel), part relative au bruit temporel",
            "entry": "eoqual.measure.radiometry.equalization",
            "builds_on": ["noise_variance"],
            "reference": "EMVA Standard 1288 (non-uniformités spatiales DSNU/PRNU)",
        },
        "cnr_target": {
            "family": "contrast",
            "status": "planned",
            "input": "Mire à zones de réflectance/contraste connus, zones fournies ou détectées",
            "outputs": "Rapport contraste sur bruit par paire de zones",
            "entry": "eoqual.measure.contrast.cnr",
            "builds_on": ["noise_variance"],
            "reference": "À documenter avec l'implémentation",
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
