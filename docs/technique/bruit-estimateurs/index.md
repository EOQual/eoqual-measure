# Variance du bruit : estimateurs sur scène

Module `eoqual/measure/noise/variance.py` : estimation de la variance du bruit radiométrique
dans des images 2D (télédétection métrique/infra-métrique), sur des zones
supposées homogènes détectées automatiquement selon plusieurs stratégies.
L'estimation par le variogramme, qui diagnostique sa propre fiabilité, a son
document : [variogramme](../bruit-variogramme.md).

| Document | Contenu |
|---|---|
| [Principe, méthodes et biais](methodes.md) | Principe commun, les six méthodes et leurs paramètres, biais résiduel des méthodes par sélection de patchs |
| [Conduite de la mesure](mesure.md) | Barres d'erreur et images multiples, taille des zones, validation synthétique ou mesure réelle, non-linéarité aux extrêmes, courbe bruit-luminance |
| [Utilisation](utilisation.md) | API, script d'intercomparaison, mode Bayer par bande |
| [Recommandations, limites et pistes](recommandations.md) | Choix de méthode, limites connues, pistes d'amélioration |

```{toctree}
:hidden:
:maxdepth: 1

methodes
mesure
utilisation
recommandations
```
