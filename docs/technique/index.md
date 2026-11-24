# Documents techniques

Un document par mesure ou famille de mesures : principe, mise en œuvre,
validation sur simulation, choix de conception et limites. Ces documents
décrivent l'état présent du code ; ce qui change d'une version à l'autre est
dans les [notes de version](../changelog.md).

| Document | Contenu | Module |
|---|---|---|
| [MTF par bord incliné](mtf-bord-incline.md) | Orientation du bord, quatre estimateurs de MTF, support Bayer, CLI | `eoqual.measure.mtf` |
| [Variance du bruit : estimateurs sur scène](bruit-estimateurs.md) | Six estimateurs par zones homogènes, ondelettes ou DCT, courbe bruit-luminance | `eoqual.measure.noise` |
| [Variance du bruit et SNR par le variogramme](bruit-variogramme.md) | Pépite du variogramme, fiabilité, rayures, modèle a + b·μ | `eoqual.measure.noise` |
| [Strehl et énergie encerclée](psf-strehl.md) | Rapport de Strehl et EE(r) sur une source ponctuelle | `eoqual.measure.psf` |
| [Mesures radiométriques](radiometrie/index.md) | Obscurité, PTC, PRNU, linéarité, défauts, convertisseur, push broom, bruit spatial, NPS, contraste | `eoqual.measure.radiometry`, `noise.spectrum`, `contrast` |
| [Simulateur de capteur](simulateur.md) | Capteurs matriciel, push broom et whisk broom à vérité connue | `eoqual.measure.synthetic` |

Les figures sont dans `figures/` ; celles qui accompagnent une étude sont
produites par un script de `examples/`, cité dans la légende.

```{toctree}
:hidden:
:maxdepth: 2

mtf-bord-incline
bruit-estimateurs
bruit-variogramme
psf-strehl
radiometrie/index
simulateur
```
