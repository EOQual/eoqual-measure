# Variance du bruit et SNR par le variogramme

Mesurer la variance du bruit σ², et donc le SNR, **sur une scène
quelconque**, sans mire ni champ uniforme, à partir d'une seule image :
c'est l'estimation dite « par variogramme » (Curran et Dungan, 1989), issue
de la géostatistique (Matheron, 1963). La méthode rend aussi un modèle de
bruit σ² = a + b·μ lorsque la scène couvre plusieurs niveaux, et deux
diagnostics qui disent quand la mesure est exploitable.

Module : `eoqual.measure.noise.variogram` (fonctions `variogram_noise` et
`variogram`, méthode `"variogram"` de `estimate_noise`). Les chapitres
gardent une numérotation unique d'un document à l'autre : un renvoi « §n »
désigne toujours le même chapitre.

| Document | Chapitres | Contenu |
|---|---|---|
| [Objet et principe](principe.md) | 1, 2 | Variogramme, pépite, palier, portée ; extrapolation à l'origine ; lien avec le NPS ; fraction de pépite F |
| [Mise en œuvre](mise-en-oeuvre.md) | 3 | Chaîne de calcul, retrait du plan, incertitude, anisotropie, modèle de bruit et SNR, exemple, paramètres |
| [Validation](validation.md) | 4 | Scènes fractales, scène urbaine, bruit corrélé, bruit blanc seul, pièges, tests |
| [Comparaison et cas réel](comparaison.md) | 5, 6 | Comparaison avec Ponomarenko et les autres estimateurs, quelle méthode choisir, orthophotographie |
| [Limites, formulaire, pistes et références](limites.md) | 7, 8, 9 | Limites, formulaire, pistes, références |

```{toctree}
:hidden:
:maxdepth: 1

principe
mise-en-oeuvre
validation
comparaison
limites
```
