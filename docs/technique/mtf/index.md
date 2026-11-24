# MTF par bord incliné

Module `eoqual.measure.mtf.slanted_edge`.

Estimation de l'orientation et de la MTF (Modulation Transfer Function)
d'un système d'imagerie à partir d'une mire à bord incliné, avec
plusieurs algorithmes indépendants comparables entre eux à chaque
étape.

**Statut** : opérationnel, validé sur cas synthétiques (bords propres,
bruités, angles variés, PSF gaussienne connue). À valider par
simulation plus poussée avant un usage métrologique de production.

---

## Contexte et objectifs

Ce pipeline est une brique logicielle d'estimation de la FTM par la
méthode du bord incliné, **fortement modulaire**, permettant de remplacer aisément une
implémentation élémentaire par une autre (détection de bord,
construction d'ESF, calcul de MTF) ou de laisser le choix à
l'utilisateur — c'est exactement le principe directeur de
l'architecture décrit en [§2.1](architecture.md) (template method + comparateur), qui
permet à `EdgeOrientationComparator`/`MTFComparator` d'exécuter et de
comparer plusieurs implémentations interchangeables sur la même donnée.

Les études de sensibilité (angle, bruit, comparaison des algorithmes)
disposent d'un harnais ([§8.12](guide.md)). L'exploitation de bords naturels sur
cibles urbaines, plutôt que de mires dédiées, reste une piste ([§12](pistes.md)).

Les chapitres gardent une numérotation unique d'un document à l'autre : un
renvoi « §n » désigne toujours le même chapitre.

| Document | Chapitres | Contenu |
|---|---|---|
| [Fondements théoriques](theorie.md) | 1 | Bord incliné, périodicité de la FFT, correction d'angle, théorème du décalage, ouverture du pixel, biais de dérivation |
| [Architecture, Bayer, visualisation, orchestration](architecture.md) | 2, 5, 6, 7 | Template method et comparateurs, conventions de repère et de fréquence, support Bayer, visualisation, bootstrap, aberration chromatique, CLI |
| [Orientation et estimateurs](estimateurs.md) | 3, 4 | Canonicalisation, trois détecteurs d'orientation, quatre estimateurs de MTF et leurs paramètres |
| [Guide d'utilisation](guide.md) | 8 | Installation, ligne de commande, usage programmatique, interprétation, études de sensibilité |
| [Choix de conception et pièges évités](choix.md) | 9 | Pièges de méthode, MTF du système ou de l'optique, justesse mesurée |
| [Bibliographie et formulaire](references.md) | 10, 11 | Références des méthodes, formulaire |
| [Pistes](pistes.md) | 12 | Bords naturels, rognage adaptatif de l'ESF, MTF 2D |

```{toctree}
:hidden:
:maxdepth: 1

theorie
architecture
estimateurs
guide
choix
references
pistes
```
