# Simulateur de capteur

Simulateur à vérité connue (`eoqual.measure.synthetic`), qui produit les
données de validation des [mesures radiométriques](../radiometrie/index.md). Le
modèle physique du pixel et les sources de bruit (modèle EMVA 1288, bruit de
photons, de lecture, d'obscurité, quantification, écrêtages,
non-uniformités) sont exposés dans leurs
[fondements communs](../radiometrie/02-fondements.md) ; ces documents décrivent
leur simulation et les effets propres au simulateur.

Production d'images de capteur à **vérité connue** : chaque grandeur que
les mesures doivent retrouver (offset, DSNU, PRNU, gain, bruit de lecture,
a, b…) est connue exactement, carte par carte.

Les effets sont regroupés en cinq familles, qui servent de repères dans
le code et dans ce document :

| Famille | Effets |
|---|---|
| S0 | capteur matriciel linéaire : photons, PRNU, puits, lecture, gain, offset et DSNU, convertisseur |
| S1 | courant d'obscurité et sa DSNU, loi en température, pixels chauds, lueur d'amplificateur, overscan, dérive d'offset |
| S2 | barrette push broom avec TDI ; scènes naturelles (fBm, gradient lent) et PSF optique |
| S3 | bruit de ligne en 1/f, motifs périodiques, rayons cosmiques, traitements à fonction de transfert connue |
| S4 | transmission optique (vignettage, poussières), non-linéarité, pixels morts et bloqués, pixels télégraphiques, convertisseur à pas inégaux, whisk broom |

**Statut** : chaque famille est vérifiée contre sa propre vérité par les
sept fichiers `tests/test_synthetic_*.py` (62 tests, robustes sur 30
graines). La QE(λ) n'est pas simulée ([§7](validation.md)).

Les chapitres gardent une numérotation unique d'un document à l'autre : un
renvoi « §n » désigne toujours le même chapitre.

| Document | Chapitres | Contenu |
|---|---|---|
| [Objet et modèles](modeles.md) | 1, 2 | Exigences ; chaîne S0, bruits colorés, push broom et TDI, scènes et optique, bruits structurés, obscurité, transmission, RTS, convertisseur, whisk broom |
| [Architecture et implémentation](implementation.md) | 3, 4 | Organisation du code, construction et acquisition, graines, bruit coloré, push broom, scènes, effets S1 à S4 |
| [Guide d'utilisation](guide.md) | 5 | Exemples et valeurs par défaut |
| [Validation, limites et formulaire](validation.md) | 6, 7, 8 | Tests contre la vérité, limites et choix de modélisation, formulaire |
| [Bibliographie](bibliographie.md) | 9 | Références |

```{toctree}
:hidden:
:maxdepth: 1

modeles
implementation
guide
validation
bibliographie
```
