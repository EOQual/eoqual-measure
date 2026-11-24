# Mesures radiométriques

Modules `eoqual.measure.radiometry`, `eoqual.measure.noise.spectrum` et
`eoqual.measure.contrast` : obscurité et courant d'obscurité, bruit temporel
et courbe de transfert photonique, PRNU, linéarité, pixels défectueux et
télégraphiques, convertisseur, bruit temporel d'un push broom et SNR, bruit
spatial et d'égalisation, spectre du bruit et bruits structurés, contraste
sur bruit. **En cours de développement** : mesures validées sur simulation,
pas encore qualifiées sur données réelles.

Dans ces pages, les renvois « §n » et « chapitre n » désignent les chapitres
des mesures radiométriques ; le simulateur qui les valide a son propre
document, [Simulateur de capteur](../simulateur/index.md).

## 1. Introduction

### 1.1 Objet

Cette partie réunit des **métriques de bruit et de défauts radiométriques**
d'imageurs (capteurs matriciels, barrettes push broom, whisk broom) :
offset et bruit à signal nul, bruit temporel, gain et bruit de lecture,
courant d'obscurité, non-uniformités de réponse, linéarité, pixels
défectueux, bruits structurés, convertisseur, contraste sur bruit. Elles
portent sur des **acquisitions contrôlées** (obscurité, champs uniformes,
piles temporelles, mires, scènes uniformes en vol), suivant la séparation
*measure* / *metrics* d'EOQual. L'étalonnage absolu (conversion en
luminance, réponse spectrale) est hors périmètre.

Chaque métrique est une fonction qui prend des tableaux NumPy et renvoie
un résultat typé (dataclass) portant ses **incertitudes-types**. Toutes ont
été validées contre un simulateur de capteur à vérité connue, décrit dans
le [simulateur de capteur](../simulateur/index.md). Elles sont **en cours de développement** : validées sur
données simulées, elles ne sont pas encore qualifiées sur données réelles
(plan de tests : `examples/PLAN_TESTS_DONNEES_REELLES.md`).

### 1.2 Familles de métriques

- **Obscurité** : `dark_frame` (offset, DSNU, bruit à signal nul,
  stationnarité), `dark_current` (courant, sa dispersion, pixels chauds,
  loi en température).
- **Bruit temporel et transfert photonique** : `temporal_noise`,
  `photon_transfer_curve` (gain K, bruit de lecture, saturation,
  linéarité), `snr_model`.
- **Non-uniformités** : `prnu` (non-uniformité de réponse, vignettage),
  `spatial_noise` (bruit spatial fixe, colonnage, bandes),
  `equalization_noise` (résidu après égalisation).
- **Linéarité** : `linearity` (erreur EMVA, courbure commune et par
  pixel).
- **Défauts de pixels** : `defective_pixels` (bloqués, morts, chauds,
  bruyants, réponse aberrante), `rts_pixels` (pixels télégraphiques).
- **Bruits corrélés et structurés** : `noise_power_spectrum`,
  `structured_noise` (motifs périodiques, bruit de ligne et de colonne).
- **Convertisseur** : `adc_histogram` (DNL, INL, codes manquants).
- **En vol, sur scène** : `pushbroom_temporal_noise` (modèle de bruit
  d'un push broom sur scènes peu texturées), `variogram_noise` (variance
  du bruit, SNR et modèle a + b·μ d'une image brute, par la pépite du
  variogramme).
- **Sur mire** : `cnr_target` (contraste sur bruit).

### 1.3 Tableau de synthèse

✔ : prévu pour ce type de capteur ; ✱ : utilisable dans un cas
particulier (indiqué) ; — : sans objet. *Sol* : banc de caractérisation ;
*vol* : images acquises en orbite.

| Métrique | Matriciel | Push broom | Whisk broom | Données d'entrée | Contexte | Chapitre |
|---|---|---|---|---|---|---|
| `dark_frame` | ✔ | ✔ bloc de lignes | — | pile ou bloc d'obscurité | sol, vol (lignes d'obscurité) | 3 |
| `temporal_noise` | ✔ | — | — | paire ou pile de champs uniformes | sol | 3 |
| `photon_transfer_curve` | ✔ | — | — | paires à plusieurs niveaux + obscurité | sol | 3 |
| `dark_current` | ✔ | ✱ bloc remis en (M, 1, W) | — | obscurité à plusieurs temps de pose (et températures) | sol | 4 |
| `prnu` | ✔ | ✱ via `spatial_noise` | — | champs uniformes + obscurité | sol | 5 |
| `linearity` | ✔ | — | — | expositions connues + obscurité | sol | 6 |
| `defective_pixels` | ✔ | — | — | obscurité, champs uniformes, résultat de `dark_current` | sol | 7 |
| `rts_pixels` | ✔ | ✱ bloc remis en (M, 1, W) | — | longue pile temporelle | sol | 8 |
| `adc_histogram` | ✔ | ✔ | ✔ | fond lisse ou rampe | sol | 9 |
| `pushbroom_temporal_noise` | — | ✔ | — | scènes peu texturées, plusieurs niveaux | vol | 10 |
| `snr_model` | ✔ (PTC) | ✔ (push broom) | ✔ | modèle σ² = a + b·μ | sol, vol | 10 |
| `spatial_noise` | ✔ pile | ✔ zone uniforme | ✔ bandes, aller/retour | champ ou zone uniforme | sol, vol | 11 |
| `equalization_noise` | — | ✔ | — | zone uniforme égalisée | sol, vol | 11 |
| `noise_power_spectrum` | ✔ | ✔ image | ✔ image | zones uniformes, produits traités | sol, vol | 12 |
| `structured_noise` | ✔ pile | ✱ image seule | ✱ image seule | obscurité, champs uniformes | sol | 12 |
| `cnr_target` | ✔ | ✔ | ✔ | mire à deux plages | sol, vol | 13 |
| `variogram_noise` | ✔ | ✔ | ✔ | une image brute, scène lisse à l'échelle du pixel | sol, vol | [variogramme](../bruit-variogramme.md) |

Sorties principales : voir l'objet de chaque chapitre. Un même résultat
sert souvent d'entrée à une autre métrique : `DarkFrameResult` (obscurité
de `temporal_noise`, `ptc`, `prnu`, `linearity`, `pushbroom_temporal_noise`),
`DarkCurrentResult` (pixels chauds de `defective_pixels`), `DefectMap.any`
(paramètre `exclude` de `prnu`, `linearity`, `rts_pixels`), tout résultat
portant `a`, `b`, `cov` (`snr_model`), dont `VariogramNoiseResult`.

### 1.4 Parcours types

**Caractérisation au sol d'un capteur matriciel** (démarche EMVA 1288) :

1. `dark_frame` sur une pile d'obscurité : offset, b, DSNU ; vérifier la
   stationnarité ;
2. `dark_current` sur des obscurités à plusieurs temps de pose : courant,
   pixels chauds, gain poissonnien ;
3. `photon_transfer_curve` sur des paires de champs uniformes : gain K,
   bruit de lecture, saturation ;
4. `defective_pixels`, puis `prnu` et `linearity` en excluant les pixels
   défectueux ;
5. selon les besoins : `structured_noise` (motifs, bruit de ligne),
   `rts_pixels`, `adc_histogram`.

**Recette d'une barrette push broom au sol** : `dark_frame` en
`layout="pushbroom"`, `spatial_noise` sur une zone uniforme (colonnage),
égalisation par la chaîne de traitement, puis `equalization_noise` (FPN
résiduel BF et HF).

**Suivi en vol d'un push broom** : `pushbroom_temporal_noise` sur des
scènes peu texturées (désert, neige, océan), avec b tiré de lignes
d'obscurité, puis `snr_model` aux niveaux spécifiés ; `spatial_noise` et
`equalization_noise` sur scènes uniformes ; vérifier l'indicateur
`texture_suspected`.

**Whisk broom** : `spatial_noise` en `layout="whiskbroom"` (bandes par
détecteur, différence aller/retour).

**Produits traités** : `noise_power_spectrum` (forme du bruit après
dématriçage, rééchantillonnage ou compression), `cnr_target` sur mire.

**Image brute d'une scène quelconque** : `variogram_noise` (σ², SNR,
modèle a + b·μ), en vérifiant `reliable`, la fraction de pépite F et
l'anisotropie ; le croiser avec `estimate_noise(..., "ponomarenko")`
([variogramme, §5.5](../bruit-variogramme.md#55-quelle-méthode-choisir-)).

### 1.5 Conventions

- **Unités** : les métriques travaillent en DN ; un résultat en e⁻
  suppose le gain K (e⁻/DN) connu ou mesuré.
- **Modèle de bruit** : σ²(μ) = a + b·μ, μ signal corrigé de l'obscurité,
  a variance à signal nul ou plancher (bruit de lecture et quantification,
  §2.2), b = 1/K la pente. Un bruit stationnaire de variance σ² s'écrit
  a = σ², b = 0. Attention : la littérature (Foi et al., 2008, modèle
  poissonnien-gaussien) note souvent σ² = a·y + b, lettres inversées.
- **Incertitudes** : chaque grandeur `x` est accompagnée de `u_x`,
  incertitude-type (1 σ) due au bruit des données. Les dispersions
  spatiales (DSNU, PRNU…) sont celles de la carte *réalisée* du capteur
  mesuré, corrigées du bruit temporel résiduel.
- **Validation** : chaque métrique est comparée à la vérité du simulateur
  avec des tolérances de 5 u, sur plusieurs cas, et pour 30 graines ; le
  détail est dans la section « Tests » de chaque chapitre.
- **Entrées** : piles (K, H, W), blocs push broom (M, W) (lignes le long
  de la trace, détecteurs en travers), images whisk broom
  (balayages × k détecteurs, W).

### 1.6 Organisation des pages

Le chapitre 2 expose les fondements communs ; les chapitres 3 à 13
présentent une métrique (ou une famille) chacun, selon le même plan :
objet, principe, mise en œuvre (fonctions, exemple, recommandations),
validation (résultats types, domaine de validité, tests), limites, puis,
s'il y a lieu, les variantes de méthode écartées et les pistes. Le
[formulaire](formulaire.md) et la [bibliographie](bibliographie.md) sont
communs. La variance du bruit et le SNR d'une image brute par le
variogramme (`variogram_noise`), qui relèvent aussi du suivi en vol, sont
décrits avec les autres estimateurs sur scène :
[variogramme](../bruit-variogramme.md). Les chemins de fichiers sont
relatifs à `src/eoqual/measure/`.

```{toctree}
:hidden:
:maxdepth: 1

02-fondements
03-obscurite-ptc
04-courant-obscurite
05-prnu
06-linearite
07-pixels-defectueux
08-pixels-telegraphiques
09-convertisseur
10-pushbroom-snr
11-bruit-spatial-egalisation
12-nps-bruits-structures
13-contraste
formulaire
bibliographie
```
