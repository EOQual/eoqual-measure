# Simulateur : architecture et implémentation

## 3. Architecture logicielle

```
eoqual/measure/synthetic/
├── __init__.py      API publique
├── patterns.py      générateurs : pattern_rng, structured_map, power_law_noise
├── scene.py         Scene, natural_scene (S2)
├── temporal_effects.py  PeriodicPattern, row_offsets, periodic_field, cosmic_rays (S3) ;
│                    RTSMap, rts_map, rts_states (S4)
├── processing.py    apply_kernel, kernel_power_transfer, gaussian_kernel (S3)
├── dark.py          dark_current_map, temperature_factor, doubling_temperature (S1)
├── response.py      transmission_map, defect_maps, adc_transfer (S4)
└── sensor.py        SensorParams, SensorTruth, MatrixSensor, Acquisition,
                     PushbroomSensor, PushbroomAcquisition (S2),
                     WhiskbroomSensor, WhiskbroomAcquisition (S4)
```

| Objet | Rôle |
|---|---|
| `SensorParams` | paramètres d'un capteur, figés (`frozen`), validés à la construction |
| `SensorTruth` | vérité : cartes, décompositions, a, b, seuils ; méthodes `expected_mean`, `expected_variance` |
| `MatrixSensor` | tire les motifs fixes à la construction, produit des piles avec `expose` / `dark` |
| `Acquisition` | résultat d'une pose : pile `uint16` (K, H, W), photons, graine, vérité |
| `StructuredMap` | carte décomposée `total`, `pixel`, `row`, `col` |
| `PushbroomSensor` | barrette (1, W), `tdi_stages` ; `expose(scene ou niveau, n_lines=, seed=)` |
| `PushbroomAcquisition` | image `uint16` (M, W), photons, graine, vérité |
| `Scene` / `natural_scene` | scène en photons et ses composantes (éclairement, texture) |

Principes :

- **la vérité est construite en même temps que le capteur**. Une mesure
  reçoit `acq.frames` et le test compare son résultat à `acq.truth` ;
- **résultats typés** : dataclasses `slots=True`, sur le modèle de
  `NoiseEstimate` d'eoqual-measure ;
- **un effet = un motif tiré dans son propre flux aléatoire**, nommé par
  une étiquette (§4.3).

## 4. Implémentation

### 4.1 Construction d'un capteur (`MatrixSensor.__init__`)

1. DSNU : `structured_map(shape, dsnu_pixel, dsnu_row, dsnu_col,
   pattern_rng(seed, "dsnu"))`, en DN.
2. PRNU : de même, avec l'étiquette `"prnu"`, sans unité. Le capteur est
   refusé si 1 + PRNU ≤ 0 sur un pixel.
3. Vérité : `offset_map = offset + DSNU`, `gain_map = 1 + PRNU`,
   a = (σ_lect/K)² + 1/12, b = 1/K, `saturation_dn` = 2ⁿ − 1.

`structured_map` tire les trois composantes gaussiennes et **centre**
chacune d'elles (moyenne spatiale nulle). L'offset moyen et le gain moyen
sont donc exactement ceux demandés, et les composantes sont celles que
les mesures doivent retrouver.

### 4.2 Acquisition (`MatrixSensor.expose`)

Pour une pile de forme (K, H, W) :

```text
mu_e = qe * photons * gain_map                 # e⁻, carte (H, W)
e = rng.poisson(mu_e, size=(K, H, W))          # bruit de photons
e = minimum(e, full_well)                      # puits
e = e + rng.normal(0, read_noise, size)        # bruit de lecture
y = e / K + offset_map                         # DN analogique
dn = clip(floor(y + 0.5), 0, 2**n - 1)         # convertisseur, uint16
```

`photons` est un scalaire (champ uniforme) ou une carte (H, W) (scène).
Il est positif, en photons moyens par pixel pendant la pose.
`dark(n_frames, seed)` équivaut à `expose(0, ...)`.

Mémoire : la pile est calculée en `float64` avant conversion, soit
8·K·H·W octets (32 Mo pour 64 poses de 256 × 256).

### 4.3 Graines et flux aléatoires

| Flux | Entropie du générateur | Rôle |
|---|---|---|
| motif `tag` | `[seed, crc32(tag)]` | DSNU (`"dsnu"`), PRNU (`"prnu"`), plus tard pixels chauds, défauts… |
| bruit temporel | `[seed, crc32("temporal"), seed_pose]` | Poisson et bruit de lecture d'une pile |

Conséquences :

- même `SensorParams` → même capteur, au bit près ;
- changer un paramètre de PRNU ne change pas la DSNU (test
  `test_fixed_pattern_does_not_depend_on_other_patterns`) ;
- ajouter un effet en S1 ne modifiera pas les cartes S0 d'un capteur
  existant ;
- deux poses de même `seed_pose` sont identiques, deux graines
  différentes donnent deux réalisations indépendantes du même capteur.

La graine de pose est obligatoire (argument nommé sans valeur par défaut) :
aucune simulation ne dépend d'un état global.

### 4.4 Bruit coloré (`power_law_noise`)

```text
white = rng.standard_normal(shape)
F = fftn(white) * |f|^(−β/2)      # f radiale, f[0] remplacée par 1
F[0] = 0                          # moyenne nulle
x = real(ifftn(F)); x /= x.std()  # écart-type 1
```

Le champ obtenu est **périodique** (synthèse par FFT) : ses bords opposés
se raccordent. Pour une scène non périodique, tirer un champ plus grand et
en extraire une partie, ce que fait `natural_scene`.

### 4.5 Barrette push broom (`PushbroomSensor`)

Les deux capteurs partagent la même chaîne temporelle (`_digitize` :
Poisson → puits → lecture → ÷K + offset → convertisseur) et la même
construction de la vérité (`_build_truth`).

- `params.shape` vaut (1, W). La composante ligne des motifs est nulle par
  construction, et la dispersion entre détecteurs se règle par les
  composantes `*_col` ou `*_pixel`.
- `expose(photons, n_lines=M, seed=)` : `photons` est un scalaire (scène
  uniforme, `n_lines` obligatoire) ou une scène (M, W). Le signal moyen
  N·η·Φ·g est diffusé sur les lignes, et le bruit tiré pour chaque pixel
  de l'image.
- `truth.detector_offset` et `truth.detector_gain` (W,) donnent l'offset
  et le gain de chaque détecteur ; `truth.tdi_stages` vaut N.

### 4.6 Scènes (`natural_scene`)

1. Texture : `power_law_noise((2M, 2W), β)` ; si σ_psf > 0, produit par
   la FTM gaussienne dans le domaine de Fourier ; recadrage à (M, W) ;
   recentrage et normalisation à un écart-type de 1, si bien que `contrast`
   est le contraste *après* optique.
2. Éclairement : 1 + g_∥·u + g_⊥·v, avec u, v les coordonnées réduites des
   centres de pixels dans [−½, ½].
3. Φ = Φ₀ · éclairement · (1 + c·T). La fonction refuse toute scène
   négative ou nulle.

### 4.7 Bruits structurés et traitements (S3)

Paramètres de `SensorParams`, tous nuls par défaut, de sorte que les
capteurs S0 et S2 sont inchangés au bit près :

- `row_noise` (DN), `row_noise_beta` ;
- `periodic` : tuple de `PeriodicPattern(amplitude, fx, fy, temporal,
  phase)` ;
- `cosmic_rate` (traces par pose), `cosmic_charge` (e⁻ par pixel),
  `cosmic_max_length` (pixels).

Chaque effet tire dans son propre flux, `[seed, crc32(étiquette),
graine_de_pose]` (étiquettes `"row_noise"`, `"periodic"`, `"cosmic"`). Les
charges cosmiques entrent dans la chaîne commune avant le puits ; le bruit
de ligne et les motifs, après la conversion en DN. Les réalisations sont
renvoyées pour la vérité : `acq.row_offsets` (K, H), `acq.periodic`
(K, H, W), `acq.cosmic_mask` (K, H, W), et en push broom
`line_offsets` (M,), `periodic` (M, W), `cosmic_mask` (M, W).
`truth.correlated_variance` donne la variance par pixel due aux effets
temporels corrélés, σ_ligne² + Σ A²/2.

Note : la séquence de bruit de ligne est normalisée à un écart-type
réalisé exactement égal à `row_noise`. En 1/f, la variance *temporelle*
par pixel (d'une pose à l'autre) est plus petite que σ_ligne², car la part
basse fréquence varie peu entre poses voisines. Le test compare à la
variance réalisée par ligne.

### 4.8 Courant d'obscurité et overscan (S1)

Paramètres de `SensorParams`, nuls par défaut : `dark_current` (e⁻/s à
`temperature_ref`), `dark_nonuniformity` (s), `hot_pixel_fraction`,
`hot_pixel_range`, `temperature_ref` (°C), `bandgap` (eV), `glow_rate`
(e⁻/s), `glow_scale` (pixels), `bias_jitter` (DN), `bias_drift` (DN par
pose), `overscan_columns`.

- `expose(..., exposure_time=1.0, temperature=None)` et `dark(...)` :
  temps de pose (s) et température (°C). `photons` reste la quantité
  intégrée sur la pose.
- Vérité : `truth.dark_current` (`DarkCurrentMap` : `current`,
  `relative`, `hot`, `hot_factor`, `glow`), `truth.hot_pixels`,
  `truth.dark_electrons(t, T)`, `truth.expected_mean(photons, t, T)`,
  `truth.overscan_offset` (H, n) = offset + composante ligne de la DSNU.
- Réalisations : `acq.bias_offsets` (K,), `acq.overscan` (K, H, n),
  `acq.exposure_time`, `acq.temperature`.
- Flux aléatoires propres : `"dark_current"`, `"hot_pixels"` (motifs
  fixes), `"bias"`, `"overscan"` (temporels).
- Push broom : courant par détecteur, temps d'intégration d'une ligne ;
  gigue d'offset tirée une fois par image et dérive par ligne. Pas
  d'overscan (refusé).

### 4.9 Transmission, non-linéarité, défauts de réponse (S4)

Paramètres : `vignetting`, `n_dust`, `dust_depth`, `dust_radius`,
`nonlinearity` (β₀), `nonlinearity_dispersion` (écart-type de δβ),
`dead_pixel_fraction`, `stuck_pixel_fraction`. Flux propres :
`"transmission"`, `"nonlinearity"`, `"defects"`.

- Vérité : `truth.transmission` (`Transmission` : `total`, `vignetting`,
  `dust`, centres et rayons), `truth.nonlinearity_map` (β par pixel),
  `truth.defects` (`Defects` : `dead`, `stuck`, `stuck_value`).
  `signal_electrons` inclut T et annule les pixels morts ;
  `expected_mean` et `expected_variance` incluent la non-linéarité (terme
  de Poisson du second ordre pour la moyenne, compression au premier ordre
  pour la variance) et les pixels bloqués.
- Chaîne : la non-linéarité s'applique dans `_digitize` après le puits ;
  le code des pixels bloqués remplace la sortie du convertisseur.
- Les cartes (H, W) ou (1, W) sont diffusées, si bien que les effets
  valent aussi en push broom. Les colonnes d'overscan, pixels virtuels,
  n'en reçoivent aucun.

### 4.10 Pixels télégraphiques, convertisseur, whisk broom (S4)

- `rts_fraction`, `rts_amplitude` (DN), `rts_rate` (p_↑, p_↓ par pose) ;
  flux `"rts"` (carte) et `"rts_state"` (chaîne de Markov par pose ou par
  ligne). Vérité : `truth.rts` (`RTSMap` : `index`, `amplitude`, `p_up`,
  `p_down`, `high_fraction`) ; réalisation `acq.rts_states` (K, n_rts).
  Le saut s'ajoute en DN.
- `adc_dnl` (LSB), `adc_missing_codes` ; flux `"adc"`. Vérité :
  `truth.adc` (`ADCTransfer` : `thresholds`, `widths`, `missing`, `dnl`,
  `inl`). `_digitize` quantifie alors par `searchsorted` sur les seuils au
  lieu de l'arrondi.
- `WhiskbroomSensor(params, scan_offset=, scan_gain=)` : `params.shape` =
  (1, k). `expose(photons, n_scans=, width=, seed=)` renvoie l'image
  (n_scans·k, W), le détecteur et le sens de chaque ligne, et le profil
  attendu. Il ne simule ni les effets S3, ni les pixels télégraphiques,
  ni l'overscan.
