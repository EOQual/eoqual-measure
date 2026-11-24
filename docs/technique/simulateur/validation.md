# Simulateur : validation, limites et formulaire

## 6. Validation

Le simulateur est vérifié contre sa propre vérité. Chaque tolérance est
déduite de l'incertitude statistique de l'estimateur ([fondements, §2.6](../radiometrie/02-fondements.md)), en général à
5 σ, et non fixée au jugé.

| Test | Vérifie |
|---|---|
| `test_same_seed_same_sensor`, `test_other_seed_other_sensor` | identité du capteur par sa graine |
| `test_temporal_seed` | reproductibilité et indépendance des poses |
| `test_fixed_pattern_does_not_depend_on_other_patterns` | séparation des flux aléatoires |
| `test_maps_are_centred` | offset moyen et gain moyen exacts |
| `test_component_amplitudes` | σ_p, σ_r, σ_c de la DSNU et de la PRNU, à 5/√(2(n−1)) près |
| `test_dark_stack_mean_and_variance` | moyenne temporelle = carte d'offset, à √(a/K) ; variance = a |
| `test_flat_mean_and_variance` (50, 500, 2 500 DN) | E[y] et Var[y] = a + b·μ |
| `test_photon_transfer_slope_is_one_over_gain` | pente de la PTC par paires = 1/K à 2 % |
| `test_adc_saturation`, `test_full_well_clipping` | plafonds ; variance réduite à a au plateau du puits |
| `test_output_format`, `test_scene_map_input` | type `uint16`, forme, contrôles d'entrée |
| `test_power_law_1d_slope`, `test_power_law_2d_radial_slope` | pente spectrale −β à 0,1 près |
| `test_power_law_normalised` | moyenne 0, écart-type 1 |
| `test_detector_statistics` (push broom) | moyenne par détecteur = offset + μ·g ; variance = a + b·μ |
| `test_lines_are_independent` | autocorrélation le long de la trace nulle à 5/√n près |
| `test_tdi` (N = 1, 4, 16) | signal ×N à flux par étage égal ; variance = a + b·μ |
| `test_natural_scene_components`, `test_scene_not_periodic` | texture normée, gradient exact, texture non périodique |
| `test_scene_psf_attenuates_high_frequencies` | spectre 2D de la texture × FTM² exactement (champ périodique) |
| `test_effects_off_by_default` | S3 désactivé par défaut |
| `test_row_noise` (β = 0 et 1) | moyennes de lignes = offsets réalisés à √(a/W) près ; variance temporelle = a + variance par ligne des offsets |
| `test_row_noise_spectrum` | pente −1 à 0,15 près pour β = 1 |
| `test_periodic_temporal_and_fixed` | amplitude projetée exacte, phase variable (temporel) ou constante (fixe) |
| `test_cosmic_rays` | charge/K sur les pixels touchés ; écrêtage au puits |
| `test_pushbroom_line_noise` | bruit commun à une ligne acquise |
| `test_apply_kernel_nps` | NPS moyen = σ²·\|H\|² ; variance σ²·Σh² ; noyau centré |
| `test_s3_params_validation` | contrôles d'entrée |
| `test_off_by_default` (S1) | sans courant, le temps de pose est sans effet |
| `test_dark_mean_and_poisson_variance` | moyenne = offset + d_e/K ; variance = a + d_e/K² |
| `test_dark_linear_in_exposure_time` | pente = courant moyen / K à 1 % |
| `test_temperature_law` | doublement exact 8,29 K à 20 °C |
| `test_lognormal_map_and_hot_pixels` | moyenne 1, DSNU √(e^{s²} − 1), nombre et facteurs (log-uniformes) des pixels chauds |
| `test_amplifier_glow` | lueur au coin, décroissance exp(−d/λ) |
| `test_bias_and_overscan_track_each_other` | l'overscan suit la variation d'offset ; correction de l'offset actif |
| `test_pushbroom_dark_and_drift`, `test_s1_validation` | push broom ; contrôles d'entrée |
| `test_transmission` (S4) | vignettage 1 − v aux coins ; poussières dans [(1 − d)ⁿ, 1 − 0,07] ; flat moyen = T·g |
| `test_nonlinearity_mean_and_variance` (500, 3 000 DN) | moyenne (terme de Poisson) et variance comprimée ; compression visible ; dispersion de β |
| `test_dead_and_stuck_pixels` | nombres exacts, ensembles disjoints ; morts à l'offset avec variance a ; bloqués constants |
| `test_pushbroom_response_effects`, `test_s4_validation` | effets en push broom ; contrôles d'entrée |
| `test_rts_levels_and_rates` | écart des niveaux = amplitude ; fréquences de bascule à 5/√n ; fraction haute stationnaire |
| `test_rts_pushbroom_along_lines` | chaîne de Markov le long de la trace |
| `test_ideal_adc_equals_rounding` | sans DNL, quantification = arrondi |
| `test_unequal_steps_and_missing_codes` | codes manquants jamais produits ; occurrences ∝ largeur sur une rampe |
| `test_whiskbroom_bands_and_scan_direction` | profil des lignes = attendu ; différence aller/retour |
| `test_s4_rest_validation` | contrôles d'entrée |
| `test_shapes_and_format`, `test_scene_input` | formes, type, contrôles d'entrée |

Robustesse : les 62 tests passent pour 30 décalages de graine
(`EOQUAL_TEST_SEED` de 0 à 29).

## 7. Limites et choix de modélisation

**Limites du modèle**

- *Quantification* : le terme 1/12 de a suppose σ_lect/K ≳ 0,5 DN. Pour
  un capteur à très faible bruit (σ < 0,5 DN), la variance réelle dépend
  de la position de l'offset entre deux codes. Le simulateur reste exact
  (il arrondit vraiment), mais `truth.a` n'est plus qu'une approximation.
- *Saturation* : `expected_mean` et `expected_variance` ignorent les
  écrêtages. Près du plafond, comparer aux images et non à ces formules.
- *Puits* : pas de débordement vers les voisins (*blooming*), pas de
  perte d'efficacité de transfert (CTE), réponse parfaitement linéaire
  jusqu'au plafond.
- *PRNU gaussienne* : non bornée. Le capteur est refusé si un gain
  devient négatif ou nul, ce qui n'arrive que pour des PRNU irréalistes
  (plus de 20 % environ).
- *Pas de dépendance spectrale* : η est un scalaire. La QE(λ) relève de
  l'étalonnage spectral et absolu, hors du périmètre des mesures
  contrôlées, et aucune mesure n'en dépend.
- *Push broom* : un gain par détecteur. En TDI, la dispersion propre des
  N étages (moyennée par le TDI) et le filé (défaut de synchronisation du
  transfert) ne sont pas simulés.
- *Scènes* : texture gaussienne isotrope, sans structure (routes,
  parcelles, nuages) ni réflectance bornée. PSF gaussienne seulement.
- *S3* : pas de bruit de colonne temporel, pas de diaphotie ; rayons
  cosmiques en segments droits à dépôt uniforme (ni diffusion de charge, ni
  distribution de Landau).
- *S1* : un seul mécanisme de courant (génération en zone de déplétion,
  E_g/2) pour tous les pixels, pixels chauds compris (ils ont souvent une
  énergie d'activation différente) ; lueur purement exponentielle ; pas de
  persistance d'image (rémanence) ni de courant d'obscurité dépendant du
  signal.
- *S4, transmission et réponse* : vignettage quadratique isotrope ; non-linéarité
  quadratique seulement (ni « deux pentes », ni coude près de la
  saturation) ; pixels bloqués à un code uniforme (souvent, en réalité, le
  code maximal ou zéro).
- *S4, RTS, convertisseur, whisk broom* : RTS à deux niveaux seulement (pas de pièges multiples) ;
  DNL gaussienne non corrélée d'un seuil à l'autre (pas de motif lié aux
  bits de poids fort) ; whisk broom sans dépendance temporelle à
  l'intérieur d'un balayage.

**Choix de modélisation**

- *Capacité du puits par défaut de 15 000 e⁻* : avec 12 000 e⁻, le plateau
  du puits (4 000 + 100 DN) tombe au niveau du code maximal (4 095 DN), et
  les deux saturations se mêlent selon la DSNU du pixel. À 15 000 e⁻, elles
  restent distinctes.
- *PSF optique appliquée aux scènes* : sans optique, une texture fBm a trop
  de puissance à Nyquist pour qu'aucune mesure en une image ne la sépare du
  bruit, situation irréaliste pour un imageur.
- *Température de doublement exacte* : `doubling_temperature` résout
  I(T + ΔT) = 2·I(T) (8,29 K à 20 °C) ; l'écart au premier ordre (8,07 K)
  ne multiplie le courant que par 1,965.
- *Push broom* : la non-linéarité et les défauts s'appliquent aussi à une
  carte (1, W) diffusée sur M lignes ; l'overscan en est exclu
  explicitement.

**Pièges courants des simulateurs de bruit, évités ici**

| Piège | Conséquence | Choix retenu |
|---|---|---|
| Poisson du courant d'obscurité tiré en DN (`poisson(I_d·t/K)`) | variance fausse d'un facteur K | tout tirage de Poisson en électrons |
| Arrondi des électrons après application de η, et du bruit de lecture | discrétisation parasite avant le convertisseur | Poisson(η·μ_p) direct, seul arrondi au convertisseur |
| Conversion par `astype(int16)` | troncature au lieu d'arrondi, repli des valeurs négatives | `floor(y + 0.5)` puis écrêtage à [0, 2ⁿ − 1], `uint16` |
| Deux gains incohérents (5,88 DN/e⁻ et 1 e⁻/DN) | modèle de bruit inexploitable | un seul K (e⁻/DN), b = 1/K |
| Mauvaises colonnes multipliant l'image finale en DN | l'offset est lui aussi multiplié | colonne = offset (DSNU) ou gain du signal (PRNU) |
| Motifs fixes et bruit temporel tirés du même générateur | ajouter un effet change tous les tirages | un flux par motif, nommé par étiquette |

## 8. Formulaire

| Grandeur | Expression | Unité |
|---|---|---|
| Signal en électrons | μ_e = η·μ_p·(1 + PRNU) | e⁻ |
| Signal en DN au-dessus de l'offset | μ = μ_e / K | DN |
| Moyenne d'un pixel | E[y] = offset + DSNU + μ | DN |
| Variance temporelle | σ²(μ) = a + b·μ | DN² |
| Variance à signal nul (plancher) | a = (σ_lect² + σ_d²)/K² + 1/12 | DN² |
| Pente | b = 1/K = K_E | DN/DN |
| Photons pour un signal s | μ_p = s·K/η | photons |
| Bruit temporel par paire | σ_temp² = var(x₁ − x₂)/2 | DN² |
| Bruit spatial corrigé | σ_spatial² = s²(moyenne de K poses) − σ_temp²/K | DN² |
| Variance des moyennes de lignes | σ_r² + σ_p²/W | DN² |
| Incertitude relative d'une variance | √(2/(n − 1)) | — |
| SNR | μ / √(a + b·μ) | — |
| Courant d'obscurité (S1) | d_e = I_d(T)·t_exp, I_d ∝ T³·exp(−E_g/2kT) | e⁻ |
| TDI à N étages | μ_e = N·η·Φ·g ; σ² = a + b·μ inchangé | e⁻ |
| FTM de la PSF gaussienne | exp(−2π²σ_psf²f²) | — |
| Scène naturelle | Φ₀·(1 + g_∥u + g_⊥v)·(1 + c·T) | photons |
| Motif périodique | A·cos(2π(f_x j + f_y i) + φ), variance A²/2 | DN |
| Bruit filtré par h | NPS = σ²·\|H(f)\|², variance σ²·Σh² | DN² |
| Carte de courant | I_d·g, ln g ~ N(−s²/2, s²) ; DSNU relative √(e^{s²} − 1) | e⁻/s |
| Doublement à 20 °C (E_g = 1,12 eV) | 8,29 K (exact) | K |
| Vignettage | T_v = 1 − v·ρ² | — |
| Non-linéarité | e' = e(1 − βe/FW) ; E[e'] = μ − β(μ² + μ)/FW ; κ = −βK/FW | e⁻, DN⁻¹ |
| RTS | séjours moyens 1/p_↑, 1/p_↓ ; fraction haute p_↑/(p_↑ + p_↓) | poses |
| Convertisseur | w_c = t_c − t_{c−1} ; DNL_c = w_c − 1 ; INL_c = t_c − (c + ½) | LSB |
