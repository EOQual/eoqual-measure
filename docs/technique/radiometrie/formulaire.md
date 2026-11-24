# Formulaire

Expressions de référence, par chapitre.

## Fondements communs (chapitre 2)

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

## Obscurité, bruit temporel, PTC (chapitre 3)

| Grandeur | Expression |
|---|---|
| Carte d'offset | ô = (1/K) Σ_k y_k |
| a | â = moyenne_ij [(1/(K − 1)) Σ_k (y_k − ô)²] ; u = â·√(2/((K − 1)N)) |
| DSNU | √(var(ô) − â/K) ; u² ≈ [4·DSNU²·v + 2v²(1 + 1/(K − 1))]/N / (4·DSNU²), v = â/K |
| Composantes | s²_pe = var(P)·HW/((H − 1)(W − 1)) ; σ_r² = var(R) − (s²_pe/W)(H − 1)/H ; σ_p² = s²_pe − v |
| Stationnarité | var(f)/(â/N) ≈ 1 ± √(2/(K − 1)) |
| Bruit temporel (paire) | σ² = var(x₁ − x₂)/2 ; u = σ²·√(2/(N − 1)) |
| PTC | σ² = a + b·μ ; K = 1/b ; u(K) = u(b)/b² |
| Bruit de lecture | σ_lect = K·√(a − 1/12) |
| Saturation (puits) | pic de variance vers FW − 2,8√FW |
| Linéarité | LE = (max δ − min δ)/2, δ = (μ − (s·x + c))/(s·x + c) en % |

## Courant d'obscurité (chapitre 4)

| Grandeur | Expression |
|---|---|
| Courant par pixel | Î = Σ (t_i − t̄)(m_i − m̄)/S_xx ; Var(Î) = Σ (t_i − t̄)²·v_i/K_i / S_xx² |
| Variance temporelle | v(t) = a + I·t/K (DN²) |
| Gain poissonnien | K = Ī / (dv/dt) |
| DSNU du courant | √(var(Î) − mean Var(Î)) |
| Loi en température | ln I − 3 ln(T/T₀) = ln I₀ − (E_g/2k)(1/T − 1/T₀) |
| Temps de pose minimal | t > σ_lect / I_d |

## Push broom et SNR (chapitre 10)

| Grandeur | Expression |
|---|---|
| Noyaux | ordre 1 : [1, −1]/√2 ; ordre 2 : [1, −2, 1]/√6 |
| Autocorrélation du bruit filtré | ρ_k = Σ_i k_i·k_{i+k} ; ordre 2 : ρ₁ = −2/3, ρ₂ = 1/6 |
| Degrés de liberté effectifs | ν = n / Σ_k ρ_k² (ordre 1 : n/1,5 ; ordre 2 : n/1,944) |
| Variance d'un segment | s² = mean(f²) ; Var(s²) = 2σ⁴/ν |
| Transmission de la scène | ∫ P_scène(f)·FTM²(f)·\|H(f)\|² df |
| Poids de l'ajustement | w = ν / (2·(a + b·μ)²) |
| Cohérence | R = mean(s²_contrôle)/mean(s²) ; u(R) = sd(x − R·y)/(√n·ȳ) ; alerte si \|R − 1\| > 5 u |
| Bartlett, retard 1 | Var(r₁) = Σ_{k≥1} (ρ_{k+1} + ρ_{k−1} − 2ρ₁ρ_k)² / n |
| SNR | μ/√(a + b·μ) ; u(σ)² = J·cov·Jᵀ, J = (1/2σ, μ/2σ) |

## Bruit spatial et égalisation (chapitre 11)

| Grandeur | Expression |
|---|---|
| Bruit spatial (pile) | σ² = var(m̄) − σ²_temp/K (− b/K_d si obscurité retirée) |
| Bruit temporel push broom | σ²_temp = mean((x_{i+1} − x_i)²)/2 |
| Bandes | σ_q² = var(Q) − (s²_intra/n)·(k − 1)/k |
| Coupure BF/HF | σ_g = √(ln 2/2)/(π·f_c) |
| BF, HF | σ²_BF = var(h∗p) − v·Σh² ; σ²_HF = var(p − h∗p) − v·Σ(δ − h)² |
| Incertitude | Var(Ŝ) ≈ [4Sv + 2v²·Σ(g⋆g)² + 2(vΣg²)²/M]/W ; u(σ) = √Var/(2σ) |
| Spectre | S(f) unilatéral, Σ S·Δf = var(p) ; bruit blanc : S = 2σ² |

## NPS et bruits structurés (chapitre 12)

| Grandeur | Expression |
|---|---|
| NPS d'une imagette | \|FFT(w·(x − plan))\|² / Σw² ; ∫∫ NPS = σ² |
| Bruit blanc filtré | NPS = σ²·\|H(f)\|² |
| Welch, Hann 50 % | ν ≈ n imagettes (1D : n/(1 + 2c²), c = 0,177) |
| Motif A·cos | variance A²/2, excès A²/4 en ±f₀ |
| Seuil | R > Γ⁻¹(1 − p_FA/(P²/2) ; ν) / médiane Gamma(ν) |
| ν par quantiles | Γ⁻¹(0,25 ; ν)/Γ⁻¹(0,5 ; ν) = q₂₅(R)/q₅₀(R) |
| Borne sur un axe | ν ≤ 0,95 × images × positions en travers |
| Fuite temporelle dans la moyenne | ≈ exponentielle de moyenne var/K ; seuil (var/K)·ln(1/p_FA) |
| Bruit de ligne | σ_ligne² = var(R) − σ_p²/W ; S(f) = 2σ_p²/W + c·f^{−β} |
