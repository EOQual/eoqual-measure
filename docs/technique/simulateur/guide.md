# Simulateur : guide d'utilisation

## 5. Guide d'utilisation

Installation : celle d'eoqual-measure (`README.md`) ; le simulateur
n'ajoute aucune dépendance (NumPy seul). Tests :

```bash
python -m pytest tests/test_synthetic_*.py
```

**Un capteur, une pile d'obscurité, une pile de champ uniforme :**

```python
from eoqual.measure.synthetic import MatrixSensor, SensorParams

sensor = MatrixSensor(SensorParams(seed=1))
truth = sensor.truth

dark = sensor.dark(n_frames=32, seed=10)               # (32, 256, 256) uint16
photons = truth.photons_for_signal(1000.0)             # 1 000 DN au-dessus de l'offset
flat = sensor.expose(photons, n_frames=2, seed=11)

print(truth.a, truth.b)                 # 1/K et σ²(0) en DN²
print(truth.dsnu_rms, truth.prnu_rms)   # DSNU (DN) et PRNU (relative)
```

**Capteur à défauts de colonnes marqués** (pour la décomposition
ligne/colonne) :

```python
p = SensorParams(seed=2, dsnu_col=2.0, prnu_col=0.01, dsnu_row=0.5)
sensor = MatrixSensor(p)
sensor.truth.dsnu.col      # composante colonne vraie (W,)
```

**Valider une mesure contre la vérité** (modèle d'un test) :

```python
import numpy as np

acq = sensor.dark(n_frames=64, seed=3)
offset_estimate = acq.frames.mean(axis=0)               # mesure « naïve »
error = offset_estimate - acq.truth.offset_map
assert abs(error.std() - np.sqrt(acq.truth.b / 64)) < 0.05
```

**Série pour une PTC :**

```python
levels = [100, 400, 1000, 2000, 3000]                   # DN
pairs = [sensor.expose(truth.photons_for_signal(s), n_frames=2, seed=100 + i)
         for i, s in enumerate(levels)]
```

**Saturation par le puits plutôt que par le convertisseur :** choisir
`full_well < K·(2ⁿ − 1 − offset)`, par exemple `full_well=6000` avec les
valeurs par défaut.

**Barrette push broom et scène naturelle :**

```python
from eoqual.measure.synthetic import PushbroomSensor, SensorParams, natural_scene, pattern_rng

pb = PushbroomSensor(SensorParams(shape=(1, 1024), seed=3, dsnu_pixel=0.0, dsnu_col=0.8,
                                  prnu_pixel=0.0, prnu_col=0.01), tdi_stages=4)
t = pb.truth
scene = natural_scene((2000, 1024), t.photons_for_signal(1500.0), pattern_rng(0, "scene"),
                      contrast=0.02, beta=3.0, psf_sigma=0.6,
                      gradient_along=0.1, gradient_across=0.05)
img = pb.expose(scene.photons, seed=4).image             # (2000, 1024) uint16
dark = pb.dark(n_lines=500, seed=5).image                # bloc d'obscurité
t.detector_offset, t.detector_gain                       # vérité par détecteur
```

**Bruits structurés (S3) et traitement :**

```python
from eoqual.measure.synthetic import (MatrixSensor, SensorParams, PeriodicPattern,
                                  apply_kernel, gaussian_kernel, kernel_power_transfer)

s3 = MatrixSensor(SensorParams(seed=7, row_noise=1.0, row_noise_beta=1.0,
                               periodic=(PeriodicPattern(0.5, fx=0.125, fy=0.0625),
                                         PeriodicPattern(0.3, fx=0.2, fy=0.0, temporal=False)),
                               cosmic_rate=3.0))
acq = s3.dark(n_frames=16, seed=1)
acq.row_offsets, acq.periodic, acq.cosmic_mask           # réalisations (vérité)

k = gaussian_kernel(1.0)
processed = apply_kernel(acq.frames.astype(float), k)    # bruit corrélé
h2 = kernel_power_transfer(k, (64, 64))                  # |H|² : NPS attendu = σ²·|H|²
```

**Courant d'obscurité, température, overscan (S1) :**

```python
s1 = MatrixSensor(SensorParams(seed=8, dark_current=2.0, dark_nonuniformity=0.3,
                               hot_pixel_fraction=1e-3, glow_rate=20.0,
                               bias_jitter=0.5, overscan_columns=16))
cold = s1.dark(n_frames=16, seed=1, exposure_time=60.0, temperature=10.0)
warm = s1.dark(n_frames=16, seed=2, exposure_time=60.0, temperature=30.0)
s1.truth.dark_electrons(60.0, temperature=30.0)          # charge vraie par pixel (e⁻)
cold.overscan.mean(axis=(1, 2))                          # suit cold.bias_offsets
```

**Robustesse des tolérances** : la variable `EOQUAL_TEST_SEED` décale
les graines des capteurs de test.

```bash
for s in $(seq 0 29); do EOQUAL_TEST_SEED=$s .venv/bin/python -m pytest -q | tail -1; done
```

### Valeurs par défaut (`SensorParams()`)

| Paramètre | Valeur | Unité |
|---|---|---|
| `shape` | (256, 256) | pixels |
| `qe` | 0,7 | e⁻/photon |
| `conversion_gain` (K) | 3,0 | e⁻/DN |
| `read_noise` | 5,0 | e⁻ |
| `full_well` | 15 000 | e⁻ |
| `bit_depth` | 12 | bits |
| `offset` | 100 | DN |
| `dsnu_pixel` / `dsnu_row` / `dsnu_col` | 0,5 / 0 / 0,3 | DN |
| `prnu_pixel` / `prnu_row` / `prnu_col` | 0,5 % / 0 / 0,2 % | — |

Grandeurs dérivées : a = 2,86 DN² (σ_lect = 1,67 DN) ; b = 0,333 ;
saturation du convertisseur à 4 095 DN, avant le puits (5 100 DN).
