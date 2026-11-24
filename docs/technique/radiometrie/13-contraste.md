# 13. Contraste sur bruit d'une mire

## 13.1 Objet

CNR d'une mire à deux plages : CNR = (μ_clair − μ_sombre) /
√(σ²_clair + σ²_sombre). Le bruit d'une plage est la variance de ses
pixels dans **une** image, bruit temporel et bruit spatial fixe compris.
Le CNR se distingue du SNR radiométrique μ/σ(μ) du chapitre 10.

**Statut** : validée contre le simulateur S0 (`tests/test_p3_measures.py`,
2 tests ; robustes sur 30 graines). Module :
`eoqual.measure.contrast.cnr:measure_cnr` (famille `contrast` du registre
d'eoqual-measure).

## 13.2 Principe

- **Plages** : fournies (masques), ou détectées par le seuil d'Otsu sur
  l'image moyenne lissée (σ = 2 pixels), puis érodées de 3 pixels pour
  écarter la transition, que la FTM rend floue.
- **CNR par image** et incertitude par la méthode delta : u(μ) = σ/√n,
  u(σ²) = σ²·√(2/(n − 1)), soit
  Var(CNR) = (σ²_c/n_c + σ²_s/n_s)/S² + (CNR/(2S²))²·(2σ⁴_c/(n_c − 1) +
  2σ⁴_s/(n_s − 1)), avec S² = σ²_c + σ²_s.
- **Pile** : CNR moyen des images, d'incertitude max(u analytique,
  dispersion entre images)/√K.

## 13.3 Mise en œuvre

```python
import numpy as np
from eoqual.measure.synthetic import MatrixSensor, SensorParams, apply_kernel, gaussian_kernel
from eoqual.measure.contrast import measure_cnr

s = MatrixSensor(SensorParams(seed=3, prnu_pixel=0.01))
t = s.truth
i, j = np.mgrid[0:256, 0:256]
disk = (i - 128) ** 2 + (j - 128) ** 2 < 60**2
scene = np.where(disk, t.photons_for_signal(600.0), t.photons_for_signal(500.0))
scene = apply_kernel(scene, gaussian_kernel(1.5))                    # optique
r = measure_cnr(s.expose(scene, n_frames=20, seed=1).frames)
print(f"CNR = {r.cnr:.3f} ± {r.u_cnr:.3f} ({r.n_bright} et {r.n_dark} pixels, seuil {r.threshold:.1f} DN)")
```

## 13.4 Validation

Résultat type (disque à 600 DN sur un fond à 500 DN, PRNU 1 %, flou
σ = 1,5 px, 20 poses) : CNR = 4,794 ± 0,005 pour 4,786 vrai.

Liste des tests : §13.4.1.

### 13.4.1 Tests

| Test | Vérifie |
|---|---|
| `test_cnr_detected_zones` | CNR = vrai (moyennes et variances attendues des plages, FPN compris) à 5 u ; plages détectées à l'intérieur du disque et du fond |
| `test_cnr_given_masks_and_uncertainty` | masques fournis ; u analytique d'une image = dispersion entre images (40 poses) |

## 13.5 Limites

- Deux plages seulement : une mire à plusieurs plages se traite par
  masques, paire par paire.
- Le CNR dépend de la PRNU, puisque le FPN entre dans σ. Pour un CNR du
  seul bruit temporel, utiliser la différence de deux poses.
