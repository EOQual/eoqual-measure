"""
Étude : performances de détection de `find_structured_noise`.

30 capteurs de 256 × 256 (graines 0 à 29), 16 poses d'obscurité,
p_FA = 10⁻³ par spectre ; pour chaque cas, nombre d'analyses avec au
moins une fausse alarme, motifs manqués, erreurs de nature (fixe /
temporel) et amplitude mesurée. Sortie : tableau Markdown repris dans
REFERENCE_TECHNIQUE.md, partie IV, §12.4.

Usage : python examples/structured_noise_detection_study.py [n_capteurs] [fraction_min]
"""

import sys

import numpy as np

from eoqual.measure.noise import find_structured_noise
from eoqual.measure.synthetic import MatrixSensor, PeriodicPattern, SensorParams

CASES = [
    ("sans effet", {}, None),
    ("bruit de ligne blanc, 1 DN", dict(row_noise=1.0), None),
    ("bruit de ligne 1/f, 1 DN", dict(row_noise=1.0, row_noise_beta=1.0), None),
    ("colonnage fixe, 2 DN", dict(dsnu_col=2.0), None),
    ("lignes fixes, 2 DN", dict(dsnu_row=2.0), None),
    ("motif temporel 0,3 DN (0,125 ; 0,0625)",
     dict(periodic=(PeriodicPattern(0.3, fx=0.125, fy=0.0625),)), ("temporal", 0.125, 0.0625)),
    ("motif fixe 0,3 DN (0,125 ; 0,0625)",
     dict(periodic=(PeriodicPattern(0.3, fx=0.125, fy=0.0625, temporal=False),)), ("fixed", 0.125, 0.0625)),
    ("motif temporel 0,5 DN sur l'axe (0,25 ; 0)",
     dict(periodic=(PeriodicPattern(0.5, fx=0.25, fy=0.0),)), ("temporal", 0.25, 0.0)),
    ("motif temporel 0,5 DN près de l'axe (1/64 ; 0,2)",
     dict(periodic=(PeriodicPattern(0.5, fx=1 / 64, fy=0.2),)), ("temporal", 1 / 64, 0.2)),
]


def main(n: int = 30, min_fraction: float = 0.0) -> None:
    print("| Cas | Analyses avec fausse alarme | Motif manqué | Mauvaise nature | Amplitude mesurée |")
    print("|---|---|---|---|---|")
    for label, kw, expected in CASES:
        fa = miss = wrong = 0
        amps = []
        for seed in range(n):
            s = MatrixSensor(SensorParams(shape=(256, 256), seed=seed, **kw))
            r = find_structured_noise(s.dark(n_frames=16, seed=1).frames, min_variance_fraction=min_fraction)
            hits = [q for q in r.peaks if expected and abs(q.fx - expected[1]) < 0.02 and abs(q.fy - expected[2]) < 0.02]
            fa += int(len(r.peaks) > len(hits))
            if expected:
                if not hits:
                    miss += 1
                else:
                    amps.append(hits[0].amplitude)
                    wrong += int(any(h.kind != expected[0] for h in hits))
        amp = f"{np.mean(amps):.3f} ± {np.std(amps):.3f}".replace(".", ",") if amps else "—"
        print(f"| {label} | {fa}/{n} | {miss if expected else '—'} | {wrong if expected else '—'} | {amp} |")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30, float(sys.argv[2]) if len(sys.argv) > 2 else 0.0)
