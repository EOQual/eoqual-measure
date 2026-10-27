"""
Étude : biais de `measure_pushbroom_noise` selon la texture de la scène.

Pour chaque scène (exposant β, contraste c, flou optique σ_psf), cinq
images de 1 024 × 513 à 200, 600, 1 200, 2 000 et 3 000 DN, et l'on
compare la pente a mesurée à la vérité 1/K, en unités de son incertitude
u(a), ainsi que le rapport de cohérence. Sortie : tableau Markdown repris
dans REFERENCE_TECHNIQUE.md, partie IV, §10.4.

Usage : python examples/pushbroom_texture_bias_study.py
"""

import numpy as np

from eoqual.measure.radiometry import measure_dark_frame, measure_pushbroom_noise
from eoqual.measure.synthetic import PushbroomSensor, SensorParams, natural_scene, pattern_rng

LEVELS = [200, 600, 1200, 2000, 3000]
SHAPE = (1024, 513)
CASES = [
    (3.0, 0.00, 0.0),
    (3.0, 0.03, 0.0),
    (3.0, 0.03, 0.6),
    (3.0, 0.10, 0.6),
    (2.5, 0.10, 0.6),
    (2.0, 0.03, 0.6),
    (2.0, 0.03, 1.0),
    (2.0, 0.10, 0.6),
    (2.0, 0.10, 0.0),
]
FILTERS = [(2, 2), (2, 1), (1, 1), (2, 0)]


def main() -> None:
    pb = PushbroomSensor(SensorParams(shape=(1, SHAPE[1]), seed=21, dsnu_pixel=0.0, dsnu_col=0.8,
                                      prnu_pixel=0.0, prnu_col=0.01))
    t = pb.truth
    dark = measure_dark_frame(pb.dark(n_lines=1000, seed=1).image, layout="pushbroom")
    head = "| β | c | σ_psf | " + " | ".join(f"(a − 1/K)/u [{fa},{fc}]" for fa, fc in FILTERS) + " | cohérence [2,2] |"
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    for beta, c, psf in CASES:
        imgs = []
        for i, lvl in enumerate(LEVELS):
            sc = natural_scene(SHAPE, t.photons_for_signal(lvl), pattern_rng(i, "scene"),
                               contrast=c, beta=beta, psf_sigma=psf,
                               gradient_along=0.1, gradient_across=0.05)
            imgs.append(pb.expose(sc.photons, seed=100 + i).image)
        cells = []
        coh = ""
        for fa, fc in FILTERS:
            r = measure_pushbroom_noise(imgs, dark=dark, along_order=fa, across_order=fc)
            cells.append(f"{(r.a - t.a) / r.u_a:+.1f}")
            if (fa, fc) == (2, 2):
                flag = " ⚠" if r.texture_suspected else ""
                coh = f"{(r.consistency - 1) / r.u_consistency:+.1f} u{flag}"
        print(f"| {beta} | {c:.0%} | {psf} | " + " | ".join(cells) + f" | {coh} |")


if __name__ == "__main__":
    main()
