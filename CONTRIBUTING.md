# Contribuer à eoqual-measure

Les contributions sont bienvenues : signalement de bug, correction,
proposition ou implémentation d'une mesure, amélioration de la
documentation.

## Signaler un bug

Ouvrez une issue avec :
- la mesure et l'algorithme concernés (`list_measures()`,
  `list_noise_methods()`) ;
- un exemple minimal reproduisant le problème — de préférence sur une
  scène générée par `eoqual.measure.synthetic` (vérité terrain connue) ;
- le résultat obtenu vs. attendu.

## Ajouter une mesure

Une mesure contrôlée a sa place ici si sa cible ou son protocole est
connu et si elle produit une grandeur physique. Un indicateur calculé sur
scène quelconque relève plutôt d'`eoqual-metrics`.

1. **Registre** : déclarer (ou passer à `"available"`) l'entrée dans
   `config.py::MEASURES_CONFIGS` — famille, entrée attendue, grandeurs
   produites, point d'entrée `module:fonction`, `builds_on`, référence.
2. **Code** : un sous-paquet par famille (`radiometry/`, `contrast/`,
   `psf/`...), une fonction d'entrée simple renvoyant un résultat typé
   (dataclass), exposée dans le `__init__.py` de la famille. Une mesure
   qui en utilise une autre passe par son API publique
   (`estimate_noise`, `measure_mtf`), jamais par ses fonctions privées.
3. **Plusieurs algorithmes ?** Suivre le modèle classe de base +
   comparateur de `mtf/slanted_edge/` pour pouvoir les exécuter côte à
   côte.
4. **Code réutilisable** par d'autres modules EOQual (numpy/scipy
   seulement) : le proposer à `eoqual-core` plutôt que de le garder ici
   (voir ses règles d'admission).
5. **Validation** : un générateur dans `synthetic/` et des tests qui
   retrouvent la vérité terrain dans une tolérance justifiée.
6. **Dépendances** : plotly/pandas et toute dépendance lourde en extra,
   importées à l'appel.

## Licence — condition de fusion

Toute implémentation reprenant ou s'inspirant de près d'un code publié
ailleurs doit citer sa source et la licence *vérifiée* de cette source,
dans le module et dans `THIRD_PARTY_LICENSES.md`. Une implémentation de
provenance non établie, ou sous licence non permissive, n'entre pas dans
le socle.

## Tests

```bash
pip install git+https://github.com/EOQual/eoqual-core.git   # ou PYTHONPATH vers son src/
PYTHONPATH=src pytest tests
```

## Documentation

Toute mesure nouvelle ou modifiée doit rester reflétée dans `README.md`
(catalogue) et `REFERENCE_TECHNIQUE.md` (principe, implémentation,
limites).
