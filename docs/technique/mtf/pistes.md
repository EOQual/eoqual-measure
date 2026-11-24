# MTF : pistes

## 12. Pistes

Pistes identifiées mais non implémentées.

### 12.1 Bords naturels / cibles urbaines

Le pipeline actuel suppose une mire dédiée à bord unique et net,
correctement cadrée. Pour une utilisation sur cibles urbaines/scènes
naturelles (moins de contraintes d'acquisition que les mires dédiées,
utile pour des passages proches du nadir), il faudrait une étape
préalable de **détection/sélection automatique de bords exploitables**
dans une scène quelconque, non implémentée ici : voir
[[VanZwanenberg2019]](references.md#10-bibliographie) et
[[Kohm2004]](references.md#10-bibliographie) (qui traite déjà les deux cas, bords
dédiés et naturels).

### 12.2 Rognage adaptatif de l'ESF (méthode KARI)

[[ViallefontRobinet2018]](references.md#10-bibliographie) décrit une méthode qui
rogne l'ESF sur-échantillonnée à un point de rupture ("knee point")
détecté sur la dérivée seconde de la LSF, plutôt qu'à une demi-largeur
fixe (`esf_half_width`, [§4.4](estimateurs.md#44-estimatorsesf_lsfpy--estimatorsesf_modelspy--esflsfmtfestimator)) -- pourrait améliorer la robustesse au
bruit et à la non-uniformité de la ROI. Non implémenté.

### 12.3 MTF 2D (PSF séparable)

Ce pipeline calcule uniquement une MTF **1D**, dans la direction
perpendiculaire au bord. Une extension possible consiste à construire une PSF 2D synthétique par produit
extérieur de la LSF 1D avec elle-même (`np.outer`), sous l'hypothèse
que la PSF est **séparable** (résolution identique dans les deux
directions) -- une approximation courante mais pas toujours valide pour
un système optique réel. La MTF 2D s'obtiendrait alors par FFT 2D de
cette PSF synthétique. Non implémenté ; utile seulement pour une
visualisation qualitative de la réponse fréquentielle dans toutes les
directions, pas pour une mesure indépendante.
