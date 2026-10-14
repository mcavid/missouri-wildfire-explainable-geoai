# Methods and computational scope

The binary response is at least one recorded wildfire in each approximately
5-km cell during 2021–2025. The frozen encoded checkpoint supplies the original
33 predictors in scripts/project.py; IDs, response and saved OOF fields are excluded.
Five continuous rows and one land-cover class were filled before CV. Two continuous
donor pairs cross an outer-fold boundary; no labels were copied. Model evaluation
therefore uses a fixed statewide preprocessed covariate surface.

Baseline RF uses 500 trees and seed 42, otherwise estimator defaults, comparing
StratifiedKFold (shuffle, seed 42) with unshuffled GroupKFold. Nested RF and XGBoost
share five outer and four inner spatial folds, with outer seed 42 and inner seeds
43–47. Fifty-km blocks remain intact within each split. Grouping is not buffering.

The nested RF grid has 12 candidates (trees 300/500, depth 8/10/None, leaf 1/3,
sqrt feature sampling, balanced classes). The full-data mapping RF comes from a
separate 72-candidate search and uses 500 trees, depth 10, leaf 1, sqrt features,
balanced classes and seed 42. Nested XGBoost uses the recorded 64-candidate grid.
The grids, seeds and versions are in results/reproducibility/validation_config.json.

Full-data TreeExplainer uses all 7,229 rows and the final RF. Resolved settings are
tree_path_dependent, raw output and no background dataset. Its cached class-1
contributions reconstruct fitted RF scores to numerical precision. This is distinct
from held-out predictive evaluation and does not establish causal effects.

Supplementary block-size and predictor-omission comparisons use the saved five-fold
metric tables. Fold-SHAP summaries rank the saved 33-feature importance vectors,
using average ties and Spearman correlations. Their reproduction entry point is
the recorded summaries, rather than a claim that the original upstream fitting
or explanation protocol has been independently regenerated.
