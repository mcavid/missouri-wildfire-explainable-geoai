# Reproduction instructions

## Verify without fitting

`python scripts/verify_repository.py` checks hashes, notebook/source validity,
the 33-feature whitelist, row counts, inner/outer block separation, saved metrics,
SHAP array consistency/additivity, figure dimensions, and citation metadata.
It writes a fresh report under results/checks/, leaving frozen results intact.

`python scripts/reproduce_validation.py --mode artifacts` reconstructs fold
assignments and checks saved RF metrics. It seeds a working copy of the supplied
completion caches under `results/new_runs/validation/` and writes there, preserving
all frozen artifacts in `results/reproducibility/`. The pinned Python/package versions and
input hashes must match the recorded configuration. A mismatch is an intentional
guard against mixing environments, not an instruction to overwrite frozen results.

## Regenerate the figures

`python scripts/make_paper_figures.py` builds all eight main and five supplementary
figures from supplied processed inputs, caches and summary tables, writing only
to figures/generated/. It saves 600-dpi PNGs, vector PDFs, plot-data tables and an
input/output manifest. The checked submission images are in figures/paper/.
Different font-rendering libraries can change pixels/PDF bytes across platforms;
the numerical content and dimensions are the reproducibility target.

Reporting-seasonality figures use saved annual/source/monthly summaries. They do
not rerun extraction from the omitted incident database. Global SHAP is read from
the cache; fold-summary rank agreement is calculated from the five importance
columns. No model fitting or new SHAP estimation occurs during figure generation.

## Refit models deliberately

`python scripts/reproduce_validation.py --mode all --workers 4` uses verified
completion caches or reconstructs the controlled RF baselines and nested XGBoost.
When fitting is necessary, XGBoost has 1,280 inner candidate fits and five refits.

`python scripts/reproduce_nested_rf.py --workers 4` reproduces the notebook's
12-candidate nested RF search from frozen outer/inner memberships, preserving
the estimator seed and comparing the results with saved probabilities/metrics.
It writes to a separate new_runs directory and does not replace the checkpoint.
This additional standalone entry point was derived from notebook cells 126–127;
the full expensive RF refit is not a prerequisite for verifying the supplied data.

`python scripts/prepare_shap.py` evaluates the saved RF, without refitting it,
and writes its independent outputs to `results/new_runs/shap/`.

The historical notebook retains its scientific source but has cleared outputs
and execution counts. Its raw-input stages need acquisition files listed in
data_sources.md and are not the repository's Run All workflow. A current processed
checkpoint is provided instead of reconstructing undocumented upstream exports.
