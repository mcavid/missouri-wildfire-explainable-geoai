# Figure mapping

All paper images are in figures/paper/. Generated equivalents go to figures/generated/.

| Figure | Content | Generator |
|---|---|---|
| 1 | Study area and spatial folds | review_spatial.study_area_coordinates |
| 2 | Workflow | make_publication_figures.figure2 |
| 3 | Validation comparison | make_publication_figures.figure3 |
| 4 | Full-data SHAP | make_publication_figures.figure4 |
| 5 | Relative susceptibility map | review_spatial.susceptibility_coordinates |
| 6 | Nested OOF class prevalence | make_publication_figures.figure6 |
| 7 | Nested prediction errors | review_diagnostics.spatial_errors |
| 8 | Spatial full-data SHAP | review_spatial.spatial_shap |
| S1 | Reporting/seasonality summaries | review_supporting.s3_reporting (saved-table adapter) |
| S2 | Block-size and omission summaries | review_supporting.s1_robustness |
| S3 | Nested ROC/precision–recall | review_diagnostics.roc_pr |
| S4 | Population/road dependence | review_spatial.population_road_dependence |
| S5 | Fold-summary rank agreement | review_supporting.s2_shap_stability (banner removed) |

The S5 wrapper retains the original ranks, correlations and ordering, removes the
editorial verification banner, and labels the recorded columns as saved folds.
The S1 adapter checks saved count arithmetic; raw incident-query verification is
recorded in the original study history and is not claimed to rerun here.
