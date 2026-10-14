"""Recreate the notebook's SHAP values from the preserved fitted RF; never fit.

The original call is deliberately retained: TreeExplainer(model), followed by
shap_values(X), interpreting output index 1. No background or perturbation
argument is added. This script records the settings resolved by the installed
SHAP version separately from the arguments explicitly used in the notebook.
"""
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import shap

from project import FEATURES, ROOT, load_model_data

REPRO = ROOT / "results/new_runs/shap"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    REPRO.mkdir(parents=True, exist_ok=True)
    frame, X, y, _ = load_model_data()
    model_path = ROOT / "models/random_forest_wildfire_5km.joblib"
    model = joblib.load(model_path)
    if model.feature_names_in_.tolist() != FEATURES:
        raise ValueError("Saved RF predictor names/order differ from original X.")
    if model.classes_.tolist() != [0, 1]:
        raise ValueError("Saved RF classes are not [0, 1].")
    expected_params = dict(n_estimators=500, max_depth=10, min_samples_leaf=1,
                           max_features="sqrt", class_weight="balanced",
                           random_state=42, n_jobs=-1)
    for name, value in expected_params.items():
        if model.get_params()[name] != value:
            raise ValueError(f"Unexpected final RF {name}: {model.get_params()[name]}")
    print("Computing original-method SHAP for 7,229 cells / 33 predictors.", flush=True)
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(X))
    if values.shape != (7229, 33, 2):
        raise ValueError(f"Unexpected SHAP array shape: {values.shape}")
    fire = values[:, :, 1]
    importance = np.abs(fire).mean(axis=0)
    original_path = ROOT / "results/rf_shap_importance.csv"
    original = pd.read_csv(original_path)
    feature_column = next(c for c in original.columns if "feature" in c.lower())
    value_column = next(c for c in original.columns if c != feature_column)
    comparison = pd.DataFrame({"feature": FEATURES, "recomputed_mean_abs_shap": importance})
    comparison["original_mean_abs_shap"] = comparison.feature.map(
        original.set_index(feature_column)[value_column])
    comparison["absolute_difference"] = np.abs(
        comparison.recomputed_mean_abs_shap - comparison.original_mean_abs_shap)
    if comparison.original_mean_abs_shap.isna().any():
        raise ValueError("Missing original SHAP importance entry.")
    max_difference = float(comparison.absolute_difference.max())
    if not np.allclose(comparison.recomputed_mean_abs_shap,
                       comparison.original_mean_abs_shap, rtol=1e-7, atol=1e-10):
        raise ValueError(f"Recomputed SHAP differs from manuscript importance: {max_difference}")
    base = np.asarray(explainer.expected_value)
    probability = model.predict_proba(X)[:, 1]
    reconstructed = base[1] + fire.sum(axis=1)
    additivity_error = float(np.max(np.abs(probability - reconstructed)))
    if not np.allclose(probability, reconstructed, rtol=1e-6, atol=1e-7):
        raise ValueError(f"Class-1 SHAP reconstruction differs: {additivity_error}")
    output_path = REPRO / "rf_shap_values.npz"
    np.savez_compressed(output_path, shap_values=values, shap_class1=fire,
                        X=X.to_numpy(), grid_id=frame.grid_id.to_numpy(),
                        feature_names=np.asarray(FEATURES), expected_value=base,
                        class1_prediction=probability)
    comparison.to_csv(REPRO / "rf_shap_importance_verification.csv", index=False)
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_notebook": "etl.ipynb (historical source; execution outputs cleared)",
        "source_cells_1_based": [114, 115],
        "method": "shap.TreeExplainer(model).shap_values(X)[:, :, 1]",
        "model_fitted_here": False, "rows": len(X), "features": FEATURES,
        "classes": model.classes_.tolist(), "interpreted_class": 1,
        "class_definition": "At least one recorded wildfire in the grid cell",
        "constructor_arguments": {"model": "preserved final fitted RF",
                                   "feature_perturbation": "not explicitly specified",
                                   "data_background": "not explicitly specified",
                                   "model_output": "not explicitly specified"},
        "resolved_explainer": {
            "class": type(explainer).__name__,
            "feature_perturbation": explainer.feature_perturbation,
            "model_output": explainer.model.model_output,
            "background_present": explainer.data is not None,
            "expected_value": base.tolist()},
        "output_scale_verification": "Base value plus class-1 SHAP sum matches saved RF class-1 predict_proba output; score is not asserted calibrated.",
        "maximum_additivity_error": additivity_error,
        "maximum_importance_difference": max_difference,
        "model_parameters": model.get_params(),
        "input_sha256": {
            str(p.relative_to(ROOT)): sha256(p) for p in [model_path,
                ROOT / "data/missouri_wildfire_model_encoded.csv", original_path]},
        "output_sha256": sha256(output_path),
        "environment": {"python": platform.python_version(), **{
            p: importlib.metadata.version(p) for p in
            ["shap", "scikit-learn", "numpy", "pandas", "joblib"]}},
    }
    (REPRO / "rf_shap_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {output_path}; max importance difference={max_difference:.3g}; "
          f"max additivity error={additivity_error:.3g}", flush=True)


if __name__ == "__main__":
    main()
