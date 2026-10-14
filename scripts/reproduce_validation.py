"""Reproduce the original validation design without overwriting historical files.

Run with the geoai environment (scikit-learn 1.8.0). The missing historical
nested-XGBoost implementation is reconstructed from the preserved RF design,
the notebook's 64-candidate XGBoost grid, and the saved comparison artifacts.
Its outputs are marked verified only if those independent checks agree.

Modes can be run independently; completed folds are reused only when the
configuration, input hashes, versions, and output hashes agree.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold, StratifiedKFold
from xgboost import XGBClassifier

from project import FEATURES, REPRO as FROZEN_REPRO, ROOT, load_model_data

REPRO = ROOT / "results/new_runs/validation"


SCRIPT_VERSION = 1
XGB_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [3, 5],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 5],
}
METRICS = ["roc_auc", "accuracy", "precision", "recall", "f1"]
EXPECTED_TEST_SIZES = [1462, 1423, 1474, 1447, 1423]
BASELINE_RANDOM_ROUNDED = [0.755, 0.794, 0.753, 0.758, 0.777]
BASELINE_GROUPED_ROUNDED = [0.688, 0.639, 0.694, 0.663, 0.680]
SUMMARY_COLUMNS = ["model", "validation", "mean_auc", "sd_auc", "pooled_oof_auc", "provenance", "verified"]


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, data):
    """Replace only a new reproducibility artifact, using an atomic rename."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def csv_write(path, frame):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def scores(y, probability, prediction=None):
    if prediction is None:
        prediction = (np.asarray(probability) >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction)),
        "recall": float(recall_score(y, prediction)),
        "f1": float(f1_score(y, prediction)),
    }


def frame_metrics_equal(left, right, columns=METRICS, tolerance=1e-12):
    return bool(np.allclose(left[columns].to_numpy(float), right[columns].to_numpy(float), rtol=0, atol=tolerance))


def configuration(workers):
    sources = [
        "data/missouri_wildfire_model_encoded.csv",
        "results/rf_spatial_cv_results.csv", "results/xgb_spatial_cv_results.csv",
        "results/rf_nested_spatial_cv_results.csv", "results/rf_nested_best_params.csv",
        "results/nested_rf_xgb_fold_results.csv",
    ]
    config = {
        "script_version": SCRIPT_VERSION,
        "source_sha256": {name: sha256(ROOT / name) for name in sources},
        "feature_order": FEATURES,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "outer_cv": {"class": "StratifiedGroupKFold", "n_splits": 5, "shuffle": True, "random_state": 42},
        "inner_cv": {"class": "StratifiedGroupKFold", "n_splits": 4, "shuffle": True, "seeds": [43, 44, 45, 46, 47]},
        "random_baseline_cv": {"class": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 42},
        "grouped_baseline_cv": {"class": "GroupKFold", "n_splits": 5},
        "baseline_rf_explicit_parameters": {"n_estimators": 500, "random_state": 42, "n_jobs": -1},
        "xgb_explicit_parameters": {"objective": "binary:logistic", "eval_metric": "logloss", "random_state": 42},
        "xgb_parameter_grid": XGB_GRID,
        "scoring": "roc_auc", "threshold": 0.5, "sd_ddof": 1,
        "execution": {"workers": workers, "grid_jobs": workers, "xgb_threads_per_fit": 1,
                      "parallelism_note": "The original used n_jobs=-1 at both levels. Only execution parallelism is bounded here; statistical settings are unchanged."},
    }
    config["fingerprint"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return config


def initialize(config):
    if not REPRO.exists():
        shutil.copytree(FROZEN_REPRO, REPRO)
    REPRO.mkdir(parents=True, exist_ok=True)
    path = REPRO / "validation_config.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["fingerprint"] != config["fingerprint"]:
            raise RuntimeError("Reproduction configuration/input/version mismatch: preserve or separately archive existing new artifacts before another configuration.")
    else:
        json_write(path, config)


def completion_path(name):
    return REPRO / f"{name}.complete.json"


def completed(name, config):
    path = completion_path(name)
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result["fingerprint"] != config["fingerprint"]:
        raise RuntimeError(f"Cannot reuse {name}: configuration changed.")
    for filename, expected in result["output_sha256"].items():
        output = REPRO / filename
        if not output.exists() or sha256(output) != expected:
            raise RuntimeError(f"Cannot reuse {name}: output changed or missing: {filename}")
    return result


def save_completion(name, config, outputs, detail):
    result = {"fingerprint": config["fingerprint"],
              "output_sha256": {p.name: sha256(p) for p in outputs}, **detail}
    json_write(completion_path(name), result)
    return result


def fold_artifacts(frame, X, y, groups):
    outer_splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups))
    sizes = [len(te) for _, te in outer_splits]
    if sizes != EXPECTED_TEST_SIZES:
        raise RuntimeError(f"Outer fold sizes differ from saved notebook: {sizes} != {EXPECTED_TEST_SIZES}")
    assignments = frame[["grid_id", "block_id"]].copy()
    assignments.insert(0, "row_index", np.arange(len(frame)))
    assignments["outer_fold"] = 0
    inner_tables, inner_splits, rf_checks = [], [], []
    reference = read_csv(ROOT / "results/rf_nested_spatial_cv_results.csv")
    for fold, (tr, te) in enumerate(outer_splits, start=1):
        assert not set(groups.iloc[tr]).intersection(groups.iloc[te])
        assignments.loc[te, "outer_fold"] = fold
        local_splits = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42 + fold).split(X.iloc[tr], y.iloc[tr], groups.iloc[tr]))
        inner_splits.append(local_splits)
        inner_table = frame.iloc[tr][["grid_id", "block_id"]].copy()
        inner_table.insert(0, "row_index", tr)
        inner_table.insert(0, "outer_train_row_position", np.arange(len(tr)))
        inner_table.insert(0, "outer_fold", fold)
        inner_table["inner_validation_fold"] = 0
        for inner_fold, (itr, ite) in enumerate(local_splits, start=1):
            assert not set(groups.iloc[tr[itr]]).intersection(groups.iloc[tr[ite]])
            inner_table.loc[tr[ite], "inner_validation_fold"] = inner_fold
        assert (inner_table["inner_validation_fold"] > 0).all()
        inner_tables.append(inner_table)
        observed = scores(y.iloc[te], frame.iloc[te]["rf_nested_oof_prob"], frame.iloc[te]["rf_nested_oof_pred"])
        row = {"fold": fold, "test_cells": len(te), **observed}
        row["max_absolute_metric_difference"] = max(abs(observed[m] - float(reference.iloc[fold - 1][m])) for m in METRICS)
        row["verified"] = row["max_absolute_metric_difference"] <= 1e-12
        rf_checks.append(row)
    if not all(r["verified"] for r in rf_checks):
        raise RuntimeError(f"Saved RF OOF metrics do not identify these reconstructed folds: {rf_checks}")
    assert (assignments["outer_fold"] > 0).all()
    csv_write(REPRO / "outer_fold_assignments.csv", assignments)
    csv_write(REPRO / "inner_fold_assignments.csv", pd.concat(inner_tables, ignore_index=True))
    csv_write(REPRO / "rf_saved_oof_fold_verification.csv", pd.DataFrame(rf_checks))
    log("Outer assignments reproduce all five saved RF nested fold metrics; inner fold assignments exported.")
    return outer_splits, inner_splits


def baseline(name, splitter, expected, expected_mean, expected_sd, frame, X, y, groups, config):
    previous = completed(name, config)
    if previous:
        log(f"Reusing verified completed artifacts for {name}.")
        return previous
    predictions = np.full(len(X), np.nan)
    classes = np.full(len(X), -1, dtype=int)
    fold_ids = np.zeros(len(X), dtype=int)
    rows = []
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups if isinstance(splitter, GroupKFold) else None), 1):
        log(f"{name}: fitting original 500-tree RF, fold {fold}/5.")
        model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
        model.fit(X.iloc[tr], y.iloc[tr])
        prob = model.predict_proba(X.iloc[te])[:, 1]
        pred = model.predict(X.iloc[te])
        predictions[te], classes[te], fold_ids[te] = prob, pred, fold
        rows.append({"fold": fold, **scores(y.iloc[te], prob, pred)})
    result = pd.DataFrame(rows)
    oof = frame[["grid_id", "block_id", "fire_presence"]].copy()
    oof.insert(0, "row_index", np.arange(len(frame)))
    oof["fold"], oof["oof_probability"], oof["oof_prediction"] = fold_ids, predictions, classes
    csv_path, oof_path = REPRO / f"{name}_fold_results.csv", REPRO / f"{name}_oof_predictions.csv"
    csv_write(csv_path, result)
    csv_write(oof_path, oof)
    mean, sd = float(result.roc_auc.mean()), float(result.roc_auc.std(ddof=1))
    historical_match = list(result.roc_auc.round(3)) == expected and round(mean, 3) == expected_mean
    if expected_sd is not None:
        historical_match = historical_match and round(sd, 3) == expected_sd
    detail = {"verified": bool(historical_match), "mean_auc": mean, "sd_auc": sd,
              "pooled_oof_auc": float(roc_auc_score(y, predictions)),
              "historical_comparison": "Rounded notebook fold AUCs and mean" + (" and SD" if expected_sd is not None else "; original full-precision random SD was not saved"),
              "historical_rounding_expected": {"fold_auc": expected, "mean_auc": expected_mean, "sd_auc": expected_sd},
              "provenance": "Reproduced original baseline from preserved 33-feature checkpoint; exact values belong to this reproduction."}
    save_completion(name, config, [csv_path, oof_path], detail)
    log(f"{name}: AUC {mean:.9f} ± {sd:.9f}; historical rounded-score match={historical_match}.")
    if not historical_match:
        raise RuntimeError(f"{name} differs from saved historical rounded metrics; outputs preserved but not promoted.")
    return detail


def nested_xgb(frame, X, y, outer_splits, inner_splits, config, workers):
    historical = read_csv(ROOT / "results/nested_rf_xgb_fold_results.csv")
    historical = historical[historical.Model == "XGBoost"].sort_values("Fold").reset_index(drop=True)
    output_probability = np.full(len(X), np.nan)
    output_prediction = np.full(len(X), -1, dtype=int)
    output_fold = np.zeros(len(X), dtype=int)
    fold_rows, comparisons = [], []
    for fold, ((tr, te), inner) in enumerate(zip(outer_splits, inner_splits), 1):
        name = f"xgb_nested_fold_{fold}"
        state = completed(name, config)
        prediction_path = REPRO / f"{name}_predictions.csv"
        if state is None:
            log(f"Nested XGBoost fold {fold}/5: 64 candidates × 4 inner spatial folds; {workers} concurrent single-thread fits.")
            started = time.monotonic()
            base = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=1)
            search = GridSearchCV(base, XGB_GRID, scoring="roc_auc", cv=inner,
                                  n_jobs=workers, refit=True, error_score="raise", return_train_score=False)
            search.fit(X.iloc[tr], y.iloc[tr])
            probability = search.best_estimator_.predict_proba(X.iloc[te])[:, 1]
            prediction = (probability >= 0.5).astype(int)
            metrics = scores(y.iloc[te], probability, prediction)
            candidate_results = pd.DataFrame(search.cv_results_)
            candidate_results["params"] = candidate_results["params"].map(lambda p: json.dumps(p, sort_keys=True))
            candidate_path = REPRO / f"{name}_cv_results.csv"
            csv_write(candidate_path, candidate_results)
            pred_frame = frame.iloc[te][["grid_id", "block_id", "fire_presence"]].copy()
            pred_frame.insert(0, "row_index", te)
            pred_frame["outer_fold"] = fold
            pred_frame["oof_probability"], pred_frame["oof_prediction"] = probability, prediction
            csv_write(prediction_path, pred_frame)
            state = save_completion(name, config, [candidate_path, prediction_path], {
                "fold": fold, "metrics": metrics, "best_params": search.best_params_,
                "inner_best_auc": float(search.best_score_), "best_index": int(search.best_index_),
                "elapsed_seconds": float(time.monotonic() - started),
                "provenance": "New full nested reconstruction; historical generating code was not preserved."})
            log(f"Nested XGBoost fold {fold}: inner AUC={state['inner_best_auc']:.12f}, outer AUC={metrics['roc_auc']:.12f}; {state['elapsed_seconds']:.1f}s.")
        else:
            log(f"Reusing full nested XGBoost fold {fold}, verified input/output hashes.")
        pred_frame = read_csv(prediction_path)
        if not np.array_equal(pred_frame.row_index.to_numpy(), te):
            raise RuntimeError(f"Nested XGBoost fold {fold} saved row membership differs.")
        output_probability[te] = pred_frame.oof_probability
        output_prediction[te] = pred_frame.oof_prediction
        output_fold[te] = fold
        old = historical.iloc[fold - 1]
        expected_params = ast.literal_eval(old.Best_params)
        metric_map = {"roc_auc": "Outer_AUC", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1"}
        differences = {m: abs(state["metrics"][m] - float(old[source])) for m, source in metric_map.items()}
        inner_difference = abs(state["inner_best_auc"] - float(old.Inner_AUC))
        matches = max(differences.values()) <= 1e-12 and inner_difference <= 1e-12 and state["best_params"] == expected_params
        comparisons.append({"fold": fold, "max_absolute_metric_difference": max(differences.values()),
                            "inner_auc_difference": inner_difference, "best_params_match": state["best_params"] == expected_params,
                            "verified": bool(matches)})
        fold_rows.append({"fold": fold, "inner_best_auc": state["inner_best_auc"], **state["metrics"],
                          "best_params": json.dumps(state["best_params"], sort_keys=True)})
        csv_write(REPRO / "xgb_nested_historical_comparison.csv", pd.DataFrame(comparisons))
        if not matches:
            # Partial candidate/prediction evidence stays available for diagnosis.
            raise RuntimeError(f"Nested XGBoost fold {fold} differs from historical records: {comparisons[-1]}. No promotion.")
    fold_results = pd.DataFrame(fold_rows)
    oof = frame[["grid_id", "block_id", "fire_presence"]].copy()
    oof.insert(0, "row_index", np.arange(len(frame)))
    oof["outer_fold"], oof["oof_probability"], oof["oof_prediction"] = output_fold, output_probability, output_prediction
    fold_path, oof_path = REPRO / "xgb_nested_fold_results.csv", REPRO / "xgb_nested_oof_predictions.csv"
    csv_write(fold_path, fold_results)
    csv_write(oof_path, oof)
    pooled = float(roc_auc_score(y, output_probability))
    pooled_matches = round(pooled, 3) == 0.645
    detail = {"verified": bool(all(c["verified"] for c in comparisons) and pooled_matches),
              "mean_auc": float(fold_results.roc_auc.mean()), "sd_auc": float(fold_results.roc_auc.std(ddof=1)),
              "pooled_oof_auc": pooled, "pooled_auc_rounds_to_requested_0645": pooled_matches,
              "historical_fold_metrics_inner_scores_parameters_match": all(c["verified"] for c in comparisons),
              "provenance": "Full nested reconstruction on explicitly shared outer/inner folds; matches historical fold metrics, inner scores, and winners. Historical generating code was absent."}
    save_completion("xgb_nested", config, [fold_path, oof_path, REPRO / "xgb_nested_historical_comparison.csv"], detail)
    log(f"Nested XGBoost pooled AUC={pooled:.12f}; rounds to requested .645={pooled_matches}.")
    if not detail["verified"]:
        raise RuntimeError("Reconstructed nested XGBoost pooled AUC does not round to .645; outputs preserved but not promoted.")
    return detail


def summary_row(model, validation, result, provenance=None):
    return {"model": model, "validation": validation, "mean_auc": result.get("mean_auc"),
            "sd_auc": result.get("sd_auc"), "pooled_oof_auc": result.get("pooled_oof_auc"),
            "provenance": provenance or result.get("provenance", ""), "verified": bool(result.get("verified", False))}


def summarize(frame, y, config, failure=None):
    rows, details, metric_rows = [], {}, []
    assignments = read_csv(REPRO / "outer_fold_assignments.csv")

    def append_metrics(model, validation, source, record):
        row = {"model": model, "validation": validation}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(source[metric].mean()) if source is not None else None
            row[f"{metric}_sd"] = float(source[metric].std(ddof=1)) if source is not None else None
        row.update({"pooled_oof_auc": record.get("pooled_oof_auc"),
                    "provenance": record.get("provenance", ""),
                    "verified": bool(record.get("verified", False))})
        metric_rows.append(row)

    for name, model, validation in [
        ("rf_random", "Random Forest", "Random 5-fold CV"),
        ("rf_grouped_baseline", "Random Forest", "Grouped spatial 5-fold CV"),
    ]:
        record = completed(name, config)
        if record is None:
            record = {"verified": False, "provenance": "Original notebook has rounded results only; full baseline reproduction pending."}
        details[name] = record
        rows.append(summary_row(model, validation, record))
        source = read_csv(REPRO / f"{name}_fold_results.csv") if record.get("verified") else None
        append_metrics(model, validation, source, record)
    for name, model, validation, csvname, probcol, predcol, expected_pooled in [
        ("rf_tuned", "Random Forest", "Tuned spatial 5-fold CV", "rf_spatial_cv_results.csv", "rf_oof_prob", "rf_oof_pred", 0.661),
        ("xgb_tuned", "XGBoost", "Tuned spatial 5-fold CV", "xgb_spatial_cv_results.csv", "xgb_oof_prob", "xgb_oof_pred", 0.647),
        ("rf_nested", "Random Forest", "Nested spatial 5-fold CV", "rf_nested_spatial_cv_results.csv", "rf_nested_oof_prob", "rf_nested_oof_pred", 0.659),
    ]:
        source = read_csv(ROOT / "results" / csvname)
        reconstructed = pd.DataFrame([
            scores(y.loc[assignments.outer_fold == fold],
                   frame.loc[assignments.outer_fold == fold, probcol],
                   frame.loc[assignments.outer_fold == fold, predcol])
            for fold in range(1, 6)
        ])
        metric_match = frame_metrics_equal(reconstructed, source)
        pooled = float(roc_auc_score(y, frame[probcol]))
        record = {"mean_auc": float(source.roc_auc.mean()), "sd_auc": float(source.roc_auc.std(ddof=1)),
                  "pooled_oof_auc": pooled, "verified": bool(metric_match and round(pooled, 3) == expected_pooled),
                  "fold_metrics_match_saved_oof": metric_match,
                  "pooled_auc_matches_notebook_rounding": round(pooled, 3) == expected_pooled,
                  "provenance": "Historical full-precision fold metrics and saved per-cell OOF probabilities; original notebook implementation preserved."}
        details[name] = record
        rows.append(summary_row(model, validation, record))
        append_metrics(model, validation, source, record)
    result = completed("xgb_nested", config)
    if result is None:
        historic = read_csv(ROOT / "results/nested_rf_xgb_fold_results.csv")
        historic = historic[historic.Model == "XGBoost"]
        result = {"mean_auc": float(historic.Outer_AUC.mean()), "sd_auc": float(historic.Outer_AUC.std(ddof=1)),
                  "pooled_oof_auc": None, "verified": False,
                  "provenance": "Historical fold metrics only; nested generating code and per-cell probabilities were absent. Full reconstruction pending."}
    details["xgb_nested"] = result
    rows.append(summary_row("XGBoost", "Nested spatial 5-fold CV", result))
    if result.get("verified"):
        nested_source = read_csv(REPRO / "xgb_nested_fold_results.csv")
    else:
        historic = read_csv(ROOT / "results/nested_rf_xgb_fold_results.csv")
        nested_source = historic[historic.Model == "XGBoost"].rename(columns={
            "Outer_AUC": "roc_auc", "Accuracy": "accuracy", "Precision": "precision", "Recall": "recall", "F1": "f1"})
    append_metrics("XGBoost", "Nested spatial 5-fold CV", nested_source, result)
    csv_write(REPRO / "model_evaluation_summary.csv", pd.DataFrame(rows, columns=SUMMARY_COLUMNS))
    csv_write(REPRO / "model_evaluation_metrics.csv", pd.DataFrame(metric_rows))
    report = {"fingerprint": config["fingerprint"], "status": "verified" if all(r["verified"] for r in rows) else "incomplete",
              "historical_files_overwritten": False, "historical_nested_xgb_source_present": False,
              "fold_assignments_verified_against_saved_rf_metrics": True,
              "summary_schema": SUMMARY_COLUMNS, "analyses": details,
              "validation_error": failure, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if failure:
        report["status"] = "mismatch_or_failure"
    json_write(REPRO / "validation_report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["artifacts", "baselines", "nested-xgb", "all"], default="artifacts")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if sklearn.__version__ != "1.8.0":
        raise RuntimeError(f"Expected original scikit-learn 1.8.0; found {sklearn.__version__}.")
    # Preserve original RF n_jobs=-1 while bounding joblib's available CPU count.
    os.environ["LOKY_MAX_CPU_COUNT"] = str(args.workers)
    config = configuration(args.workers)
    initialize(config)
    frame, X, y, groups = load_model_data()
    outer, inner = fold_artifacts(frame, X, y, groups)
    summarize(frame, y, config)
    try:
        if args.mode in {"baselines", "all"}:
            baseline("rf_random", StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                     BASELINE_RANDOM_ROUNDED, 0.768, None, frame, X, y, groups, config)
            summarize(frame, y, config)
            baseline("rf_grouped_baseline", GroupKFold(n_splits=5), BASELINE_GROUPED_ROUNDED,
                     0.673, 0.022, frame, X, y, groups, config)
            summarize(frame, y, config)
        if args.mode in {"nested-xgb", "all"}:
            nested_xgb(frame, X, y, outer, inner, config, args.workers)
    except Exception as exc:
        summarize(frame, y, config, failure=f"{type(exc).__name__}: {exc}")
        raise
    summarize(frame, y, config)
    log(f"Mode {args.mode} complete. Historical artifacts preserved; see {REPRO / 'validation_report.json'}.")


if __name__ == "__main__":
    main()
