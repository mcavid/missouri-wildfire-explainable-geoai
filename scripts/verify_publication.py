"""Audit the cleaned publication package without fitting or rendering anything.

Run after the six publication figures have been generated. This script only
reads package inputs and writes results/reproducibility/publication_verification.json.
It checks preserved originals, notebook structure and Python syntax, data/folds,
validation artifacts, the SHAP cache, and the publication PDF/PNG exports.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

from project import FEATURES, PUBLICATION, REPRO, ROOT, load_model_data


ARCHIVE = ROOT / "archive/pre_publication_cleanup_2026-10-03"
PORTABLE_INVENTORY = ROOT / "docs/original_analysis_files.csv"
METRICS = ["roc_auc", "accuracy", "precision", "recall", "f1"]
EXPECTED = {
    ("Random Forest", "Random 5-fold CV"): (0.768, 0.018),
    ("Random Forest", "Grouped spatial 5-fold CV"): (0.673, 0.022),
    ("Random Forest", "Tuned spatial 5-fold CV"): (0.663, 0.030),
    ("XGBoost", "Tuned spatial 5-fold CV"): (0.649, 0.025),
    ("Random Forest", "Nested spatial 5-fold CV"): (0.662, 0.031),
    ("XGBoost", "Nested spatial 5-fold CV"): (0.647, 0.032),
}
FOLD_FILES = {
    ("Random Forest", "Random 5-fold CV"): REPRO / "rf_random_fold_results.csv",
    ("Random Forest", "Grouped spatial 5-fold CV"): REPRO / "rf_grouped_baseline_fold_results.csv",
    ("Random Forest", "Tuned spatial 5-fold CV"): ROOT / "results/rf_spatial_cv_results.csv",
    ("XGBoost", "Tuned spatial 5-fold CV"): ROOT / "results/xgb_spatial_cv_results.csv",
    ("Random Forest", "Nested spatial 5-fold CV"): ROOT / "results/rf_nested_spatial_cv_results.csv",
    ("XGBoost", "Nested spatial 5-fold CV"): REPRO / "xgb_nested_fold_results.csv",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip", dtype={"block_id": str})


def read_json(path):
    return json.loads(Path(path).read_text())


def portable_inventory():
    with PORTABLE_INVENTORY.open() as stream:
        rows = list(csv.DictReader(stream))
    normalized = []
    for row in rows:
        relative = Path(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"Portable original inventory must contain analysis-relative paths: {relative}")
        if relative.parts and relative.parts[0] == "analysis":
            relative = Path(*relative.parts[1:])
        normalized.append({"path": relative.as_posix(), "bytes": int(row["bytes"]), "sha256": row["sha256"]})
    return normalized


def critical_original(relative):
    path = Path(relative)
    return (relative in {"data/missouri_wildfire_model_encoded.csv", "data/Missouri_Wildfire_Susceptibility_5km.gpkg"}
            or (path.parts[0] == "models" and path.suffix in {".joblib", ".pkl", ".pickle"})
            or (path.parts[0] == "results" and path.suffix == ".csv"))


def verify_originals():
    if not (ARCHIVE / "inventory_before.json").is_file():
        rows = portable_inventory()
        paths = {item["path"] for item in rows}
        required = {"data/missouri_wildfire_model_encoded.csv", "data/Missouri_Wildfire_Susceptibility_5km.gpkg",
                    "models/random_forest_wildfire_5km.joblib", "models/xgboost_wildfire_5km.joblib"}
        require(required.issubset(paths), "Portable inventory omits a critical original modeling artifact.")
        checked, skipped, optional_omissions, problems = 0, [], [], []
        for item in rows:
            relative = item["path"]
            if relative == "etl.ipynb":
                skipped.append(relative)
                continue
            path = ROOT / relative
            if not path.is_file():
                if critical_original(relative):
                    problems.append({"path": relative, "problem": "critical original missing"})
                else:
                    optional_omissions.append(relative)
                continue
            if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                problems.append({"path": relative, "problem": "original size or SHA-256 changed"})
            checked += 1
        require(not problems, f"Portable original inventory mismatch: {problems}")
        require(len(skipped) == 1, "Portable inventory must identify the intentionally edited etl.ipynb.")
        return {"inventory_mode": "portable_analysis_inventory", "original_files_unchanged": checked,
                "intentionally_edited": skipped, "optional_omissions": optional_omissions,
                "critical_originals_present_and_verified": True, "inventory_file": str(PORTABLE_INVENTORY.relative_to(ROOT)),
                "archived_original_notebook_present": False}
    inventory = read_json(ARCHIVE / "inventory_before.json")
    inventory_root = Path(inventory["project_root"])
    checked, skipped, problems = 0, [], []
    original_notebook_hash = None
    for item in inventory["files"]:
        path = inventory_root / item["path"]
        if path.resolve() == (ROOT / "etl.ipynb").resolve():
            original_notebook_hash = item["sha256"]
            skipped.append(item["path"])
            continue
        if not path.is_file():
            problems.append({"path": item["path"], "problem": "missing"})
        elif path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            problems.append({"path": item["path"], "problem": "size or SHA-256 changed"})
        checked += 1
    require(not problems, f"Original inventory mismatch: {problems}")
    require(len(skipped) == 1, "The original inventory must identify exactly the intentionally edited analysis/etl.ipynb.")
    archived_notebook = ARCHIVE / "etl.original.ipynb"
    require(sha256(archived_notebook) == original_notebook_hash, "The archived original notebook differs from its pre-cleanup inventory hash.")
    return {"inventory_mode": "complete_local_pre_cleanup_inventory", "original_files_unchanged": checked, "intentionally_edited": skipped,
            "archived_original_notebook_verified": True, "inventory_file": str((ARCHIVE / "inventory_before.json").relative_to(ROOT))}


def verify_notebook():
    notebook = nbformat.read(ROOT / "etl.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = 0
    syntax_errors = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        code_cells += 1
        try:
            ast.parse(cell.source, filename=f"etl.ipynb:cell-{index + 1}")
        except SyntaxError as exc:
            syntax_errors.append({"cell_1_based": index + 1, "error": str(exc)})
    require(not syntax_errors, f"Notebook code-cell syntax errors: {syntax_errors}")
    provenance = notebook.metadata.get("cleanup_provenance", {})
    if "original_notebook_sha256" in provenance:
        original_path = ARCHIVE / "etl.original.ipynb"
        if original_path.is_file():
            expected_hash = sha256(original_path)
        else:
            original_entry = [item for item in portable_inventory() if item["path"] == "etl.ipynb"]
            require(len(original_entry) == 1, "Portable inventory does not identify the original notebook.")
            expected_hash = original_entry[0]["sha256"]
        require(provenance["original_notebook_sha256"] == expected_hash,
                "Notebook cleanup provenance does not match the original notebook hash.")
    script_errors, script_count = [], 0
    for script in sorted((ROOT / "scripts").rglob("*.py")):
        script_count += 1
        try:
            ast.parse(script.read_text(), filename=str(script.relative_to(ROOT)))
        except SyntaxError as exc:
            script_errors.append({"script": str(script.relative_to(ROOT)), "error": str(exc)})
    require(not script_errors, f"Package script syntax errors: {script_errors}")
    return {"nbformat_valid": True, "cells": len(notebook.cells), "code_cells_parsed": code_cells,
            "scripts_parsed": script_count, "notebook_executed": False, "sha256": sha256(ROOT / "etl.ipynb")}


def verify_data_and_folds():
    frame, X, y, groups = load_model_data()
    require(X.shape == (7229, 33) and X.columns.tolist() == FEATURES, "The modeling matrix is not the original ordered 33 predictors.")
    require(sorted(y.unique().tolist()) == [0, 1] and int(y.sum()) == 2988, "The wildfire class labels/count changed.")
    excluded = {"grid_id", "block_id", "fire_count", "fire_presence"}
    require(not excluded.intersection(X.columns) and not any("oof" in c.lower() for c in X), "Identifiers, outcomes, or OOF predictions appear in X.")
    for prefix in ["lc_", "fuel_"]:
        matrix = X.loc[:, X.columns.str.startswith(prefix)].to_numpy()
        require(np.isin(matrix, [0, 1]).all() and np.all(matrix.sum(axis=1) == 1), f"{prefix} categorical encoding is not one-hot.")
    outer = read_csv(REPRO / "outer_fold_assignments.csv").sort_values("row_index").reset_index(drop=True)
    require(len(outer) == len(frame) and np.array_equal(outer.row_index, np.arange(len(frame))), "Outer assignments do not cover each row exactly once.")
    require(np.array_equal(outer.grid_id, frame.grid_id) and np.array_equal(outer.block_id, groups), "Outer assignment identities differ from the modeling checkpoint.")
    require(sorted(outer.outer_fold.unique().tolist()) == [1, 2, 3, 4, 5], "Expected five outer folds.")
    require(int(outer.groupby("block_id").outer_fold.nunique().max()) == 1, "A spatial block crosses outer folds.")
    counts = outer.groupby("outer_fold").size().tolist()
    require(counts == [1462, 1423, 1474, 1447, 1423], f"Outer test sizes differ: {counts}")
    inner = read_csv(REPRO / "inner_fold_assignments.csv")
    require(len(inner) == 28916, "Expected 28,916 outer-training row memberships in inner folds.")
    require(sorted(inner.outer_fold.unique().tolist()) == [1, 2, 3, 4, 5], "Inner assignments do not cover all five outer training sets.")
    inner_sizes = {}
    for fold in range(1, 6):
        part = inner[inner.outer_fold == fold].sort_values("outer_train_row_position")
        tr = np.flatnonzero(outer.outer_fold.to_numpy() != fold)
        require(np.array_equal(part.outer_train_row_position, np.arange(len(tr))), f"Inner row positions differ for outer fold {fold}.")
        require(np.array_equal(part.row_index, tr), f"Inner assignments omit, duplicate, or include outer-test rows in fold {fold}.")
        require(np.array_equal(part.grid_id, frame.iloc[tr].grid_id) and np.array_equal(part.block_id, groups.iloc[tr]), f"Inner identities differ in outer fold {fold}.")
        require(sorted(part.inner_validation_fold.unique().tolist()) == [1, 2, 3, 4], f"Expected four inner folds inside outer fold {fold}.")
        require(int(part.groupby("block_id").inner_validation_fold.nunique().max()) == 1, f"A spatial block crosses inner folds inside outer fold {fold}.")
        inner_sizes[str(fold)] = part.groupby("inner_validation_fold").size().tolist()
    rf = read_csv(REPRO / "rf_saved_oof_fold_verification.csv")
    require(len(rf) == 5 and (rf.max_absolute_metric_difference <= 1e-12).all() and rf.verified.eq(True).all(), "Reconstructed folds were not verified against historical nested RF OOF scores.")
    return {"rows": len(frame), "predictors": list(X.columns), "classes": [0, 1], "positive_rows": int(y.sum()),
            "spatial_blocks": int(groups.nunique()), "outer_test_sizes": counts, "inner_validation_sizes": inner_sizes,
            "outer_and_inner_block_separation": True}


def verify_validation():
    report = read_json(REPRO / "validation_report.json")
    config = read_json(REPRO / "validation_config.json")
    require(report["status"] == "verified" and report.get("validation_error") is None, "Validation report has unresolved errors or pending analyses.")
    require(report["fingerprint"] == config["fingerprint"], "Validation report/configuration fingerprints differ.")
    payload = {k: v for k, v in config.items() if k != "fingerprint"}
    require(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() == config["fingerprint"], "The recorded validation configuration fingerprint is invalid.")
    for relative, expected in config["source_sha256"].items():
        require(sha256(ROOT / relative) == expected, f"Validation source changed: {relative}")
    completions = sorted(REPRO.glob("*.complete.json"))
    require(len(completions) >= 8, "Expected baseline and nested-XGBoost completion records.")
    output_hashes = 0
    for path in completions:
        state = read_json(path)
        require(state["fingerprint"] == config["fingerprint"], f"Completion configuration differs: {path.name}")
        for filename, expected in state["output_sha256"].items():
            require(sha256(REPRO / filename) == expected, f"Completed output changed: {filename}")
            output_hashes += 1
    summary = read_csv(REPRO / "model_evaluation_summary.csv")
    metrics = read_csv(REPRO / "model_evaluation_metrics.csv")
    key_columns = ["model", "validation"]
    require(len(summary) == 6 and len(metrics) == 6, "Expected six analysis rows in both publication summaries.")
    require(not summary.duplicated(key_columns).any() and not metrics.duplicated(key_columns).any(), "Summary analyses are duplicated.")
    require(summary.verified.eq(True).all() and metrics.verified.eq(True).all(), "Publication summaries contain unverified analyses.")
    require(set(zip(summary.model, summary.validation)) == set(EXPECTED), "Publication summary analysis labels differ from the six verified contracts.")
    indexed = metrics.set_index(key_columns)
    for row in summary.itertuples(index=False):
        key = (row.model, row.validation)
        expected_mean, expected_sd = EXPECTED[key]
        require(round(row.mean_auc, 3) == expected_mean and round(row.sd_auc, 3) == expected_sd, f"Reported AUC rounding differs for {key}.")
        folds = read_csv(FOLD_FILES[key])
        require(len(folds) == 5, f"Expected five fold results for {key}.")
        metric_row = indexed.loc[key]
        for metric in METRICS:
            expected_values = [float(folds[metric].mean()), float(folds[metric].std(ddof=1))]
            actual_values = [float(metric_row[f"{metric}_mean"]), float(metric_row[f"{metric}_sd"])]
            require(np.allclose(actual_values, expected_values, rtol=0, atol=1e-12), f"Mean/sample SD differs from fold results for {key}: {metric}.")
        require(np.allclose([row.mean_auc, row.sd_auc], [metric_row.roc_auc_mean, metric_row.roc_auc_sd], rtol=0, atol=1e-12), f"The two summary tables disagree for {key}.")
        require(np.isclose(row.pooled_oof_auc, metric_row.pooled_oof_auc, rtol=0, atol=1e-12), f"Pooled AUC differs between summary tables for {key}.")
    lookup = summary.set_index(key_columns)
    require(round(float(lookup.loc[("Random Forest", "Nested spatial 5-fold CV"), "pooled_oof_auc"]), 3) == 0.659, "Nested RF pooled AUC does not round to .659.")
    require(round(float(lookup.loc[("XGBoost", "Nested spatial 5-fold CV"), "pooled_oof_auc"]), 3) == 0.645, "Nested XGBoost pooled AUC does not round to .645.")
    frame, _, y, _ = load_model_data()
    probability_sources = {
        ("Random Forest", "Random 5-fold CV"): REPRO / "rf_random_oof_predictions.csv",
        ("Random Forest", "Grouped spatial 5-fold CV"): REPRO / "rf_grouped_baseline_oof_predictions.csv",
        ("XGBoost", "Nested spatial 5-fold CV"): REPRO / "xgb_nested_oof_predictions.csv",
    }
    for key, path in probability_sources.items():
        oof = read_csv(path).sort_values("row_index")
        require(np.array_equal(oof.row_index, np.arange(len(frame))) and np.array_equal(oof.grid_id, frame.grid_id), f"OOF row identities differ: {path.name}")
        require(np.array_equal(oof.fire_presence, y), f"OOF labels differ: {path.name}")
        actual = float(roc_auc_score(y, oof.oof_probability))
        require(abs(actual - float(lookup.loc[key, "pooled_oof_auc"])) <= 1e-12, f"Pooled AUC differs from OOF predictions: {key}")
    for key, column in {
        ("Random Forest", "Tuned spatial 5-fold CV"): "rf_oof_prob",
        ("XGBoost", "Tuned spatial 5-fold CV"): "xgb_oof_prob",
        ("Random Forest", "Nested spatial 5-fold CV"): "rf_nested_oof_prob",
    }.items():
        require(abs(roc_auc_score(y, frame[column]) - float(lookup.loc[key, "pooled_oof_auc"])) <= 1e-12, f"Pooled AUC differs from historical predictions: {key}")
    comparison = read_csv(REPRO / "xgb_nested_historical_comparison.csv")
    require(len(comparison) == 5 and comparison.verified.eq(True).all() and comparison.best_params_match.eq(True).all(), "Nested XGBoost does not match historical tuning/results.")
    require((comparison.max_absolute_metric_difference <= 1e-12).all() and (comparison.inner_auc_difference <= 1e-12).all(), "Nested XGBoost historical score discrepancies exceed tolerance.")
    for fold in range(1, 6):
        candidates = read_csv(REPRO / f"xgb_nested_fold_{fold}_cv_results.csv")
        require(len(candidates) == 64 and all(f"split{i}_test_score" in candidates for i in range(4)), f"Incomplete nested XGBoost search evidence for fold {fold}.")
    return {"analyses_verified": 6, "mean_and_sample_sd_match_fold_results": True,
            "pooled_auc_matches_per_cell_predictions": True, "completed_output_hashes_verified": output_hashes,
            "nested_xgb_candidates_per_outer_fold": 64, "inner_folds": 4,
            "nested_xgb_historical_source_absent_reconstruction_explicit": report["historical_nested_xgb_source_present"] is False}


def verify_shap():
    metadata = read_json(REPRO / "rf_shap_metadata.json")
    path = REPRO / "rf_shap_values.npz"
    require(sha256(path) == metadata["output_sha256"], "SHAP cache hash differs from metadata.")
    for relative, expected in metadata["input_sha256"].items():
        require(sha256(ROOT / relative) == expected, f"SHAP input changed: {relative}")
    require(metadata["classes"] == [0, 1] and metadata["interpreted_class"] == 1, "SHAP class interpretation changed.")
    require(metadata["features"] == FEATURES and metadata["rows"] == 7229 and metadata["model_fitted_here"] is False, "SHAP metadata does not describe the preserved model/matrix.")
    frame, X, _, _ = load_model_data()
    with np.load(path, allow_pickle=False) as cache:
        values = cache["shap_values"]
        fire = cache["shap_class1"]
        require(values.shape == (7229, 33, 2) and fire.shape == (7229, 33), "SHAP cache dimensions changed.")
        require(np.array_equal(values[:, :, 1], fire), "SHAP class-1 cache is inconsistent with the complete SHAP array.")
        require(cache["feature_names"].tolist() == FEATURES and np.array_equal(cache["grid_id"], frame.grid_id), "SHAP cache feature/row identities differ.")
        require(np.array_equal(cache["X"], X.to_numpy()), "SHAP cached matrix differs from the original 33 predictors.")
        require(np.isfinite(values).all(), "SHAP values contain nonfinite entries.")
        require(np.array_equal(cache["expected_value"], metadata["resolved_explainer"]["expected_value"]), "SHAP base values differ from metadata.")
        reconstructed = cache["expected_value"][1] + fire.sum(axis=1)
        probability = cache["class1_prediction"]
        require(probability.shape == (7229,) and ((probability >= 0) & (probability <= 1)).all(), "SHAP reference class-1 predictions are invalid.")
        additivity_error = float(np.max(np.abs(probability - reconstructed)))
        require(np.allclose(probability, reconstructed, rtol=1e-6, atol=1e-7), "SHAP class-1 additivity check failed.")
        require(abs(additivity_error - metadata["maximum_additivity_error"]) <= 1e-12, "SHAP additivity maximum differs from metadata.")
        mean_abs = np.abs(fire).mean(axis=0)
    # Match the original SHAP preparation script's CSV parsing when comparing
    # decimal values at approximately machine precision.
    original = pd.read_csv(ROOT / "results/rf_shap_importance.csv")
    feature_column = next(c for c in original.columns if "feature" in c.lower())
    value_column = next(c for c in original.columns if c != feature_column)
    reference = original.set_index(feature_column).loc[FEATURES, value_column].to_numpy()
    importance_difference = float(np.max(np.abs(mean_abs - reference)))
    require(np.allclose(mean_abs, reference, rtol=1e-7, atol=1e-10), "SHAP mean absolute importance differs from historical results.")
    require(abs(importance_difference - metadata["maximum_importance_difference"]) <= 1e-12, "SHAP importance maximum differs from metadata.")
    verification = read_csv(REPRO / "rf_shap_importance_verification.csv").set_index("feature").loc[FEATURES]
    require(np.allclose(verification.recomputed_mean_abs_shap, mean_abs, rtol=0, atol=1e-12), "SHAP verification CSV differs from cached values.")
    require(np.allclose(verification.original_mean_abs_shap, reference, rtol=0, atol=1e-12), "SHAP verification CSV historical reference differs.")
    require(np.allclose(verification.absolute_difference, np.abs(mean_abs - reference), rtol=0, atol=1e-12), "SHAP verification CSV differences are inconsistent.")
    return {"cache_sha256_verified": True, "input_hashes_verified": len(metadata["input_sha256"]),
            "maximum_additivity_error": additivity_error, "maximum_importance_difference": importance_difference,
            "class": 1, "model_fitted_here": False}


def manifest_hashes(value):
    """Read common manifest structures without imposing an undocumented schema."""
    if isinstance(value, list):
        for item in value:
            yield from manifest_hashes(item)
    elif isinstance(value, dict):
        digest = value.get("sha256")
        filename = value.get("path", value.get("filename", value.get("file")))
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest) and isinstance(filename, str):
            yield filename, digest.lower()
        for key, item in value.items():
            if isinstance(item, str) and re.fullmatch(r"[0-9a-fA-F]{64}", item):
                if Path(key).suffix.lower() in {".png", ".pdf", ".csv", ".npz", ".json", ".tif", ".joblib", ".ipynb"}:
                    yield key, item.lower()
                if key.endswith("_sha256") and isinstance(value.get(key[:-7]), str):
                    yield value[key[:-7]], item.lower()
            elif isinstance(item, (dict, list)):
                yield from manifest_hashes(item)


def verify_figures():
    require(PUBLICATION.is_dir(), "Publication figure directory is missing.")
    patterns = re.compile(r"(?:figure|fig)[_ -]*0?([1-6])(?=[_. -]|$)", re.IGNORECASE)
    groups = {}
    for path in sorted(PUBLICATION.iterdir()):
        if path.suffix.lower() not in {".pdf", ".png"}:
            continue
        match = patterns.search(path.stem)
        if match:
            groups.setdefault(int(match.group(1)), {}).setdefault(path.suffix.lower(), []).append(path)
    require(set(groups) == set(range(1, 7)), f"Expected publication figures 1–6; found {sorted(groups)}.")
    details = []
    for number, formats in sorted(groups.items()):
        require(set(formats) == {".png", ".pdf"} and all(len(v) == 1 for v in formats.values()), f"Figure {number} needs exactly one PDF and one PNG.")
        png, pdf = formats[".png"][0], formats[".pdf"][0]
        require(png.stem == pdf.stem, f"Figure {number} PDF/PNG basenames differ.")
        with Image.open(png) as image:
            require(image.format == "PNG", f"Invalid PNG format: {png.name}")
            dimensions = image.size
            dpi = image.info.get("dpi")
            require(dpi is not None and len(dpi) == 2 and all(abs(float(d) - 600) <= 0.5 for d in dpi), f"PNG resolution is not 600 dpi: {png.name}: {dpi}")
            require(dimensions[0] > 0 and dimensions[1] > 0, f"PNG dimensions are empty: {png.name}")
            image.verify()
        content = pdf.read_bytes()
        require(content.startswith(b"%PDF-") and content.rstrip().endswith(b"%%EOF"), f"Invalid PDF header/trailer: {pdf.name}")
        box = re.search(rb"/MediaBox\s*\[\s*([-+.\d]+)\s+([-+.\d]+)\s+([-+.\d]+)\s+([-+.\d]+)\s*\]", content)
        require(box is not None, f"Cannot inspect PDF page dimensions: {pdf.name}")
        x0, y0, x1, y1 = map(float, box.groups())
        require(x1 > x0 and y1 > y0, f"Invalid PDF page dimensions: {pdf.name}")
        pages = len(re.findall(rb"/Type\s*/Page(?=\s|/|>)", content))
        require(pages == 1, f"Expected one-page figure PDF: {pdf.name}; found {pages}")
        details.append({"figure": number, "png": str(png.relative_to(ROOT)), "pdf": str(pdf.relative_to(ROOT)),
                        "png_pixels": list(dimensions), "png_dpi": list(dpi),
                        "png_physical_inches": [dimensions[i] / dpi[i] for i in range(2)],
                        "pdf_page_inches": [(x1 - x0) / 72, (y1 - y0) / 72],
                        "png_sha256": sha256(png), "pdf_sha256": sha256(pdf)})
    manifests = sorted(set(PUBLICATION.glob("*manifest*.json")) | set(PUBLICATION.glob("*manifest*.csv")) |
                       set(REPRO.glob("*figure*manifest*.json")) | set(REPRO.glob("*figure*manifest*.csv")))
    manifest_details = []
    for path in manifests:
        if path.suffix.lower() == ".json":
            data = read_json(path)
        else:
            with path.open() as stream:
                data = list(csv.DictReader(stream))
        hashes = list(dict.fromkeys(manifest_hashes(data)))
        for filename, expected in hashes:
            candidate = Path(filename)
            if not candidate.is_absolute():
                candidates = [ROOT / candidate, PUBLICATION / candidate, path.parent / candidate]
                candidate = next((p for p in candidates if p.is_file()), candidates[0])
            require(candidate.is_file() and sha256(candidate) == expected, f"Manifest hash mismatch: {path.name}: {filename}")
        declared_figures = data.get("figures", data.get("figs")) if isinstance(data, dict) else None
        if isinstance(declared_figures, dict):
            require(set(declared_figures) == {str(i) for i in range(1, 7)}, f"Manifest does not declare figures 1–6: {path.name}")
            for observed in details:
                declared = declared_figures[str(observed["figure"])]
                require(int(declared["png_dpi"]) == 600, f"Manifest dpi differs for figure {observed['figure']}.")
                require(list(declared["png_size_pixels"]) == observed["png_pixels"], f"Manifest pixel dimensions differ for figure {observed['figure']}.")
                inches = declared["size_inches"]
                require(len(inches) == 2 and all(float(v) > 0 for v in inches), f"Invalid declared dimensions for figure {observed['figure']}.")
                require(np.allclose(observed["pdf_page_inches"], inches, rtol=0, atol=0.05),
                        f"PDF dimensions differ from manifest for figure {observed['figure']}.")
        manifest_details.append({"path": str(path.relative_to(ROOT)), "hashes_verified": len(hashes),
                                 "sha256_fields_present": bool(hashes), "figure_dimensions_checked": isinstance(declared_figures, dict)})
    return {"figure_pairs": 6, "figures": details, "manifests": manifest_details,
            "visual_review_performed_by_this_script": False}


def main():
    results = []
    for name, check in [
        ("original_inventory", verify_originals),
        ("notebook_structure_and_syntax", verify_notebook),
        ("model_matrix_and_fold_separation", verify_data_and_folds),
        ("validation_metrics_and_provenance", verify_validation),
        ("shap_cache_and_numerical_consistency", verify_shap),
        ("six_publication_figure_exports", verify_figures),
    ]:
        try:
            details = check()
            results.append({"name": name, "status": "passed", "details": details})
            print(f"PASS {name}", flush=True)
        except Exception as exc:
            results.append({"name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAIL {name}: {exc}", flush=True)
    verified = all(item["status"] == "passed" for item in results)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "status": "verified" if verified else "failed", "training_performed": False,
              "figures_rendered": False, "checks": results}
    REPRO.mkdir(parents=True, exist_ok=True)
    path = REPRO / "publication_verification.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
    print(f"Verification {report['status']}: {path}", flush=True)
    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())
