"""Additional spatial-error and ROC/PR figures from saved nested OOF predictions.

This script never fits a model, changes a threshold, recalibrates a score, or
modifies historical results. Every new artifact is written through review_common
to figures/review_2026-10-08/. The fixed-0.50 classifications are the saved
predictions, including the original classifier's handling of any exact ties.
"""

from __future__ import annotations

import json

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

from review_common import (
    ROOT, OUT, DATA, INK, MUTED, RF_COLOR, XGB_COLOR,
    read_csv, input_record, save_figure, save_table,
    load_nested, load_grid, load_boundary, finish_map, panel,
)


MODEL_SPEC = {
    "Random Forest": {"score": "rf_score", "prediction": "rf_pred", "color": RF_COLOR, "marker": "o", "short": "RF"},
    "XGBoost": {"score": "xgb_score", "prediction": "xgb_pred", "color": XGB_COLOR, "marker": "s", "short": "XGBoost"},
}
SUPPORT_MINIMUM = 10
SCORE_THRESHOLD = 0.50
REPRO = ROOT / "results/reproducibility"
NESTED_INPUTS = [
    ROOT / "data/missouri_wildfire_model_encoded.csv",
    REPRO / "xgb_nested_oof_predictions.csv",
    REPRO / "outer_fold_assignments.csv",
    REPRO / "model_evaluation_summary.csv",
    REPRO / "validation_report.json",
    ROOT / "results/rf_nested_spatial_cv_results.csv",
    REPRO / "xgb_nested_fold_results.csv",
]


def metric_record(model, part, fold=0):
    spec = MODEL_SPEC[model]
    target = part.fire_presence.to_numpy(dtype=int)
    probability = part[spec["score"]].to_numpy(dtype=float)
    prediction = part[spec["prediction"]].to_numpy(dtype=int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(target, prediction, labels=[0, 1]).ravel())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    return {
        "model": model, "outer_fold": fold, "cells": len(part),
        "recorded_fire_cells": tp + fn, "no_recorded_fire_cells": tn + fp,
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
        "saved_threshold": SCORE_THRESHOLD,
        "true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp,
        "precision_at_0_50": precision, "recall_at_0_50": recall,
        "false_negative_rate_at_0_50": fn / (tp + fn) if tp + fn else np.nan,
        "false_positive_rate_at_0_50": fp / (tn + fp) if tn + fp else np.nan,
        "accuracy_at_0_50": (tn + tp) / len(part),
        "f1_at_0_50": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
    }


def prepare_diagnostics():
    nested = load_nested().copy()
    if len(nested) != 7229 or not nested.grid_id.is_unique:
        raise ValueError("Expected exactly 7,229 original modeling cells.")
    if sorted(nested.fire_presence.unique().tolist()) != [0, 1]:
        raise ValueError("Expected binary recorded-wildfire labels.")
    if nested.block_id.nunique() != 94 or nested.groupby("block_id").outer_fold.nunique().max() != 1:
        raise ValueError("The original 94 spatial blocks or their outer folds changed.")
    reference = read_csv(REPRO / "model_evaluation_summary.csv")
    reference = reference[reference.validation == "Nested spatial 5-fold CV"].set_index("model")
    report = json.loads((REPRO / "validation_report.json").read_text())
    if report.get("status") != "verified" or not reference.verified.astype(str).str.lower().eq("true").all():
        raise ValueError("Use verified nested prediction artifacts before generating review diagnostics.")
    pooled_records, fold_records, roc_records, pr_records = [], [], [], []
    for model, spec in MODEL_SPEC.items():
        probability = nested[spec["score"]].to_numpy(dtype=float)
        prediction = nested[spec["prediction"]].to_numpy(dtype=int)
        if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
            raise ValueError(f"Invalid saved nested scores for {model}.")
        if not np.isin(prediction, [0, 1]).all():
            raise ValueError(f"Invalid saved binary predictions for {model}.")
        # Exact 0.50 ties retain the saved estimator's decision rule.
        away_from_tie = probability != SCORE_THRESHOLD
        if not np.array_equal(prediction[away_from_tie], (probability[away_from_tie] >= SCORE_THRESHOLD).astype(int)):
            raise ValueError(f"Saved predictions do not use the original 0.50 rule for {model}.")
        pooled = metric_record(model, nested)
        expected_auc = float(reference.loc[model, "pooled_oof_auc"])
        if not np.isclose(pooled["roc_auc"], expected_auc, rtol=0, atol=1e-12):
            raise ValueError(f"Pooled {model} AUC differs from verified reference: {pooled['roc_auc']} vs {expected_auc}.")
        pooled_records.append(pooled)
        fold_reference = read_csv(ROOT / "results/rf_nested_spatial_cv_results.csv" if model == "Random Forest"
                                  else REPRO / "xgb_nested_fold_results.csv").set_index("fold")
        for fold, part in [(0, nested)] + [(f, nested[nested.outer_fold == f]) for f in range(1, 6)]:
            if fold:
                result = metric_record(model, part, fold)
                original = fold_reference.loc[fold]
                comparisons = {"roc_auc": "roc_auc", "accuracy_at_0_50": "accuracy",
                               "precision_at_0_50": "precision", "recall_at_0_50": "recall", "f1_at_0_50": "f1"}
                if any(not np.isclose(result[new], float(original[old]), rtol=0, atol=1e-12) for new, old in comparisons.items()):
                    raise ValueError(f"Saved {model} outer fold {fold} metrics do not match the preserved results.")
                fold_records.append(result)
            target = part.fire_presence.to_numpy(dtype=int)
            score = part[spec["score"]].to_numpy(dtype=float)
            fpr, tpr, thresholds = roc_curve(target, score, drop_intermediate=False)
            roc_records.append(pd.DataFrame({
                "model": model, "scope": "pooled" if fold == 0 else "outer_fold", "outer_fold": fold,
                "point_index": np.arange(len(fpr)), "false_positive_rate": fpr, "true_positive_rate": tpr,
                "threshold": np.where(np.isfinite(thresholds), thresholds, np.nan),
            }))
            precision, recall, thresholds = precision_recall_curve(target, score, drop_intermediate=False)
            pr_records.append(pd.DataFrame({
                "model": model, "scope": "pooled" if fold == 0 else "outer_fold", "outer_fold": fold,
                "point_index": np.arange(len(recall)), "recall": recall, "precision": precision,
                "threshold": np.r_[thresholds, np.nan],
            }))
    pooled = pd.DataFrame(pooled_records)
    folds = pd.DataFrame(fold_records)
    roc_points = pd.concat(roc_records, ignore_index=True)
    pr_points = pd.concat(pr_records, ignore_index=True)
    tables = {
        "pooled": save_table(pooled, "diagnostics_nested_model_metrics.csv"),
        "folds": save_table(folds, "diagnostics_outer_fold_metrics.csv"),
        "roc": save_table(roc_points, "diagnostics_roc_points.csv"),
        "pr": save_table(pr_points, "diagnostics_pr_points.csv"),
    }
    return nested, pooled, folds, roc_points, pr_points, tables


def support_category(denominator):
    if denominator == 0:
        return "no_relevant_observations"
    return "sparse_1_to_9" if denominator < SUPPORT_MINIMUM else "supported_at_least_10"


def block_errors(nested):
    rows = []
    for block, part in nested.groupby("block_id", sort=True):
        positive = part.fire_presence.eq(1)
        negative = ~positive
        n_positive, n_negative = int(positive.sum()), int(negative.sum())
        row = {"block_id": block, "outer_fold": int(part.outer_fold.iloc[0]), "cells": len(part),
               "recorded_fire_cells": n_positive, "no_recorded_fire_cells": n_negative,
               "false_negative_rate_support": support_category(n_positive),
               "false_positive_rate_support": support_category(n_negative)}
        for prefix in ["rf", "xgb"]:
            prediction = part[f"{prefix}_pred"]
            fn = int((positive & prediction.eq(0)).sum())
            fp = int((negative & prediction.eq(1)).sum())
            row[f"{prefix}_false_negative"] = fn
            row[f"{prefix}_false_positive"] = fp
            row[f"{prefix}_false_negative_rate"] = fn / n_positive if n_positive else np.nan
            row[f"{prefix}_false_positive_rate"] = fp / n_negative if n_negative else np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 94 or result.cells.sum() != len(nested):
        raise ValueError("Block error summaries do not cover the original modeling cells.")
    return result


def spatial_errors(nested, pooled, table_paths):
    errors = block_errors(nested)
    block_path = save_table(errors, "diagnostics_block_errors.csv")
    grid = load_grid()[["grid_id", "geometry"]].copy()
    if grid.crs.to_epsg() != 5070:
        raise ValueError("Use preserved EPSG:5070 grid geometry.")
    cells = grid.merge(nested[["grid_id", "block_id"]], on="grid_id", how="inner", validate="one_to_one")
    if len(cells) != 7229:
        raise ValueError("Spatial join lost original grid cells.")
    blocks = cells.dissolve(by="block_id", as_index=False)[["block_id", "geometry"]]
    blocks = blocks.merge(errors, on="block_id", validate="one_to_one")
    boundary = load_boundary()
    norm = Normalize(0, 1)
    cmap = plt.get_cmap("YlOrRd")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.75))
    fig.subplots_adjust(left=.073, right=.985, top=.87, bottom=.205, wspace=.18)
    rf = pooled.set_index("model").loc["Random Forest"]
    map_specs = [
        ("a", "False-negative rate", "rf_false_negative_rate", "recorded_fire_cells", "false_negative_rate_support",
         f"{int(rf.false_negative):,} / {int(rf.recorded_fire_cells):,} recorded-fire cells missed ({rf.false_negative_rate_at_0_50:.1%})"),
        ("b", "False-positive rate", "rf_false_positive_rate", "no_recorded_fire_cells", "false_positive_rate_support",
         f"{int(rf.false_positive):,} / {int(rf.no_recorded_fire_cells):,} no-recorded-fire cells flagged ({rf.false_positive_rate_at_0_50:.1%})"),
    ]
    labels_rendered = {}
    for ax, (letter, title, rate, denominator, support, annotation) in zip(axes, map_specs):
        supported = blocks[blocks[denominator] >= SUPPORT_MINIMUM]
        sparse = blocks[(blocks[denominator] > 0) & (blocks[denominator] < SUPPORT_MINIMUM)]
        absent = blocks[blocks[denominator] == 0]
        if len(supported):
            supported.plot(ax=ax, column=rate, cmap=cmap, norm=norm, edgecolor="#807f79", linewidth=.35)
        if len(sparse):
            sparse.plot(ax=ax, facecolor="#d5d8da", edgecolor="#7c8084", linewidth=.35, hatch="...")
        if len(absent):
            absent.plot(ax=ax, facecolor="white", edgecolor="#7c8084", linewidth=.4, hatch="///")
        boundary.boundary.plot(ax=ax, edgecolor=INK, linewidth=.7)
        finish_map(ax, boundary)
        # MaxNLocator supplies ticks just beyond the visible limits; omit those
        # labels explicitly so export auditing sees only displayed coordinates.
        x_limits, y_limits = ax.get_xlim(), ax.get_ylim()
        ax.set_xticks([tick for tick in ax.get_xticks() if x_limits[0] <= tick <= x_limits[1]])
        ax.set_yticks([tick for tick in ax.get_yticks() if y_limits[0] <= tick <= y_limits[1]])
        panel(ax, letter, title)
        ax.text(.0, 1.01, annotation, transform=ax.transAxes, color=MUTED, fontsize=6.7, va="bottom")
        labels = 0
        for row in blocks.itertuples(index=False):
            # Very small clipped edge fragments cannot carry a readable label.
            if row.geometry.area < 250_000_000:
                continue
            position = row.geometry.representative_point()
            if position.distance(row.geometry.boundary) < 9_000:
                continue
            count = getattr(row, denominator)
            value = getattr(row, rate)
            color = "white" if count >= SUPPORT_MINIMUM and value >= .62 else INK
            ax.text(position.x, position.y, str(count), ha="center", va="center", fontsize=5.9,
                    color=color, zorder=5, clip_on=True)
            labels += 1
        labels_rendered[rate] = labels
    cax = fig.add_axes([.26, .105, .48, .025])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    colorbar.set_ticks([0, .25, .5, .75, 1], labels=["0", "25", "50", "75", "100"])
    colorbar.ax.tick_params(labelsize=7, length=2)
    colorbar.set_label("Error rate among the relevant observed class (%)", fontsize=8, labelpad=3)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    handles = [Patch(facecolor="#d5d8da", edgecolor="#7c8084", hatch="...", label="Sparse support: 1–9 relevant cells"),
               Patch(facecolor="white", edgecolor="#7c8084", hatch="///", label="Undefined rate: no relevant cells")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .145), ncol=2,
               frameon=False, fontsize=7, handlelength=1.7, columnspacing=1.5)
    fig.text(.073, .963, "Where nested Random Forest predictions miss or flag recorded wildfire", fontsize=10.3,
             fontweight="semibold", color=INK, va="top")
    fig.text(.073, .016, "Labels show class-specific denominators where space permits. Fixed 0.50 rule; 94 spatial blocks.",
             fontsize=7.1, color=MUTED, va="bottom")
    detail = {
        "inputs": [input_record(p) for p in NESTED_INPUTS] + [input_record(ROOT / "data/Missouri_Wildfire_Susceptibility_5km.gpkg")],
        "derived_tables": [input_record(block_path), input_record(table_paths["pooled"])],
        "model": "Random Forest", "prediction_source": "Preserved rf_nested_oof_pred; geometry joined by grid_id; GeoPackage rf_susc/rf_oof are not used.",
        "threshold": SCORE_THRESHOLD, "threshold_optimized_here": False, "model_fitting_performed": False,
        "support_threshold": SUPPORT_MINIMUM,
        "support_rule": "Rates are colored only for denominators >=10; 1–9 are sparse gray stipple; 0 is undefined white hatch.",
        "denominators": {"false_negative_rate": "Recorded-fire-positive cells in block", "false_positive_rate": "Cells with no recorded wildfire in block"},
        "denominator_labels": "Printed where block polygon area is at least 250 square kilometers and representative-point edge clearance is at least 9 km; all counts retained in diagnostics_block_errors.csv.",
        "denominator_labels_rendered": labels_rendered,
        "support_counts": {name: {str(k): int(v) for k, v in errors[name].value_counts().items()}
                           for name in ["false_negative_rate_support", "false_positive_rate_support"]},
        "global_error_counts_and_metrics": rf.to_dict(),
        "projection": "EPSG:5070", "geometry": "Unions of clipped original modeling cells within saved 50-km block IDs.",
        "interpretation": "Descriptive spatial error heterogeneity for one nested OOF prediction per cell; not predictive uncertainty, calibrated risk, or independent external validation.",
    }
    boundary_path = ROOT / "data/derived/missouri_boundary.geojson"
    if boundary_path.exists():
        detail["inputs"].append(input_record(boundary_path))
    save_figure(fig, "A1_nested_spatial_errors", detail)


def roc_pr(nested, pooled, folds, roc_points, pr_points, table_paths):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.95))
    fig.subplots_adjust(left=.085, right=.985, top=.85, bottom=.20, wspace=.25)
    pooled_index = pooled.set_index("model")
    for model, spec in MODEL_SPEC.items():
        line_style = "-" if model == "Random Forest" else "--"
        for fold in range(1, 6):
            roc = roc_points[(roc_points.model == model) & (roc_points.outer_fold == fold)]
            pr = pr_points[(pr_points.model == model) & (pr_points.outer_fold == fold)]
            axes[0].plot(roc.false_positive_rate, roc.true_positive_rate, color=spec["color"], alpha=.18,
                         lw=.65, linestyle=line_style, zorder=1)
            axes[1].plot(pr.recall, pr.precision, color=spec["color"], alpha=.16, lw=.55,
                         linestyle=line_style, drawstyle="steps-post", zorder=1)
        roc = roc_points[(roc_points.model == model) & (roc_points.outer_fold == 0)]
        pr = pr_points[(pr_points.model == model) & (pr_points.outer_fold == 0)]
        metric = pooled_index.loc[model]
        axes[0].plot(roc.false_positive_rate, roc.true_positive_rate, color=spec["color"], lw=1.9,
                     linestyle=line_style, label=f"{spec['short']}  AUC = {metric.roc_auc:.3f}", zorder=3)
        axes[1].plot(pr.recall, pr.precision, color=spec["color"], lw=1.8, linestyle=line_style,
                     drawstyle="steps-post", label=f"{spec['short']}  AP = {metric.average_precision:.3f}", zorder=3)
        axes[0].scatter([metric.false_positive_rate_at_0_50], [metric.recall_at_0_50], marker=spec["marker"],
                        s=34, c=[spec["color"]], edgecolor="white", linewidth=.65, zorder=5)
        axes[1].scatter([metric.recall_at_0_50], [metric.precision_at_0_50], marker=spec["marker"],
                        s=34, c=[spec["color"]], edgecolor="white", linewidth=.65, zorder=5)
    prevalence = float(nested.fire_presence.mean())
    axes[0].plot([0, 1], [0, 1], color="#969b9f", lw=.8, linestyle=":", zorder=0)
    axes[1].axhline(prevalence, color="#969b9f", lw=.8, linestyle=":", zorder=0)
    axes[1].text(.03, prevalence - .035, f"Recorded-fire prevalence: {prevalence:.1%}", fontsize=7,
                 color=MUTED, va="top")
    panel(axes[0], "a", "Receiver operating characteristic")
    panel(axes[1], "b", "Precision–recall")
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("Recall / true-positive rate")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    for ax in axes:
        ax.set_xlim(-.01, 1.01)
        ax.set_ylim(-.01, 1.01)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(color="#e4e7e9", linewidth=.5, alpha=.65)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5, length=3)
        ax.legend(loc="lower right", fontsize=7.5, frameon=False, handlelength=2.3)
    fig.text(.085, .965, "Nested spatial validation: RF and XGBoost across classification thresholds", fontsize=10.2,
             fontweight="semibold", color=INK, va="top")
    fig.text(.085, .08, "Bold curves: pooled held-out predictions. Thin curves: five outer folds, shown descriptively.",
             fontsize=7.2, color=MUTED)
    fig.text(.085, .035, "Points: saved 0.50 classifications. AP: average precision. No threshold selection or confidence bands.",
             fontsize=7.2, color=MUTED)
    save_figure(fig, "S4_nested_roc_pr", {
        "inputs": [input_record(p) for p in NESTED_INPUTS],
        "derived_tables": [input_record(p) for p in table_paths.values()],
        "model_fitting_performed": False, "threshold_optimized_here": False, "score_calibration_performed": False,
        "pooled_metrics": pooled.to_dict(orient="records"), "outer_fold_metrics": folds.to_dict(orient="records"),
        "precision_recall_metric": "sklearn.metrics.average_precision_score; not trapezoidal PR-AUC",
        "operating_points": "Pooled confusion counts from saved binary predictions at the original 0.50 rule; not averages of fold precision/recall.",
        "pooled_roc_auc_verified_against_cache": True, "outer_metrics_verified_against_cache": True,
        "reference_prevalence": prevalence, "curves": "Bold pooled OOF curves plus thin per-outer-fold curves; thin curves are not confidence bands.",
        "threshold_csv_note": "Infinite initial ROC thresholds and undefined terminal PR thresholds are stored blank.",
        "uncertainty": "None estimated. One nested OOF prediction per cell; overlapping training partitions do not give independent replications.",
    })


def main():
    nested, pooled, folds, roc_points, pr_points, table_paths = prepare_diagnostics()
    spatial_errors(nested, pooled, table_paths)
    roc_pr(nested, pooled, folds, roc_points, pr_points, table_paths)
    for row in pooled.itertuples(index=False):
        print(f"{row.model}: pooled AUC={row.roc_auc:.9f}; AP={row.average_precision:.9f}; "
              f"FN={row.false_negative}/{row.recorded_fire_cells}; FP={row.false_positive}/{row.no_recorded_fire_cells}; "
              f"precision={row.precision_at_0_50:.6f}; recall={row.recall_at_0_50:.6f}", flush=True)
    print(f"Review diagnostics saved only under {OUT}", flush=True)


if __name__ == "__main__":
    main()
