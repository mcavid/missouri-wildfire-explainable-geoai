"""Create supplementary review figures from preserved result tables only.

No model is fitted, tuned, predicted, or explained here. Original artifacts are
read-only; figures, numeric tables, and provenance go to review_2026-10-08 only.
Run in the project geoai environment: python scripts/review_supporting.py
"""

from __future__ import annotations

import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from review_common import (
    ROOT, OUT, DATA, read_csv, save_table, input_record, save_figure,
    panel, INK, MUTED, RF_COLOR, XGB_COLOR,
)


METRICS = ["ROC_AUC", "Accuracy", "Precision", "Recall", "F1"]
FOLDS = [f"Fold_{i}" for i in range(1, 6)]
MANUSCRIPT = ROOT / "docs/methods.md"
HISTORICAL_LIMIT = (
    "Original generation code and fitted fold models for these historical summaries "
    "are absent from the project. Arithmetic checks verify the saved tables' internal "
    "consistency; they do not independently verify upstream modeling or SHAP generation."
)


def table(path):
    return read_csv(ROOT / path)


def check_folds(df, group):
    for label, part in df.groupby(group, sort=False):
        if sorted(part["Fold"].tolist()) != [1, 2, 3, 4, 5]:
            raise ValueError(f"Expected exactly five saved fold rows for {label}.")
    if not np.isfinite(df[METRICS].to_numpy()).all():
        raise ValueError("Nonfinite saved metric.")


def metric_summary(df, group):
    return pd.DataFrame([
        {group: label, "metric": metric, "n_folds": len(part),
         "mean": part[metric].mean(), "sd_ddof1": part[metric].std(ddof=1),
         "minimum": part[metric].min(), "maximum": part[metric].max()}
        for label, part in df.groupby(group, sort=False) for metric in METRICS
    ])


def check_rounded_summary(df, group, path):
    # These historical pandas exports have three header rows; use the saved
    # full-precision fold table for calculation and the rounded file only as QA.
    saved = pd.read_csv(ROOT / path, header=[0, 1], index_col=0).dropna(how="all")
    for label, part in df.groupby(group, sort=False):
        for metric in METRICS:
            for stat, value in [("mean", part[metric].mean()),
                                ("std", part[metric].std(ddof=1))]:
                if not np.isclose(round(value, 3), float(saved.loc[label, (metric, stat)]), atol=1e-12):
                    raise ValueError(f"Rounded historical summary mismatch: {label}, {metric}, {stat}.")


def subtle_grid(ax, axis="y"):
    ax.grid(axis=axis, color="#D8DFE5", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def s1_robustness():
    paths = [
        "results/block_size_sensitivity_fold_results.csv",
        "results/temporal_sensitivity_fold_results.csv",
        "results/block_size_sensitivity_summary.csv",
        "results/temporal_sensitivity_summary.csv",
    ]
    blocks, omission = table(paths[0]), table(paths[1])
    check_folds(blocks, "Block_size")
    check_folds(omission, "Model")
    check_rounded_summary(blocks, "Block_size", paths[2])
    check_rounded_summary(omission, "Model", paths[3])
    full = omission[omission.Model == "Full model"].set_index("Fold").sort_index()
    fifty = blocks[blocks.Block_size == "50 km"].set_index("Fold").sort_index()
    if not np.allclose(full[METRICS], fifty[METRICS], atol=1e-14, rtol=0):
        raise ValueError("Full-model omission reference does not match saved 50-km results.")

    deltas = []
    for label, part in omission.groupby("Model", sort=False):
        for _, row in part.iterrows():
            entry = {"Model": label, "Fold": int(row.Fold)}
            for metric in METRICS:
                entry[f"{metric}_full"] = full.loc[row.Fold, metric]
                entry[f"{metric}_omission"] = row[metric]
                entry[f"{metric}_delta"] = row[metric] - full.loc[row.Fold, metric]
            deltas.append(entry)
    delta = pd.DataFrame(deltas)
    block_summary = metric_summary(blocks, "Block_size")
    omission_summary = metric_summary(omission, "Model")
    delta_summary = pd.DataFrame([
        {"Model": label, "metric": metric, "n_folds": len(part),
         "mean_delta": part[f"{metric}_delta"].mean(),
         "sd_delta_ddof1": part[f"{metric}_delta"].std(ddof=1)}
        for label, part in delta.groupby("Model", sort=False) for metric in METRICS
    ])
    save_table(blocks, "supporting_block_size_fold_results.csv")
    save_table(block_summary, "supporting_block_size_summary.csv")
    save_table(omission, "supporting_omission_fold_results.csv")
    save_table(omission_summary, "supporting_omission_summary.csv")
    save_table(delta, "supporting_omission_fold_deltas.csv")
    save_table(delta_summary, "supporting_omission_delta_summary.csv")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.75), gridspec_kw={"width_ratios": [1, 1.08]})
    fig.subplots_adjust(left=.09, right=.96, bottom=.27, top=.83, wspace=.70)
    panel(ax, "a", "Spatial block size")
    jitter = np.linspace(-.14, .06, 5)
    for x, label in enumerate(["25 km", "50 km", "75 km"]):
        values = blocks[blocks.Block_size == label].sort_values("Fold").ROC_AUC.to_numpy()
        ax.scatter(x + jitter, values, s=18, color=RF_COLOR, alpha=.65, linewidths=0, zorder=3)
        ax.errorbar(x + .16, values.mean(), yerr=values.std(ddof=1), fmt="D", markersize=4,
                    color=INK, ecolor=INK, lw=1.2, capsize=3, zorder=4)
        ax.text(x, .765, f"{values.mean():.3f}", ha="center", fontsize=8, color=INK)
    ax.set(xlim=(-.42, 2.45), ylim=(.60, .78), ylabel="ROC-AUC",
           xticks=[0, 1, 2], xticklabels=["25 km", "50 km", "75 km"])
    subtle_grid(ax)
    panel(bx, "b", "Predictor omission")
    bx.axvline(0, color=MUTED, lw=.9, linestyle="--", zorder=1)
    omission_order = ["No NDVI", "No FBFM40", "No NDVI + No FBFM40"]
    for y, label in enumerate(omission_order):
        values = delta[delta.Model == label].sort_values("Fold").ROC_AUC_delta.to_numpy()
        bx.scatter(values, y + jitter, s=18, color=XGB_COLOR, alpha=.7, linewidths=0, zorder=3)
        bx.errorbar(values.mean(), y + .16, xerr=values.std(ddof=1), fmt="D", markersize=4,
                    color=INK, ecolor=INK, lw=1.2, capsize=3, zorder=4)
    bx.set(yticks=[0, 1, 2], yticklabels=["Without NDVI", "Without fuels", "Without both"],
           ylim=(2.5, -.5), xlim=(-.0046, .0046), xlabel="ROC-AUC change from full model")
    bx.set_xticks([-.004, -.002, 0, .002, .004])
    bx.set_xticklabels(["−0.004", "−0.002", "0", "+0.002", "+0.004"])
    subtle_grid(bx, "x")
    handles = [Line2D([], [], marker="o", linestyle="", color=MUTED, markersize=4, label="Saved fold value"),
               Line2D([], [], marker="D", linestyle="", color=INK, markersize=4, label="Mean ± sample SD (5 folds)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .11), frameon=False, ncol=2, fontsize=8)
    fig.text(.5, .035, "Fixed-configuration spatial CV · historical results\nSD describes fold variation, not a confidence interval.",
             ha="center", va="bottom", fontsize=7.7, color=MUTED)
    caption = (
        "Figure S1. Sensitivity of saved fixed-configuration Random Forest spatial validation. "
        "(a) ROC-AUC for 25-, 50-, and 75-km grouping blocks; points are individual saved fold values. "
        "Fold numbers at different block sizes do not identify equivalent test partitions and are not joined. "
        "(b) Change in ROC-AUC after omitting NDVI, FBFM40 indicators, or both, relative to the full model, "
        "matched by the fold labels in the saved tables. The full-model mean is "
        f"{full.ROC_AUC.mean():.6f}. Diamonds and whiskers show mean and sample SD (ddof=1) across five "
        "folds or five within-label differences; these are not confidence intervals. These historical "
        "fixed-configuration results are separate from nested model-family evaluation. " + HISTORICAL_LIMIT
    )
    save_figure(fig, "S1_validation_robustness", {
        "inputs": [input_record(ROOT / p) for p in paths] + [input_record(MANUSCRIPT)], "caption": caption,
        "method": "Full-precision fold CSV summaries; no model execution. Sample SD uses ddof=1. Omission differences subtract the same saved Fold label.",
        "provenance_limits": [HISTORICAL_LIMIT, "Omission partition identity is reported in the manuscript but is not independently established by these metric-only files."],
        "checks": {"five_rows_each_configuration": True, "rounded_historical_summaries_match": True,
                   "full_model_equals_50km_all_saved_metrics": True},
    })
    plt.close(fig)


FEATURE_LABELS = {
    "population": "Population", "ndvi": "NDVI", "canopy": "Tree canopy cover",
    "precipitation": "Precipitation", "road_distance": "Distance to roads",
    "temperature": "Temperature", "elevation": "Elevation", "slope": "Slope",
    "lc_41": "Deciduous forest (LC 41)", "wind": "Wind speed", "pdsi": "PDSI",
    "lc_82": "Land-cover class 82", "fuel_93": "Fuel class 93",
    "northness": "Northness", "eastness": "Eastness",
}


def s2_shap_stability():
    paths = ["results/shap_spatial_fold_stability.csv", "results/shap_fold_rank_correlations.csv"]
    saved = table(paths[0]).set_index("Unnamed: 0")
    saved.index.name = "feature"
    corr_saved = table(paths[1]).set_index("Unnamed: 0").loc[FOLDS, FOLDS]
    importance = saved[FOLDS]
    if importance.shape != (33, 5) or importance.isna().any().any() or (importance < 0).any().any():
        raise ValueError("Expected nonnegative, complete 33-by-5 SHAP importance table.")
    ranks = importance.rank(axis=0, ascending=False, method="average")
    calc = pd.DataFrame({"Mean_SHAP": importance.mean(axis=1), "Mean_Rank": ranks.mean(axis=1),
                         "Top10_Count": (ranks <= 10).sum(axis=1)})
    for column in calc:
        if not np.allclose(calc[column], saved[column], rtol=0, atol=1e-12):
            raise ValueError(f"SHAP stability arithmetic mismatch: {column}.")
    corr = ranks.corr(method="pearson")
    if not np.allclose(corr, corr_saved, rtol=0, atol=1e-12):
        raise ValueError("Saved rank correlations do not match the 33-feature fold rankings.")
    ranked_export = ranks.rename(columns={c: f"rank_{c}" for c in FOLDS}).join(calc)
    ranked_export["Minimum_Rank"] = ranks.min(axis=1)
    ranked_export["Maximum_Rank"] = ranks.max(axis=1)
    save_table(importance.reset_index(), "supporting_shap_fold_importance.csv")
    save_table(ranked_export.reset_index(), "supporting_shap_fold_ranks.csv")
    save_table(corr.rename_axis("fold").reset_index(), "supporting_shap_rank_correlations.csv")
    top = calc.sort_values("Mean_SHAP", ascending=False).head(15).index
    shown = ranks.loc[top]
    off_diagonal = corr.to_numpy()[np.triu_indices(5, 1)]
    stable_count = int((calc.Top10_Count == 5).sum())

    fig = plt.figure(figsize=(7.2, 5.65))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], left=.205, right=.97,
                          bottom=.15, top=.89, wspace=.42)
    ax = fig.add_subplot(gs[0, 0])
    right = gs[0, 1].subgridspec(2, 1, height_ratios=[1.1, .7], hspace=.20)
    bx = fig.add_subplot(right[0, 0])
    note = fig.add_subplot(right[1, 0]); note.axis("off")
    panel(ax, "a", "Feature ranks by fold")
    im = ax.pcolormesh(np.arange(6) - .5, np.arange(16) - .5, shown,
                       cmap="Blues_r", vmin=1, vmax=20, shading="flat")
    ax.set(xticks=np.arange(5), xticklabels=range(1, 6),
           yticks=np.arange(15), yticklabels=[FEATURE_LABELS.get(f, f) for f in top],
           xlabel="Outer fold (saved label)", xlim=(-.5, 4.5), ylim=(14.5, -.5))
    ax.tick_params(axis="both", length=0, labelsize=8)
    for y in range(15):
        for x in range(5):
            val = shown.iloc[y, x]
            ax.text(x, y, f"{val:g}", ha="center", va="center", fontsize=8,
                    color="white" if val <= 8 else INK)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=.12, fraction=.035, aspect=23)
    cb.set_ticks([1, 5, 10, 15, 20]); cb.ax.tick_params(labelsize=7, length=2)
    cb.set_label("Rank among 33 features (1 = highest)", fontsize=7.5)
    panel(bx, "b", "Rank agreement")
    bx.pcolormesh(np.arange(6) - .5, np.arange(6) - .5, corr,
                  cmap="YlGnBu", vmin=.95, vmax=1, shading="flat")
    bx.set_aspect("equal")
    bx.set(xticks=np.arange(5), xticklabels=range(1, 6), yticks=np.arange(5), yticklabels=range(1, 6),
           xlabel="Outer fold", ylabel="Outer fold", xlim=(-.5, 4.5), ylim=(4.5, -.5))
    bx.tick_params(length=0, labelsize=8)
    for y in range(5):
        for x in range(5):
            value = corr.iloc[y, x]
            bx.text(x, y, f"{value:.3f}", ha="center", va="center", fontsize=7.5,
                    color="white" if value >= .978 else INK)
    note.text(0, .96, f"Spearman ρ: {off_diagonal.min():.3f}–{off_diagonal.max():.3f}\n"
              f"{stable_count} features in the top 10\nin every saved fold.",
              ha="left", va="top", fontsize=9, color=INK, linespacing=1.7)
    note.text(0, .32, "All 33 features determine correlations.\nPanel a shows the leading 15\nby mean absolute SHAP importance.",
              ha="left", va="top", fontsize=7.8, color=MUTED, linespacing=1.4)
    fig.text(.5, .025, "Historical fold summaries: arithmetic checked; original SHAP generation not independently verified.",
             ha="center", va="bottom", fontsize=7.3, color=MUTED)
    caption = (
        "Figure S2. Consistency of the historical fold-specific Random Forest SHAP importance summaries. "
        "(a) Ranks within each saved fold among all 33 predictors, with ties assigned average ranks; "
        "the 15 features with largest mean absolute SHAP importance across the five folds are displayed. "
        "(b) Pairwise Spearman correlations computed from all 33 feature rankings. "
        f"Off-diagonal correlations range from {off_diagonal.min():.6f} to {off_diagonal.max():.6f}; "
        f"{stable_count} features rank in the top ten in every fold. Mean importance, mean ranks, top-ten "
        "counts, and the correlation matrix reproduce the supplied summary arithmetic. These are "
        "historical fold summaries, not the separately cached full-data SHAP explanation. " + HISTORICAL_LIMIT
    )
    save_figure(fig, "S2_shap_fold_stability", {
        "inputs": [input_record(ROOT / p) for p in paths] + [input_record(MANUSCRIPT)], "caption": caption,
        "method": "Rank saved mean absolute SHAP values within each fold (descending, average ties); correlate ranks across all 33 features. No SHAP computation.",
        "provenance_limits": [HISTORICAL_LIMIT, "The held-out outer-fold interpretation is documented by the manuscript; original fitted fold models and explanation arrays are not available for confirmation."],
        "checks": {"shape": [33, 5], "mean_importance_matches": True, "mean_rank_matches": True,
                   "top10_counts_match": True, "rank_correlations_match": True,
                   "off_diagonal_min": float(off_diagonal.min()), "off_diagonal_max": float(off_diagonal.max()),
                   "features_top10_all_folds": stable_count},
    })
    plt.close(fig)


def verify_raw_reporting(annual, composition, monthly):
    """Attribute-only SQL checks; no geometry processing or spatial ETL."""
    path = ROOT / "data/InFORM_FireOccurrence_MO.gpkg"
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        raw_annual = pd.read_sql_query(
            'SELECT CalendarYear, count(*) AS Wildfire_records FROM Incidents '
            'WHERE CalendarYear BETWEEN 2015 AND 2026 GROUP BY CalendarYear', connection)
        raw_sources = pd.read_sql_query(
            'SELECT CalendarYear, CreatedBySystem, count(*) AS Records FROM Incidents '
            'WHERE CalendarYear BETWEEN 2015 AND 2026 GROUP BY CalendarYear, CreatedBySystem', connection)
        raw_month = pd.read_sql_query(
            'SELECT CAST(substr(FireDiscoveryDateTime, 6, 2) AS INTEGER) AS month, count(*) AS fires '
            'FROM Incidents WHERE CalendarYear BETWEEN 2021 AND 2025 GROUP BY month', connection)
        categories = pd.read_sql_query(
            'SELECT IncidentTypeCategory, count(*) AS records FROM Incidents '
            'WHERE CalendarYear BETWEEN 2021 AND 2025 GROUP BY IncidentTypeCategory', connection)
    a = annual.merge(raw_annual, on="CalendarYear", suffixes=("_saved", "_raw"), validate="one_to_one")
    b = composition.merge(raw_sources, on=["CalendarYear", "CreatedBySystem"], suffixes=("_saved", "_raw"), validate="one_to_one")
    c = monthly.merge(raw_month, on="month", suffixes=("_saved", "_raw"), validate="one_to_one")
    if not (len(a) == len(annual) == len(raw_annual)
            and len(b) == len(composition) == len(raw_sources)
            and len(c) == len(monthly) == len(raw_month)):
        raise ValueError("Reporting tables and raw GPKG have different group keys.")
    for df, stem in [(a, "Wildfire_records"), (b, "Records"), (c, "fires")]:
        if not (df[f"{stem}_saved"] == df[f"{stem}_raw"]).all():
            raise ValueError(f"Saved reporting counts disagree with raw input: {stem}.")
    return {"annual_counts_match_raw": True, "source_counts_match_raw": True,
            "monthly_counts_match_raw": True, "raw_incident_categories_2021_2025": categories.to_dict("records"),
            "month_definition": "Month substring of FireDiscoveryDateTime; records selected by CalendarYear 2021–2025; no time-zone conversion.",
            "scope": "All records in the downloaded Missouri file; no spatial join, geometry filtering, or deduplication applied."}


def s3_reporting():
    paths = ["results/nifc_annual_record_counts_2015_2026.csv",
             "results/nifc_created_by_system_2015_2026.csv",
             "results/monthly_wildfire_distribution_2021_2025.csv"]
    annual, composition, monthly = [table(p) for p in paths]
    total = int(monthly.fires.sum())
    if total != int(annual[annual.CalendarYear.between(2021, 2025)].Wildfire_records.sum()):
        raise ValueError("Study-period monthly and annual totals differ.")
    if not np.allclose(monthly.percent, monthly.fires / total * 100, atol=1e-12, rtol=0):
        raise ValueError("Saved monthly percentages disagree with counts.")
    source_total = composition.groupby("CalendarYear").Records.sum()
    if not source_total.equals(annual.set_index("CalendarYear").Wildfire_records.rename("Records")):
        raise ValueError("Annual source totals disagree with annual counts.")
    calculated_pct = composition.Records / composition.CalendarYear.map(source_total) * 100
    if not np.allclose(composition.Percent, calculated_pct, atol=1e-12, rtol=0):
        raise ValueError("Source percentages disagree with counts.")
    raw_checks = verify_raw_reporting(annual, composition, monthly)
    years = list(range(2015, 2026))
    annual_plot = annual[annual.CalendarYear.isin(years)].copy()
    composition_plot = composition[composition.CalendarYear.isin(years)].copy()
    order = composition_plot.groupby("CreatedBySystem").Records.sum().sort_values(ascending=False).index.tolist()
    wide = composition_plot.pivot(index="CalendarYear", columns="CreatedBySystem", values="Percent").reindex(years).fillna(0)[order]
    jan_mar = float(monthly[monthly.month <= 3].percent.sum())
    save_table(annual, "supporting_reporting_annual_counts.csv")
    save_table(composition, "supporting_reporting_source_composition.csv")
    save_table(monthly, "supporting_reporting_monthly_2021_2025.csv")
    save_table(pd.DataFrame([{"study_records": total, "jan_mar_percent": jan_mar,
                             "excluded_incomplete_2026_records": int(annual.loc[annual.CalendarYear == 2026, "Wildfire_records"].iloc[0]),
                             "annual_counts_match_raw": True, "source_counts_match_raw": True,
                             "monthly_counts_match_raw": True}]), "supporting_reporting_checks.csv")

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.2), gridspec_kw={"height_ratios": [1, 1.10, 1.05]})
    fig.subplots_adjust(left=.105, right=.975, bottom=.105, top=.94, hspace=.67)
    ax, bx, cx = axes
    panel(ax, "a", "Annual record counts")
    colors = [RF_COLOR if year >= 2021 else "#A9B3BD" for year in years]
    ax.bar(years, annual_plot.Wildfire_records, color=colors, width=.7, linewidth=0, zorder=3)
    ax.axvline(2020.5, color=INK, lw=.8, linestyle="--")
    for year in [2020, 2021]:
        value = int(annual_plot.loc[annual_plot.CalendarYear == year, "Wildfire_records"].iloc[0])
        ax.text(year, value + 100, f"{value:,}", ha="center", fontsize=8, color=INK)
    ax.text(2023, 2800, "2021–2025 study period", ha="center", fontsize=8, color=RF_COLOR)
    ax.set(xticks=years, xlim=(2014.4, 2025.6), ylim=(0, 3100), ylabel="Records")
    subtle_grid(ax)
    panel(bx, "b", "Record-creation system")
    palette = [RF_COLOR, "#737B83", "#BE744E", "#9474AB", "#A69B56", "#CE919C", "#748A82", "#BBC8D2"]
    bottom = np.zeros(len(years))
    for name, color in zip(order, palette):
        bx.bar(years, wide[name], bottom=bottom, width=.72, color=color, label=name, linewidth=0, zorder=3)
        bottom += wide[name].to_numpy()
    bx.axvline(2020.5, color=INK, lw=.8, linestyle="--")
    bx.set(xticks=years, xlim=(2014.4, 2025.6), ylim=(0, 100), ylabel="Share of records (%)")
    subtle_grid(bx)
    bx.legend(loc="upper center", bbox_to_anchor=(.5, -.21), ncol=4, frameon=False,
              fontsize=6.8, handlelength=1.2, columnspacing=1.1, labelspacing=.45)
    panel(cx, "c", "Monthly occurrence, 2021–2025")
    cx.axvspan(3.5, 10.5, color="#DDEDE7", alpha=.95, zorder=0)
    cx.bar(monthly.month, monthly.percent, color=RF_COLOR, width=.68, zorder=3)
    cx.text(1, 32, f"Jan–Mar: {jan_mar:.1f}%", ha="left", fontsize=8, color=INK)
    cx.text(7, 28, "Apr–Oct NDVI window", ha="center", fontsize=8, color=RF_COLOR)
    cx.set(xticks=range(1, 13), xticklabels=[name[:3] for name in monthly.month_name],
           xlim=(.4, 12.6), ylim=(0, 35), ylabel="Study-period records (%)")
    subtle_grid(cx)
    fig.text(.5, .03, f"Downloaded-record counts, not a completeness estimate · n = {total:,} in 2021–2025\n"
             "Incomplete 2026 records are excluded from annual panels; original system labels are retained.",
             ha="center", va="bottom", fontsize=7.5, color=MUTED)
    caption = (
        "Figure S3. Reporting characteristics and seasonality of the downloaded Missouri wildfire records. "
        "(a) Annual record counts for 2015–2025, with the 2021–2025 study period highlighted. "
        "(b) Annual proportions by the recorded CreatedBySystem value, retaining source labels such as "
        "wildcad and wildcade separately. (c) Monthly proportions of the 11,434 records selected by "
        "CalendarYear 2021–2025, using the stored FireDiscoveryDateTime month. "
        f"January–March account for {jan_mar:.6f}%; the shaded April–October window is the NDVI "
        "summary period reported in the manuscript, not independently verified export metadata. "
        "All displayed counts and proportions were checked against the saved summaries and attribute-only "
        "queries of the original GeoPackage. Counts describe the downloaded records rather than true "
        "wildfire incidence or reporting completeness. The 128 incomplete-year 2026 records are omitted "
        "from the annual panels. These counts precede assignment to modeling grid cells."
    )
    save_figure(fig, "S3_reporting_seasonality", {
        "inputs": [input_record(ROOT / p) for p in paths] + [input_record(ROOT / "data/InFORM_FireOccurrence_MO.gpkg"), input_record(MANUSCRIPT)],
        "caption": caption,
        "method": "Replot saved reporting summaries; independently verify counts using read-only SQLite aggregate queries of original GeoPackage attributes. No model or spatial ETL.",
        "provenance_limits": ["Counts establish properties of downloaded records, not statewide reporting completeness.",
                              "NDVI timing is taken from the manuscript; its upstream compositing code is not present.",
                              "Original generation code for the CSV summaries is absent; the displayed descriptive counts were independently reproduced from raw attributes."],
        "checks": {**raw_checks, "percentages_match_counts": True, "study_records": total,
                   "jan_mar_percent": jan_mar, "excluded_2026_records": 128},
    })
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    s1_robustness()
    s2_shap_stability()
    s3_reporting()


if __name__ == "__main__":
    main()
