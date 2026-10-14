"""Draw manuscript Figures 1–6 from preserved data and verified results.

Usage: python scripts/make_publication_figures.py --figures 1 2 3 4 5 6
       python scripts/make_publication_figures.py --figures 4 --prepare-shap

No model is fitted by this script. The optional SHAP preparation reproduces
the original TreeExplainer call on the preserved fitted model. Existing
manuscript figures are never overwritten. PDFs contain editable vector paths,
text, and points; PNGs are exported at 600 dpi.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
import numpy as np
import pandas as pd

from project import FEATURES, PUBLICATION, REPRO, ROOT, load_model_data

ARCHIVE = "archive/pre_publication_cleanup_2026-10-03/etl.original.ipynb"
DERIVED = ROOT / "data" / "derived"
CLASS_ORDER = ["Very Low", "Low", "Moderate", "High", "Very High"]
CLASS_COLORS = ["#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c", "#800026"]
FOLD_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377"]
INK = "#263442"
MUTED = "#5e6872"
FIGURE_NAMES = {
    1: "figure1_study_area_spatial_folds",
    2: "figure2_workflow",
    3: "figure3_model_validation",
    4: "figure4_shap_interpretation",
    5: "figure5_relative_susceptibility",
    6: "figure6_nested_oof_assessment",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return str(Path(path).relative_to(ROOT))


def input_record(path):
    return {"path": relative(path), "sha256": sha256(path)}


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def load_grid():
    path = ROOT / "data/Missouri_Wildfire_Susceptibility_5km.gpkg"
    grid = gpd.read_file(path, layer="wildfire_susceptibility")
    if len(grid) != 7229 or grid.grid_id.nunique() != 7229:
        raise ValueError("Expected 7,229 unique saved grid cells.")
    if grid.crs.to_epsg() != 5070 or grid.geometry.isna().any() or grid.geometry.is_empty.any():
        raise ValueError("Saved grid geometry/CRS differs from the original.")
    if set(grid.susceptibility_class) != set(CLASS_ORDER):
        raise ValueError("Saved susceptibility classes differ.")
    return grid, path


def prepare_map_helpers():
    """Export only study-year point coordinates and a geometry-only boundary.

    Coordinate CSV uses EPSG:5070 explicitly. Boundary GeoJSON uses standard
    longitude/latitude (EPSG:4326) and is projected to 5070 when read.
    Existing helpers are checked against their recorded checksums before use.
    """
    DERIVED.mkdir(parents=True, exist_ok=True)
    points_path = DERIVED / "recorded_wildfire_locations_2021_2025.csv"
    boundary_path = DERIVED / "missouri_boundary.geojson"
    metadata_path = DERIVED / "figure1_input_provenance.json"
    if not all(p.exists() for p in [points_path, boundary_path, metadata_path]):
        raw_fires = ROOT / "data/InFORM_FireOccurrence_MO.gpkg"
        raw_boundary = ROOT / "data/MO_State_Boundary/MO_State_Boundary.shp"
        if not raw_fires.exists() or not raw_boundary.exists():
            raise FileNotFoundError("Figure 1 needs the derived helpers or original fire/boundary inputs.")
        fires = gpd.read_file(raw_fires, layer="Incidents",
                              columns=["CalendarYear"])
        fires = fires.loc[fires.CalendarYear.between(2021, 2025)].to_crs(5070)
        if len(fires) != 11434 or not (fires.geom_type == "Point").all():
            raise ValueError("Study-period point count/type differs from original notebook.")
        locations = pd.DataFrame({"CalendarYear": fires.CalendarYear.astype(int),
                                  "x": fires.geometry.x, "y": fires.geometry.y})
        if not np.isfinite(locations[["x", "y"]]).all().all():
            raise ValueError("Fire coordinates are missing/nonfinite.")
        boundary = gpd.read_file(raw_boundary)[["geometry"]].to_crs(4326)
        if len(boundary) != 1 or not boundary.geometry.is_valid.all():
            raise ValueError("Unexpected Missouri boundary geometry.")
        locations.to_csv(points_path, index=False)
        boundary.to_file(boundary_path, driver="GeoJSON", RFC7946="YES")
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_notebook": ARCHIVE, "source_cells_1_based": [68, 69, 150],
            "source_inputs": [input_record(raw_fires), input_record(raw_boundary)],
            "filter": "CalendarYear.between(2021, 2025)",
            "location_columns": ["CalendarYear", "x", "y"],
            "incident_names_identifiers_and_other_attributes_exported": False,
            "location_crs": "EPSG:5070", "location_count": len(locations),
            "boundary_crs": "EPSG:4326", "boundary_feature_count": len(boundary),
            "outputs": [input_record(points_path), input_record(boundary_path)],
        }
        write_json(metadata_path, metadata)
    metadata = json.loads(metadata_path.read_text())
    for entry in metadata["outputs"]:
        if sha256(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Figure 1 helper checksum changed: {entry['path']}")
    locations = read_csv(points_path)
    if list(locations.columns) != ["CalendarYear", "x", "y"] or len(locations) != 11434:
        raise ValueError("Unexpected derived Figure 1 point schema/count.")
    if not locations.CalendarYear.between(2021, 2025).all():
        raise ValueError("Derived points contain an out-of-period record.")
    boundary = gpd.read_file(boundary_path).to_crs(5070)
    return locations, boundary, [points_path, boundary_path, metadata_path]


def panel(ax, letter, title):
    ax.set_title(f"({letter})  {title}", loc="left", pad=10, fontweight="semibold")


def map_extent(boundary):
    x0, y0, x1, y1 = boundary.total_bounds
    pad = (x1 - x0) * .055
    return (x0 - pad, x1 + pad, y0 - pad, y1 + pad)


def map_finish(ax, extent, scale=True, north=True):
    x0, x1, y0, y1 = extent
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if scale:
        # Coordinates are metres in EPSG:5070; scale is drawn in data units.
        bx, by = x0 + .06 * (x1 - x0), y0 + .035 * (y1 - y0)
        height = .006 * (y1 - y0)
        for i, color in enumerate([INK, "white"]):
            ax.add_patch(Rectangle((bx + i * 50000, by), 50000, height,
                                   facecolor=color, edgecolor=INK, linewidth=.55))
        for distance in [0, 50, 100]:
            ax.text(bx + distance * 1000, by + height * 2.0,
                    str(distance), ha="center", va="bottom", fontsize=7)
        ax.text(bx + 109000, by + height * .4, "km", fontsize=7, va="center")
    if north:
        ax.text(.92, .98, "N", transform=ax.transAxes, ha="center", va="center",
                fontsize=8, fontweight="semibold")
        ax.annotate("", xy=(.92, .95), xytext=(.92, .87),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=.7, color=INK))


def figure1():
    grid, grid_path = load_grid()
    locations, boundary, helper_paths = prepare_map_helpers()
    fold_path = REPRO / "outer_fold_assignments.csv"
    if not fold_path.exists():
        raise FileNotFoundError("Create outer_fold_assignments.csv with reproduce_validation.py first.")
    folds = read_csv(fold_path)
    frame, _, _, _ = load_model_data()
    if (len(folds) != 7229 or not np.array_equal(folds.grid_id, frame.grid_id)
            or set(folds.outer_fold) != {1, 2, 3, 4, 5}
            or folds.groupby("block_id").outer_fold.nunique().max() != 1):
        raise ValueError("Outer assignments do not match saved cells and intact blocks.")
    grid = grid.merge(folds[["grid_id", "block_id", "outer_fold"]],
                      on="grid_id", validate="one_to_one")
    if grid.outer_fold.isna().any() or grid.block_id.nunique() != 94:
        raise ValueError("Missing fold assignment or changed spatial block count.")
    blocks = grid[["block_id", "geometry"]].dissolve(by="block_id")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.35))
    fig.subplots_adjust(left=.025, right=.975, bottom=.12, top=.90, wspace=.045)
    grid.plot(ax=axes[0], facecolor="#f8f9fa", edgecolor="#c7ccd1", linewidth=.10)
    axes[0].scatter(locations.x, locations.y, s=.85, c="#b54b31", alpha=.65,
                    linewidths=0, rasterized=False)
    boundary.boundary.plot(ax=axes[0], color=INK, linewidth=.7)
    panel(axes[0], "a", "Recorded wildfire, 2021–2025")
    axes[0].legend(handles=[Line2D([], [], marker="o", linestyle="", color="#b54b31",
                                  markersize=3, label="Recorded wildfire"),
                            Patch(facecolor="#f8f9fa", edgecolor="#b2bac2", linewidth=.6,
                                  label="5-km modeling cell")],
                    loc="upper center", bbox_to_anchor=(.5, -.035), frameon=False,
                    ncol=2, columnspacing=1.0, handletextpad=.5)
    for fold, color in enumerate(FOLD_COLORS, 1):
        grid.loc[grid.outer_fold == fold].plot(ax=axes[1], color=color, linewidth=0)
    blocks.boundary.plot(ax=axes[1], color="#ffffff", linewidth=.35, alpha=.9)
    boundary.boundary.plot(ax=axes[1], color=INK, linewidth=.7)
    panel(axes[1], "b", "Spatial blocks and outer folds")
    axes[1].legend(handles=[Patch(facecolor=c, label=str(i))
                            for i, c in enumerate(FOLD_COLORS, 1)],
                    title="Outer fold", loc="upper center", bbox_to_anchor=(.5, -.005),
                    frameon=False, ncol=5, columnspacing=.7, handlelength=1.0,
                    handletextpad=.3, title_fontsize=8)
    for ax in axes:
        map_finish(ax, map_extent(boundary))
    return fig, {"inputs": [input_record(p) for p in [grid_path, fold_path, *helper_paths]],
                 "source_cells_1_based": [68, 69, 98, 134, 150],
                 "checks": {"cells": len(grid), "recorded_fire_locations": len(locations),
                            "spatial_blocks": len(blocks),
                            "fold_sizes": folds.outer_fold.value_counts().sort_index().to_dict(),
                            "projection": "EPSG:5070", "matching_panel_extents": True},
                 "notes": "Original outer-fold IDs preserved; drawn block boundaries are unions of saved modeling cells grouped by original block_id. No new split or point filtering beyond CalendarYear 2021–2025."}


def flow_box(ax, x, y, width, height, title, text, fill="#f2f5f7", title_size=9):
    patch = FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                           boxstyle="round,pad=0.008,rounding_size=0.012",
                           facecolor=fill, edgecolor="#a1acb5", linewidth=.75)
    ax.add_patch(patch)
    ax.text(x, y + height * .22, title, ha="center", va="center", fontsize=title_size,
            fontweight="semibold")
    ax.text(x, y - height * .15, text, ha="center", va="center", fontsize=8,
            linespacing=1.4)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=.85, color=MUTED, shrinkA=2, shrinkB=2))


def figure2():
    fig, ax = plt.subplots(figsize=(7.2, 6.7))
    fig.subplots_adjust(left=.015, right=.985, bottom=.015, top=.99)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    flow_box(ax, .245, .93, .43, .10, "Wildfire occurrence",
             "Recorded locations, 2021–2025")
    flow_box(ax, .755, .93, .43, .10, "Predictor datasets",
             "Terrain · vegetation · climate\nHuman influence · fuels")
    arrow(ax, (.245, .87), (.40, .827)); arrow(ax, (.755, .87), (.60, .827))
    flow_box(ax, .5, .762, .80, .116, "Common modeling dataset",
             "7,229 cells on a 5-km grid · 33 encoded predictors\nRecorded wildfire / no recorded wildfire", fill="#e7eff5")
    arrow(ax, (.36, .697), (.245, .646)); arrow(ax, (.64, .697), (.755, .646))
    flow_box(ax, .245, .566, .43, .142, "Validation benchmarks",
             "RF: random and grouped spatial CV\nRF + XGBoost: tuned spatial CV\nFive validation folds")
    flow_box(ax, .755, .566, .43, .142, "Nested spatial cross-validation",
             "Shared splits: 5 outer / 4 inner folds\n50-km groups · inner parameter tuning\nHeld-out Spatial Evaluation: outer folds", title_size=8.8)
    arrow(ax, (.245, .485), (.245, .448)); arrow(ax, (.755, .485), (.755, .448))
    flow_box(ax, .245, .382, .43, .113, "Validation comparison",
             "Mean ROC-AUC ± fold SD\nRandom versus spatial performance")
    flow_box(ax, .755, .382, .43, .113, "Model-family comparison",
             "Compare held-out RF and XGBoost\nouter-fold performance; retain RF")
    arrow(ax, (.755, .317), (.59, .268))
    flow_box(ax, .5, .212, .80, .095, "Final Random Forest",
             "Use original full-data tuning result; fit all 7,229 cells", fill="#e7eff5")
    arrow(ax, (.36, .156), (.245, .121)); arrow(ax, (.64, .156), (.755, .121))
    flow_box(ax, .245, .066, .43, .090, "SHAP interpretation",
             "Class-1 feature associations", fill="#edf4ec")
    flow_box(ax, .755, .066, .43, .090, "Relative susceptibility mapping",
             "Model score and five ordered classes", fill="#fff3df", title_size=8.7)
    return fig, {"inputs": [], "source_cells_1_based": [89, 94, 97, 100, 102, 103, 104, 110, 114, 119, 122, 134, 135],
                 "checks": {"all_elements_are_vector": True, "response_labels": ["Recorded wildfire", "No recorded wildfire"]},
                 "notes": "Workflow redraw documents the original RF final-fit provenance; nested tuning and model-family comparison are separate. Nested XGBoost is reconstructed in scripts/reproduce_validation.py. No claim of calibrated wildfire probability."}


def figure3():
    path = REPRO / "model_evaluation_summary.csv"
    summary = read_csv(path)
    required = [("Random Forest", "Random 5-fold CV"),
                ("Random Forest", "Grouped spatial 5-fold CV"),
                ("Random Forest", "Tuned spatial 5-fold CV"),
                ("XGBoost", "Tuned spatial 5-fold CV"),
                ("Random Forest", "Nested spatial 5-fold CV"),
                ("XGBoost", "Nested spatial 5-fold CV")]
    selected = summary.set_index(["model", "validation"]).loc[required].reset_index()
    if selected[["mean_auc", "sd_auc"]].isna().any().any():
        raise ValueError("Figure 3 needs complete mean/SD results; run reproduce_validation.py first.")
    verified = selected.verified.astype(str).str.lower().eq("true")
    if not verified.all():
        raise ValueError("Figure 3 requires verified results for all six analyses; validation is still pending.")
    fig = plt.figure(figsize=(7.2, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.2], left=.31, right=.98,
                         top=.89, bottom=.19, wspace=.05)
    ax = fig.add_subplot(gs[0]); numbers = fig.add_subplot(gs[1], sharey=ax)
    labels = ["RF · Random CV", "RF · Grouped spatial CV", "RF · Tuned spatial CV",
              "XGBoost · Tuned spatial CV", "RF · Nested spatial CV", "XGBoost · Nested spatial CV"]
    y = np.arange(6)
    for i, row in selected.iterrows():
        color, marker = ("#35688f", "o") if row.model == "Random Forest" else ("#bc622d", "s")
        ax.errorbar(row.mean_auc, i, xerr=row.sd_auc, fmt=marker, markersize=5,
                     color=color, ecolor=color, capsize=3, elinewidth=1.1,
                     markeredgecolor="white", markeredgewidth=.5)
        numbers.text(.5, i, f"{row.mean_auc:.3f} ± {row.sd_auc:.3f}",
                      ha="center", va="center", fontsize=9, color=color)
    ax.set_yticks(y, labels); ax.set_ylim(5.55, -.55)
    ax.set_xlim(.58, .81); ax.set_xticks([.60, .65, .70, .75, .80])
    ax.set_xlabel("ROC-AUC"); ax.tick_params(axis="y", length=0, pad=9)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#e2e5e8", linewidth=.6); ax.set_axisbelow(True)
    for divider in [1.5, 3.5]:
        ax.axhline(divider, color="#e5e8eb", lw=.7)
        numbers.axhline(divider, color="#e5e8eb", lw=.7)
    numbers.set_xlim(0, 1); numbers.axis("off")
    numbers.set_title("Mean ± SD", fontsize=9, pad=10)
    fig.text(.31, .07, "Five validation folds · sample standard deviation (ddof = 1)",
             fontsize=8, color=MUTED)
    return fig, {"inputs": [input_record(path)], "source_cells_1_based": [102, 103, 106, 110, 135],
                 "checks": {"all_six_rows_verified": True, "sd_ddof": 1,
                            "plotted_values": selected.to_dict(orient="records")},
                 "notes": "Points and horizontal whiskers show fold mean ROC-AUC ± sample SD. Values are read at full precision from the verified summary, with display rounded to three decimals."}


def checked_shap(prepare=False):
    values_path = REPRO / "rf_shap_values.npz"
    metadata_path = REPRO / "rf_shap_metadata.json"
    if prepare:
        from prepare_shap import main as prepare_main, REPRO as NEW_SHAP
        prepare_main()
        values_path = NEW_SHAP / "rf_shap_values.npz"
        metadata_path = NEW_SHAP / "rf_shap_metadata.json"
    if not values_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Preserved SHAP artifacts are required; verify the extracted package.")
    metadata = json.loads(metadata_path.read_text())
    if metadata["output_sha256"] != sha256(values_path):
        raise ValueError("SHAP cache checksum changed; inspect or explicitly re-prepare it.")
    for name, expected in metadata["input_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"SHAP source changed: {name}; cached values must not be reused.")
    frame, X, _, _ = load_model_data()
    with np.load(values_path, allow_pickle=False) as cache:
        if cache["feature_names"].tolist() != FEATURES or not np.array_equal(cache["grid_id"], frame.grid_id):
            raise ValueError("SHAP feature/cell order mismatch.")
        if not np.array_equal(cache["X"], X.to_numpy()):
            raise ValueError("SHAP predictor values differ from current data.")
        values = cache["shap_class1"].copy()
        if values.shape != (7229, 33) or metadata["classes"] != [0, 1] or metadata["interpreted_class"] != 1:
            raise ValueError("SHAP dimensions or interpreted output changed.")
    return X, values, metadata, values_path, metadata_path


DISPLAY_NAMES = {
    "population": "Population", "ndvi": "NDVI", "canopy": "Tree canopy cover",
    "precipitation": "Precipitation", "road_distance": "Distance to roads",
    "elevation": "Elevation", "temperature": "Temperature", "wind": "Wind speed",
    "slope": "Slope", "pdsi": "PDSI", "northness": "Northness", "eastness": "Eastness",
    "lc_41": "Deciduous forest\n(LC 41 indicator)",
    "lc_82": "Cultivated crops\n(LC 82 indicator)",
    "fuel_93": "Agriculture\n(NB3 fuel indicator)",
}


def beeswarm_offsets(values, seed, bins=80):
    """Deterministic binned jitter for display only; all x-values are unchanged."""
    rng = np.random.default_rng(seed)
    edges = np.linspace(values.min(), values.max() + np.finfo(float).eps, bins + 1)
    assignments = np.clip(np.digitize(values, edges) - 1, 0, bins - 1)
    counts = np.bincount(assignments, minlength=bins)
    offsets = np.zeros(len(values))
    for bin_id in np.flatnonzero(counts):
        indices = np.flatnonzero(assignments == bin_id)
        height = .31 * np.sqrt(len(indices) / counts.max())
        local = np.linspace(-height, height, len(indices))
        rng.shuffle(local); offsets[indices] = local
    return offsets


def figure4(prepare=False):
    X, values, metadata, values_path, metadata_path = checked_shap(prepare)
    order = np.argsort(np.abs(values).mean(axis=0))[-15:][::-1]
    fig = plt.figure(figsize=(7.4, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.23, 1], height_ratios=[1, 1],
                         left=.245, right=.98, top=.90, bottom=.105, wspace=.48, hspace=.48)
    summary_ax = fig.add_subplot(gs[:, 0])
    ndvi_ax = fig.add_subplot(gs[0, 1]); canopy_ax = fig.add_subplot(gs[1, 1])
    cmap = plt.get_cmap("coolwarm")
    for row, feature_index in enumerate(order):
        raw = X.iloc[:, feature_index].to_numpy(float)
        lo, hi = np.nanpercentile(raw, [5, 95])
        if hi <= lo: lo, hi = raw.min(), raw.max()
        if hi <= lo: hi = lo + 1
        colors = np.clip((raw - lo) / (hi - lo), 0, 1)
        offsets = beeswarm_offsets(values[:, feature_index], seed=4200 + feature_index)
        summary_ax.scatter(values[:, feature_index], row + offsets, c=colors, cmap=cmap,
                            vmin=0, vmax=1, s=2.7, linewidths=0, alpha=.62,
                            rasterized=False)
    summary_ax.set_yticks(np.arange(15), [DISPLAY_NAMES.get(FEATURES[i], FEATURES[i]) for i in order])
    summary_ax.set_ylim(14.65, -.65); summary_ax.tick_params(axis="y", length=0, pad=6, labelsize=7.5)
    summary_ax.set_xlabel("SHAP value\n(susceptibility score)", fontsize=8)
    summary_ax.axvline(0, color="#9da5ad", linewidth=.7, zorder=0)
    summary_ax.grid(axis="y", color="#eef0f2", linewidth=.55); summary_ax.set_axisbelow(True)
    summary_ax.spines["left"].set_visible(False)
    panel(summary_ax, "a", "SHAP summary")
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    cax = fig.add_axes([.272, .966, .23, .011])
    colorbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 1])
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.ax.set_xticklabels(["Low", "High"]); colorbar.ax.tick_params(length=0, labelsize=7, pad=1)
    colorbar.outline.set_visible(False)
    fig.text(.386, .991, "Feature value", ha="center", va="top", fontsize=7.5)
    fig.text(.025, .015, "Indicators: low = absent; high = present.",
             ha="left", va="bottom", fontsize=7.2, color=MUTED)
    for ax, name, letter in [(ndvi_ax, "ndvi", "b"), (canopy_ax, "canopy", "c")]:
        index = FEATURES.index(name)
        ax.scatter(X[name], values[:, index], s=4, alpha=.35, c="#376c96",
                   linewidths=0, rasterized=False)
        ax.axhline(0, color="#9da5ad", linewidth=.7, zorder=0)
        ax.set_xlabel(DISPLAY_NAMES[name], fontsize=8)
        ax.set_ylabel("SHAP value\n(susceptibility score)", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(color="#edf0f2", linewidth=.5); ax.set_axisbelow(True)
        panel(ax, letter, DISPLAY_NAMES[name])
    return fig, {"inputs": [input_record(values_path), input_record(metadata_path)],
                 "source_cells_1_based": [114, 115, 116, 117, 144, 145, 146],
                 "checks": {"cells": len(X), "features": len(FEATURES),
                            "interpreted_class": 1, "top_15": [FEATURES[i] for i in order],
                            "maximum_importance_difference": metadata["maximum_importance_difference"],
                            "maximum_additivity_error": metadata["maximum_additivity_error"]},
                 "notes": "All 7,229 class-1 SHAP values plotted for each displayed feature. Horizontal coordinates are unchanged; deterministic vertical jitter is display-only. No units are asserted for raw canopy or NDVI. Feature colors use per-feature 5th/95th percentile limits, clipped for color only. LC and fuel indicators are labeled explicitly."}


def figure5():
    grid, grid_path = load_grid()
    _, boundary, helper_paths = prepare_map_helpers()
    fig, ax = plt.subplots(figsize=(7.2, 6.05))
    fig.subplots_adjust(left=.025, right=.975, top=.985, bottom=.11)
    for label, color in zip(CLASS_ORDER, CLASS_COLORS):
        grid.loc[grid.susceptibility_class == label].plot(ax=ax, color=color, linewidth=0)
    boundary.boundary.plot(ax=ax, color=INK, linewidth=.65)
    map_finish(ax, map_extent(boundary))
    ax.legend(handles=[Patch(facecolor=c, edgecolor="#a3a3a3", linewidth=.35, label=l)
                        for l, c in zip(CLASS_ORDER, CLASS_COLORS)],
              loc="upper center", bbox_to_anchor=(.5, -.022), ncol=5,
              title="Relative susceptibility", frameon=False, title_fontsize=9,
              columnspacing=1.3, handlelength=1.3, handletextpad=.5)
    return fig, {"inputs": [input_record(grid_path), input_record(helper_paths[1])],
                 "source_cells_1_based": [119, 120, 122, 127, 147],
                 "checks": {"cells": len(grid), "class_order": CLASS_ORDER,
                            "class_counts": grid.susceptibility_class.value_counts().reindex(CLASS_ORDER).to_dict(),
                            "projection": "EPSG:5070", "scores_or_classes_recomputed": False},
                 "notes": "Saved rf_susc scores and susceptibility_class assignments are unchanged. Five equal-frequency classes are relative rankings, not calibrated probabilities."}


def figure6():
    path = ROOT / "results/table4_nested_oof_susceptibility_validation.csv"
    table = read_csv(path).set_index("nested_oof_class").loc[CLASS_ORDER].reset_index()
    expected = np.array([19.71, 34.30, 42.42, 54.01, 56.22])
    if not np.array_equal(table.Fire_Positive_Cells_Pct.to_numpy(), expected):
        raise ValueError("Saved nested OOF class percentages differ from manuscript.")
    if table.Cells.sum() != 7229:
        raise ValueError("Nested OOF validation table has changed cell total.")
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    fig.subplots_adjust(left=.11, right=.985, top=.97, bottom=.19)
    bars = ax.bar(np.arange(5), table.Fire_Positive_Cells_Pct, width=.62,
                  color=CLASS_COLORS, edgecolor="#666666", linewidth=.45)
    for bar, value in zip(bars, table.Fire_Positive_Cells_Pct):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.15,
                f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(5), CLASS_ORDER)
    ax.set_xlabel("Nested out-of-fold susceptibility class", labelpad=11)
    ax.set_ylabel("Grid cells with recorded wildfire (%)")
    ax.set_ylim(0, 65); ax.set_yticks(np.arange(0, 61, 10))
    ax.grid(axis="y", color="#e5e8ea", linewidth=.6); ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=7)
    return fig, {"inputs": [input_record(path)], "source_cells_1_based": [142, 148],
                 "checks": {"cells": int(table.Cells.sum()), "class_order": CLASS_ORDER,
                            "percentages": table.Fire_Positive_Cells_Pct.tolist(),
                            "class_colors_match_figure5": True},
                 "notes": "Plots preserved table percentages; no classification or validation analysis is rerun. Colors encode the same ordered class names as Figure 5, while values are from held-out nested OOF scores."}


def save_figure(number, fig, detail):
    PUBLICATION.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_NAMES[number]
    png_path = PUBLICATION / f"{stem}.png"
    pdf_path = PUBLICATION / f"{stem}.pdf"
    fig.savefig(pdf_path, format="pdf", metadata={"Title": stem.replace("_", " "),
                                                  "Creator": "Matplotlib / make_publication_figures.py"})
    fig.savefig(png_path, dpi=600)
    width, height = fig.get_size_inches()
    plt.close(fig)
    detail.update({"figure": number, "function": f"figure{number}",
                   "source_notebook": ARCHIVE,
                   "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "size_inches": [float(width), float(height)], "png_dpi": 600,
                   "png_size_pixels": [round(width * 600), round(height * 600)],
                   "pdf_artists_rasterized": False,
                   "outputs": [input_record(png_path), input_record(pdf_path)]})
    manifest_path = PUBLICATION / "publication_figure_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"figures": {}}
    manifest.update({"generator": "scripts/make_publication_figures.py",
                     "generator_sha256": sha256(Path(__file__)),
                     "original_figure_files_overwritten": False,
                     "model_fitting_performed": False,
                     "class_palette": dict(zip(CLASS_ORDER, CLASS_COLORS))})
    manifest["figures"][str(number)] = detail
    write_json(manifest_path, manifest)
    print(f"Figure {number}: {relative(png_path)} + {relative(pdf_path)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=int, nargs="+", choices=range(1, 7), default=list(range(1, 7)))
    parser.add_argument("--prepare-shap", action="store_true",
                        help="Explicitly recompute SHAP from the saved RF before Figure 4; never fits a model.")
    args = parser.parse_args()
    for number in args.figures:
        print(f"Preparing Figure {number}…", flush=True)
        result = figure4(args.prepare_shap) if number == 4 else globals()[f"figure{number}"]()
        save_figure(number, *result)


if __name__ == "__main__":
    main()
