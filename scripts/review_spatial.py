"""Create spatial review figures from preserved values, without fitting models.

All exports go to figures/review_2026-10-08/. Existing figures, data, models,
and manuscript documents are never edited. Population dependence uses an
explicit log(1 + x) display axis while retaining every original predictor and
SHAP value; no predictor units are inferred from numeric ranges.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter

from review_common import (
    ROOT, OUT, DATA, FEATURES, load_model_data, read_csv, input_record,
    save_figure, save_table, load_grid, load_boundary, finish_map, panel,
    INK, MUTED, RF_COLOR,
)
from make_publication_figures import (
    checked_shap, CLASS_ORDER, CLASS_COLORS, FOLD_COLORS,
)


GRID_PATH = ROOT / "data/Missouri_Wildfire_Susceptibility_5km.gpkg"
BOUNDARY_PATH = ROOT / "data/derived/missouri_boundary.geojson"
SHAP_NAMES = ["population", "ndvi", "canopy"]
SHAP_LABELS = ["Population", "NDVI", "Tree canopy cover"]


def export_figure(fig, stem, detail):
    """Retain only ticks inside each axis before the shared text-bound check.

    Matplotlib locators may create extra, undrawn ticks outside view limits.
    Fixing the visible tick locations does not alter data or the axis limits.
    """
    for ax in fig.axes:
        if not ax.axison:
            continue
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            low, high = sorted(limits)
            ticks = axis.get_majorticklocs()
            axis.set_major_locator(FixedLocator([t for t in ticks if low <= t <= high]))
    return save_figure(fig, stem, detail)


def shap_inputs():
    """Read the existing, hash-checked full-data explanation; never recompute it."""
    for path in [ROOT / "results/reproducibility/rf_shap_values.npz",
                 ROOT / "results/reproducibility/rf_shap_metadata.json"]:
        if not path.exists():
            raise FileNotFoundError(f"Required preserved SHAP artifact is missing: {path}")
    X, values, metadata, cache_path, metadata_path = checked_shap(False)
    frame, original_X, _, _ = load_model_data()
    if not np.array_equal(X.to_numpy(), original_X.to_numpy()):
        raise ValueError("SHAP predictor matrix differs from saved modeling values.")
    if values.shape != (7229, 33) or not np.isfinite(values).all():
        raise ValueError("Unexpected class-1 SHAP shape or nonfinite values.")
    return frame, X, values, metadata, [cache_path, metadata_path]


def spatial_shap():
    frame, X, values, metadata, shap_paths = shap_inputs()
    grid = load_grid()
    boundary = load_boundary()
    selected = [FEATURES.index(name) for name in SHAP_NAMES]
    data = frame[["grid_id"]].copy()
    for feature, index in zip(SHAP_NAMES, selected):
        data[f"shap_{feature}"] = values[:, index]
    geometry = grid[["grid_id", "geometry"]].merge(data, on="grid_id", validate="one_to_one")
    if len(geometry) != 7229 or geometry[[f"shap_{n}" for n in SHAP_NAMES]].isna().any().any():
        raise ValueError("Spatial SHAP join lost a cell or attribution.")
    # Export numeric join coordinates and contributions, leaving source files intact.
    numeric = geometry.drop(columns="geometry").copy()
    centroids = geometry.geometry.centroid
    numeric.insert(1, "centroid_x_epsg5070", centroids.x)
    numeric.insert(2, "centroid_y_epsg5070", centroids.y)
    table_name = "spatial_shap_contributions.csv"
    save_table(numeric, table_name)
    maximum = float(np.max(np.abs(values[:, selected])))
    limit = float(np.ceil(maximum * 100) / 100)
    norm = Normalize(vmin=-limit, vmax=limit)
    cmap = plt.get_cmap("RdBu_r")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.5))
    fig.subplots_adjust(left=.095, right=.975, bottom=.075, top=.94,
                        wspace=.29, hspace=.28)
    for ax, feature, label, letter in zip(axes.flat[:3], SHAP_NAMES, SHAP_LABELS, "abc"):
        geometry.plot(column=f"shap_{feature}", ax=ax, cmap=cmap, norm=norm,
                       linewidth=0, rasterized=False)
        boundary.boundary.plot(ax=ax, color=INK, linewidth=.55)
        finish_map(ax, boundary)
        panel(ax, letter, label)
    legend_ax = axes.flat[3]
    legend_ax.set_axis_off()
    legend_ax.text(.05, .91, "Full-data RF attribution",
                   transform=legend_ax.transAxes, fontsize=10, fontweight="semibold",
                   ha="left", va="top")
    legend_ax.text(.05, .77,
                   "Each map shows the contribution of one\npredictor to the class-1 susceptibility score.\n\n"
                   "Blue: lower score\nRed: higher score\n\n"
                   "All maps use the same zero-centered scale.\n7,229 modeled cells; original values retained.",
                   transform=legend_ax.transAxes, fontsize=8.4, linespacing=1.65,
                   ha="left", va="top", color=MUTED)
    cax = legend_ax.inset_axes([.05, .15, .88, .075])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                        orientation="horizontal", ticks=[-limit, 0, limit])
    if cbar.solids is not None:
        cbar.solids.set_rasterized(False)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Signed SHAP contribution", fontsize=8.5, labelpad=7)
    cbar.outline.set_linewidth(.5)
    detail = {
        "inputs": [input_record(p) for p in [GRID_PATH, BOUNDARY_PATH, *shap_paths]],
        "supporting_data": [input_record(DATA / table_name)],
        "rows": 7229, "features": SHAP_NAMES, "interpreted_class": 1,
        "source_values_changed": False, "models_fitted": 0,
        "explanation_scope": "Preserved final RF fitted to all cells; not held-out explanations.",
        "common_color_limits": [-limit, limit], "maximum_absolute_value": maximum,
        "values_clipped_for_color": False, "map_crs": "EPSG:5070",
        "shap_additivity_error_from_verification": metadata["maximum_additivity_error"],
        "caption": "Spatial distribution of signed class-1 SHAP contributions from population, NDVI, and canopy in the final full-data Random Forest. Positive and negative values shift the susceptibility score above or below its expected value. A common symmetric color scale is used. Contributions are fitted-model associations, not causal effects or held-out uncertainty estimates.",
    }
    export_figure(fig, "A2_spatial_shap", detail)


def population_road_dependence():
    frame, X, values, _, shap_paths = shap_inputs()
    features = ["population", "road_distance"]
    labels = ["Population predictor", "Distance-to-roads predictor"]
    table = frame[["grid_id"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.05), sharey=True)
    fig.subplots_adjust(left=.115, right=.975, bottom=.29, top=.88, wspace=.18)
    for ax, feature, label, letter in zip(axes, features, labels, "ab"):
        index = FEATURES.index(feature)
        raw_x = X[feature].to_numpy()
        if not np.isfinite(raw_x).all():
            raise ValueError("Dependence plot predictor contains nonfinite values.")
        table[feature] = raw_x
        table[f"shap_{feature}"] = values[:, index]
        ax.scatter(raw_x, values[:, index], s=4.5, c=RF_COLOR,
                    alpha=.35, linewidths=0, rasterized=False)
        ax.axhline(0, color="#949fa8", linewidth=.7, zorder=0)
        ax.set_xlabel(label, labelpad=9)
        ax.grid(color="#e8ecef", linewidth=.55)
        ax.set_axisbelow(True)
        panel(ax, letter, "Population" if feature == "population" else "Road accessibility")
        if feature == "population":
            if (raw_x < 0).any():
                raise ValueError("log1p population display requires nonnegative predictor values.")
            ax.set_xscale("function", functions=(np.log1p, np.expm1))
            ticks = [0, 10, 100, 1000, 10000, 50000]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
            ax.set_xlim(0, max(raw_x) * 1.05)
            ax.tick_params(axis="x", labelsize=7.5, rotation=25)
    axes[0].set_ylabel("SHAP contribution to susceptibility score")
    fig.text(.115, .065, "Population axis: log(1 + value) spacing; tick labels show original values.",
             fontsize=7.6, color=MUTED)
    fig.text(.115, .018, "All 7,229 cells shown. Predictor units are not inferred from the export.",
             fontsize=7.6, color=MUTED)
    table_name = "spatial_population_road_dependence.csv"
    save_table(table, table_name)
    detail = {
        "inputs": [input_record(p) for p in shap_paths],
        "supporting_data": [input_record(DATA / table_name)],
        "rows": 7229, "features": features, "interpreted_class": 1,
        "source_values_changed": False, "models_fitted": 0,
        "population_display_transform": "log(1 + raw x), with raw-value tick labels; all points retained",
        "road_display_transform": "none", "predictor_units": "not verified from source export",
        "caption": "Dependence of the preserved final RF's class-1 SHAP contributions on population and road-distance predictors. Every original cell and SHAP value is shown. Population uses log(1 + value) horizontal spacing to display its skewed distribution; road distance is linear. Axes do not assert unverified physical units. These are full-data fitted-model associations.",
    }
    export_figure(fig, "S5_population_road_dependence", detail)


def study_area_coordinates():
    grid = load_grid()
    boundary = load_boundary()
    points_path = ROOT / "data/derived/recorded_wildfire_locations_2021_2025.csv"
    folds_path = ROOT / "results/reproducibility/outer_fold_assignments.csv"
    points = read_csv(points_path)
    folds = read_csv(folds_path)
    if len(points) != 11434 or not points.CalendarYear.between(2021, 2025).all():
        raise ValueError("Unexpected study-period wildfire locations.")
    if len(folds) != 7229 or set(folds.outer_fold) != {1, 2, 3, 4, 5}:
        raise ValueError("Unexpected original outer-fold assignments.")
    grid = grid.merge(folds[["grid_id", "block_id", "outer_fold"]], on="grid_id", validate="one_to_one")
    if grid.outer_fold.isna().any() or grid.block_id.nunique() != 94:
        raise ValueError("Missing fold data or changed spatial groups.")
    if grid.groupby("block_id").outer_fold.nunique().max() != 1:
        raise ValueError("Spatial group split between outer folds.")
    blocks = grid[["block_id", "geometry"]].dissolve(by="block_id")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.7))
    fig.subplots_adjust(left=.09, right=.975, bottom=.235, top=.9, wspace=.22)
    grid.plot(ax=axes[0], facecolor="#f8f9fa", edgecolor="#c7ccd1", linewidth=.10)
    axes[0].scatter(points.x, points.y, s=.85, c="#b54b31", alpha=.65,
                    linewidths=0, rasterized=False)
    for fold, color in enumerate(FOLD_COLORS, 1):
        grid.loc[grid.outer_fold == fold].plot(ax=axes[1], color=color, linewidth=0)
    blocks.boundary.plot(ax=axes[1], color="white", linewidth=.35, alpha=.9)
    for ax in axes:
        boundary.boundary.plot(ax=ax, color=INK, linewidth=.65)
        finish_map(ax, boundary)
    panel(axes[0], "a", "Recorded wildfire, 2021–2025")
    panel(axes[1], "b", "Spatial blocks and outer folds")
    left_handles = [Line2D([], [], marker="o", linestyle="", color="#b54b31",
                           markersize=3, label="Recorded wildfire"),
                    Patch(facecolor="#f8f9fa", edgecolor="#b2bac2", linewidth=.6,
                          label="5-km modeling cell")]
    fig.legend(handles=left_handles, loc="lower center", bbox_to_anchor=(.29, .045),
                frameon=False, fontsize=7.5, ncol=1, handlelength=1.3)
    fig.legend(handles=[Patch(facecolor=c, label=str(i)) for i, c in enumerate(FOLD_COLORS, 1)],
                title="Outer fold", loc="lower center", bbox_to_anchor=(.755, .046),
                frameon=False, ncol=5, fontsize=7.5, title_fontsize=8,
                columnspacing=.7, handlelength=1.0, handletextpad=.3)
    detail = {
        "inputs": [input_record(p) for p in [GRID_PATH, BOUNDARY_PATH, points_path, folds_path]],
        "rows": 7229, "wildfire_locations": 11434, "spatial_groups": 94,
        "source_values_changed": False, "models_fitted": 0,
        "map_crs": "EPSG:5070", "matching_panel_extents": True,
        "fold_colors": dict(zip(range(1, 6), FOLD_COLORS)),
        "revision_of": "figures/publication/figure1_study_area_spatial_folds",
        "caption": "Coordinate-axis revision of the existing study-area figure. The same 11,434 recorded wildfire locations, 7,229 modeled cells, 94 spatial groups, and original outer-fold assignments are retained. Group boundaries are unions of original modeled cells. Coordinate axes are explicitly labeled in EPSG:5070 kilometres; distances and north direction are drawn by the common map formatter.",
    }
    export_figure(fig, "R1_study_area_coordinates", detail)


def susceptibility_coordinates():
    grid = load_grid()
    boundary = load_boundary()
    if set(grid.susceptibility_class) != set(CLASS_ORDER):
        raise ValueError("Saved susceptibility class labels changed.")
    fig, ax = plt.subplots(figsize=(7.4, 6.5))
    fig.subplots_adjust(left=.12, right=.975, bottom=.19, top=.965)
    for label, color in zip(CLASS_ORDER, CLASS_COLORS):
        grid.loc[grid.susceptibility_class == label].plot(ax=ax, color=color, linewidth=0)
    boundary.boundary.plot(ax=ax, color=INK, linewidth=.65)
    finish_map(ax, boundary)
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#a3a3a3", linewidth=.35, label=l)
                         for l, c in zip(CLASS_ORDER, CLASS_COLORS)],
                loc="lower center", bbox_to_anchor=(.54, .025), ncol=5,
                title="Relative susceptibility", frameon=False, title_fontsize=9,
                columnspacing=1.1, handlelength=1.2, handletextpad=.4)
    detail = {
        "inputs": [input_record(GRID_PATH), input_record(BOUNDARY_PATH)],
        "rows": 7229, "class_order": CLASS_ORDER,
        "class_counts": grid.susceptibility_class.value_counts().reindex(CLASS_ORDER).to_dict(),
        "source_values_changed": False, "models_fitted": 0,
        "map_crs": "EPSG:5070", "scores_or_classes_recomputed": False,
        "revision_of": "figures/publication/figure5_relative_susceptibility",
        "caption": "Coordinate-axis revision of the existing final-RF susceptibility map. The saved model scores, cell geometry, and five equal-frequency class assignments are unchanged. These classes represent relative statewide susceptibility rankings rather than calibrated wildfire probabilities.",
    }
    export_figure(fig, "R5_susceptibility_coordinates", detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", choices=["A2", "S5", "R1", "R5"],
                        default=["A2", "S5", "R1", "R5"])
    args = parser.parse_args()
    functions = {"A2": spatial_shap, "S5": population_road_dependence,
                 "R1": study_area_coordinates, "R5": susceptibility_coordinates}
    for name in args.figures:
        print(f"Preparing {name} from preserved values…", flush=True)
        functions[name]()


if __name__ == "__main__":
    main()
