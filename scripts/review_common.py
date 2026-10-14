"""Shared read-only inputs and exports for the separate manuscript review package.

No fitting or modification of historical artifacts is performed here.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import inspect
import json

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
from pyproj import Transformer

from project import ROOT, FEATURES, load_model_data

OUT = ROOT / 'figures/review_2026-10-08'
DATA = OUT / 'data'
INK = '#263442'
MUTED = '#586776'
RF_COLOR = '#29658d'
XGB_COLOR = '#b95825'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
    'axes.titlesize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'axes.edgecolor': MUTED, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})


def path_of(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    h = hashlib.sha256()
    with path_of(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def input_record(path):
    path = path_of(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha256(path)}


def read_csv(path):
    return pd.read_csv(path_of(path), float_precision='round_trip', dtype={'block_id': str})


def json_default(value):
    if isinstance(value, Path):
        return value.relative_to(ROOT).as_posix() if value.is_absolute() else value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError('Review JSON exports must stay inside the review directory.')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + '\n')


def save_table(frame, name):
    path = DATA / name
    if not path.resolve().is_relative_to(DATA.resolve()):
        raise ValueError('Review tables must stay inside the review data directory.')
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_nested():
    frame, _, y, groups = load_model_data()
    outer = read_csv('results/reproducibility/outer_fold_assignments.csv')
    xgb = read_csv('results/reproducibility/xgb_nested_oof_predictions.csv')
    if not np.array_equal(outer.grid_id, frame.grid_id):
        raise ValueError('Outer fold row identities differ from the saved checkpoint.')
    if not np.array_equal(outer.block_id, groups):
        raise ValueError('Outer block IDs differ.')
    xgb = xgb.set_index('grid_id').loc[frame.grid_id].reset_index()
    for name, expected in [('fire_presence', y), ('block_id', groups), ('outer_fold', outer.outer_fold)]:
        if not np.array_equal(xgb[name], expected):
            raise ValueError(f'Nested XGBoost {name} differs from RF/shared folds.')
    result = frame[['grid_id', 'block_id', 'fire_presence', 'fire_count']].copy()
    result['outer_fold'] = outer.outer_fold.to_numpy()
    result['rf_score'] = frame.rf_nested_oof_prob
    result['rf_pred'] = frame.rf_nested_oof_pred.astype(int)
    result['xgb_score'] = xgb.oof_probability.to_numpy()
    result['xgb_pred'] = xgb.oof_prediction.to_numpy(int)
    for family in ['rf', 'xgb']:
        score = result[f'{family}_score']
        pred = result[f'{family}_pred']
        if not np.isfinite(score).all() or not score.between(0, 1).all():
            raise ValueError(f'Invalid {family} scores.')
        if not np.array_equal(pred, (score > .5).astype(int)):
            raise ValueError(f'{family} saved classifications differ at threshold 0.50.')
    if len(result) != 7229 or result.grid_id.nunique() != 7229 or y.sum() != 2988:
        raise ValueError('Modeling population changed.')
    return result


def load_grid():
    grid = gpd.read_file(ROOT / 'data/Missouri_Wildfire_Susceptibility_5km.gpkg',
                         layer='wildfire_susceptibility')
    if len(grid) != 7229 or grid.grid_id.nunique() != 7229 or grid.crs.to_epsg() != 5070:
        raise ValueError('Saved map geometry/identities changed.')
    if grid.geometry.is_empty.any() or grid.geometry.isna().any():
        raise ValueError('Missing saved cell geometry.')
    return grid


def load_boundary():
    path = ROOT / 'data/derived/missouri_boundary.geojson'
    metadata = json.loads((ROOT / 'data/derived/figure1_input_provenance.json').read_text())
    expected = next(r['sha256'] for r in metadata['outputs'] if r['path'] == path.relative_to(ROOT).as_posix())
    if sha256(path) != expected:
        raise ValueError('Saved boundary helper changed.')
    return gpd.read_file(path).to_crs(5070)


def panel(ax, letter, title):
    ax.set_title(f'({letter})  {title}', loc='left', fontweight='semibold', pad=10)


def finish_map(ax, boundary):
    """Projected coordinates in km, projected scale, and local true-north arrow."""
    x0, y0, x1, y1 = boundary.total_bounds
    width, height = x1 - x0, y1 - y0
    x0 -= .075 * width; x1 += .075 * width
    y0 -= .10 * height; y1 += .075 * height
    ax.set_axis_on()
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal')
    formatter = FuncFormatter(lambda value, _: f'{value / 1000:,.0f}')
    ax.xaxis.set_major_formatter(formatter); ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax.set_xlabel('Easting (km)', fontsize=8, labelpad=3)
    ax.set_ylabel('Northing (km)', fontsize=8, labelpad=3)
    ax.tick_params(labelsize=7, length=2.5, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color('#bac2c9'); spine.set_linewidth(.5)
    bx, by = x0 + .06 * (x1 - x0), y0 + .055 * (y1 - y0)
    h = .008 * (y1 - y0)
    for i, color in enumerate([INK, 'white']):
        ax.add_patch(Rectangle((bx + i * 50000, by), 50000, h,
                              facecolor=color, edgecolor=INK, linewidth=.6, zorder=6))
    ax.text(bx, by + 2 * h, '0', fontsize=6.5, ha='center', zorder=6)
    ax.text(bx + 100000, by + 2 * h, '100 km', fontsize=6.5, ha='center', zorder=6)
    px, py = x0 + .90 * (x1 - x0), y0 + .87 * (y1 - y0)
    to_geo = Transformer.from_crs(5070, 4326, always_xy=True)
    to_map = Transformer.from_crs(4326, 5070, always_xy=True)
    lon, lat = to_geo.transform(px, py)
    nx, ny = to_map.transform(lon, lat + .25)
    direction = np.array([nx - px, ny - py]); direction /= np.linalg.norm(direction)
    end = np.array([px, py]) + direction * .085 * (y1 - y0)
    ax.annotate('', xy=end, xytext=(px, py),
                arrowprops={'arrowstyle': '-|>', 'lw': .8, 'color': INK}, zorder=7)
    ax.text(end[0], end[1] + .025 * (y1 - y0), 'N', ha='center', fontsize=8, fontweight='bold')


def save_figure(fig, stem, detail):
    if Path(stem).name != stem:
        raise ValueError('Use a figure basename, not a path.')
    OUT.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        for collection in ax.collections:
            collection.set_rasterized(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    outside = []
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -2 or box.y0 < -2 or box.x1 > width + 2 or box.y1 > height + 2):
            outside.append(artist.get_text())
    if outside:
        raise ValueError(f'{stem}: text outside canvas: {outside}')
    pdf = OUT / f'{stem}.pdf'; png = OUT / f'{stem}.png'
    fig.savefig(pdf, metadata={'Title': stem, 'Creator': 'Missouri wildfire review figure scripts'})
    fig.savefig(png, dpi=600)
    size = fig.get_size_inches()
    caller = Path(inspect.getsourcefile(inspect.currentframe().f_back)).resolve()
    record = {**detail, 'stem': stem, 'generated_utc': datetime.now(timezone.utc).isoformat(),
              'generator': input_record(caller), 'shared_generator': input_record(Path(__file__).resolve()),
              'outputs': [input_record(pdf), input_record(png)],
              'size_inches': size.tolist(), 'png_size_pixels': np.rint(size * 600).astype(int).tolist(),
              'png_dpi': 600, 'models_fitted': False, 'text_outside_canvas': outside,
              'original_files_overwritten': False}
    write_json(OUT / f'{stem}.json', record)
    plt.close(fig)
    print(f'Saved {stem}: PDF and 600 dpi PNG', flush=True)
    return record
