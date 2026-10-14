"""Shared paths and the original, ordered 33-column modeling matrix.

Loading the saved checkpoint must not turn its OOF prediction columns into
predictors. This whitelist matches the executed notebook's original X.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "results" / "reproducibility"
PUBLICATION = ROOT / "figures" / "publication"
FEATURES = [
    "elevation", "slope", "northness", "eastness", "canopy", "ndvi",
    "temperature", "precipitation", "pdsi", "wind", "road_distance", "population",
    "lc_11", "lc_22", "lc_23", "lc_24", "lc_41", "lc_42", "lc_43",
    "lc_81", "lc_82", "lc_90", "lc_95",
    "fuel_102", "fuel_103", "fuel_182", "fuel_185", "fuel_186", "fuel_188",
    "fuel_189", "fuel_91", "fuel_93", "fuel_98",
]


def load_model_data():
    """Read the preserved checkpoint without changing row or feature order."""
    frame = pd.read_csv(
        ROOT / "data" / "missouri_wildfire_model_encoded.csv",
        float_precision="round_trip",
        dtype={"block_id": str},
    )
    if len(frame) != 7229 or not frame["grid_id"].is_unique:
        raise ValueError("Expected the original 7,229 unique modeling cells.")
    missing = set(FEATURES).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing original predictors: {sorted(missing)}")
    X = frame.loc[:, FEATURES].copy()
    if X.isna().any().any():
        raise ValueError("The saved modeling checkpoint contains missing predictors.")
    y = frame["fire_presence"].astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError("Expected recorded-wildfire labels 0 and 1.")
    groups = frame["block_id"]
    if groups.nunique() != 94:
        raise ValueError("Expected the original 94 spatial blocks.")
    return frame, X, y, groups
