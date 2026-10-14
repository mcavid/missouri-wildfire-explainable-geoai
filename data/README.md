# Processed research inputs

- missouri_wildfire_model_encoded.csv: 7,229 unique cells; 33 predictor columns
  selected by scripts/project.py; response, count, IDs and saved predictions.
- Missouri_Wildfire_Susceptibility_5km.gpkg: cell geometry in EPSG:5070, mapped RF
  score/classes and saved evaluation fields. Full-data and nested scores differ.
- derived/recorded_wildfire_locations_2021_2025.csv: 11,434 study-period coordinates
  in EPSG:5070; identifiers, names and other incident attributes are not supplied.
- derived/missouri_boundary.geojson: plotting boundary in EPSG:4326.
- LF2024_FBFM40_CONUS/metadata.xml: provider metadata supporting LANDFIRE v2.5.0.

The modeled grid contains 11,369 assigned incident records; 65 selected boundary
records are unassigned under the original extraction. See docs/data_sources.md
for source conventions and DATA_LICENSE.md for provider attribution.
