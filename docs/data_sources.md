# Source-product conventions and retained inputs

The analysis starts with retained processed rasters/checkpoint data. Published
product-family specifications below are distinct from the export-level QA,
resampling and transformations. No upstream processing is inferred from filenames.

| Source family | Published convention | Retained representation |
|---|---|---|
| USGS 3DEP | 1/3 arc-second, approximately 10 m, elevation in meters | Terrain bands in 250-m stack |
| USFS/GTAC canopy | 30-m native pixels; 2021 NLCD canopy is percent cover | canopy band in 250-m stack |
| MODIS MOD13Q1 V6.1 | 250 m; 16-day indices; NDVI scale 0.0001; QA bands | ndvi band; recorded April–October 2021–2025 summary |
| NLCD | 30-m categorical land-cover classes | 250-m export; dominant class and indicators |
| PRISM | approximately 4 km; temperature Celsius, precipitation mm | temperature/precipitation summary bands |
| gridMET and gridMET Drought | approximately 4 km; 10-m-height wind m/s; PDSI index | wind/pdsi summary bands |
| WorldPop | count and density products have different units | sum of retained 100-m population values; no density conversion |
| Census TIGER/Line | vector road geometry | road-distance band in 250-m stack |
| LANDFIRE LF2024 FBFM40 v2.5.0 | 30-m categorical fuel codes | dominant class and fuel indicators |
| NIFC InFORM | downloaded wildfire occurrence records | selected 2021–2025 record coordinates and cell response |

The 11-band continuous and NLCD exports use 250-m spacing; population uses
100 m. Models/plots keep the stored predictor values. Product scale factors and
QA flags describe source conventions and are not re-applied to processed summaries.
Population masks are zero-filled and summed, and categorical rasters use dominant
class aggregation. The fixed feature list excludes response and prediction fields.

Official sources (checked 13 October 2026):

- Canopy: https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/metadata/nlcd_tcc_conus_2021_v2021-4.tif.html
- 3DEP: https://developers.google.com/earth-engine/datasets/catalog/USGS_3DEP_10m
- MODIS: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1
- NLCD: https://developers.google.com/earth-engine/datasets/catalog/USGS_NLCD_RELEASES_2021_REL_NLCD
- PRISM: https://prism.oregonstate.edu/faq/
- gridMET: https://www.climatologylab.org/gridmet.html
- gridMET Drought: https://developers.google.com/earth-engine/datasets/catalog/GRIDMET_DROUGHT
- WorldPop: https://developers.google.com/earth-engine/datasets/catalog/WorldPop_GP_100m_pop
- TIGER/Line: https://www2.census.gov/geo/pdfs/maps-data/data/tiger/tgrshp2016/TGRSHP2016_TechDoc.pdf
- LANDFIRE: https://landfire.gov/fuel/fbfm40 and the included provider metadata
- NIFC: https://data-nifc.opendata.arcgis.com/pages/d6ef1367fadc4405b5f09c98e52ed972

Raw acquisitions are separate from this processed-input repository: the original
InFORM incident GeoPackage, Missouri boundary shapefile, continuous predictor
stack, NLCD 2021 export, WorldPop 2020 export and clipped LANDFIRE raster. The
notebook documents the downstream extraction calls; this release does not supply
an invented GEE asset manifest, QA mask, or upstream export program.
