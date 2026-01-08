#!/bin/bash
set -e

# ============================================================
# Run ENVIRONMENT preparation for TRAIN cities and TEST cities
# ============================================================

CITYNAME="澳门"
PROVINCE="澳门特别行政区"

SHP_PATH="./datasets/data_preparation/geographic_data/全国地级市边界/2024年初地级.shp"
POP_PATH="./datasets/data_preparation/geographic_data/chn_ppp_2020_constrained.tif"

OSM_DIR="./datasets/data_preparation/geographic_data/${CITYNAME}/OSM_${PROVINCE}"
POI_DIR="./datasets/data_preparation/geographic_data/${CITYNAME}/POI_${CITYNAME}"

python ./datasets/data_preparation/Env_extra.py \
  --cityname "${CITYNAME}" \
  --province "${PROVINCE}" \
  --shp_path "${SHP_PATH}" \
  --pop_path "${POP_PATH}" \
  --osm_dir "${OSM_DIR}" \
  --poi_dir "${POI_DIR}"
