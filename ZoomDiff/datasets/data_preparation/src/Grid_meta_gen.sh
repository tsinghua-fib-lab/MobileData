#!/bin/bash
set -e

# =====================================
# Run DATA preparation for TRAIN cities
# =====================================

CITYNAME="南京"
PROVINCE="江苏省"
DATATYPE="traffic"

DATA_PATH="./datasets/data_preparation/filtered_data/${CITYNAME}/filtered_${CITYNAME}_${DATATYPE}.npz"
SHP_PATH="./datasets/data_preparation/geographic_data/全国地级市边界/2024年初地级.shp"
POP_PATH="./datasets/data_preparation/geographic_data/chn_ppp_2020_constrained.tif"

OSM_DIR="./datasets/data_preparation/geographic_data/${CITYNAME}/OSM_${PROVINCE}"
POI_DIR="./datasets/data_preparation/geographic_data/${CITYNAME}/POI_${CITYNAME}"

python ./datasets/data_preparation/Grid_meta.py \
  --cityname "${CITYNAME}" \
  --province "${PROVINCE}" \
  --data_path "${DATA_PATH}" \
  --shp_path "${SHP_PATH}" \
  --pop_path "${POP_PATH}"
