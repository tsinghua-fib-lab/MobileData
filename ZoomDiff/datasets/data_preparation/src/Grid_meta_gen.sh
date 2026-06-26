set -e

# =====================================
# Run DATA preparation for TRAIN cities
# =====================================

CITYNAME="南京"
PROVINCE="江苏省"
DATATYPE="traffic"

DATA_PATH="./ZoomDiff/datasets/data_preparation/filtered_data/${CITYNAME}/filtered_${CITYNAME}_${DATATYPE}.npz"
SHP_PATH="./ZoomDiff/datasets/data_preparation/geographic_data/China_city_boundaries/china_city_boundaries_2024.shp"
POP_PATH="./ZoomDiff/datasets/data_preparation/geographic_data/chn_ppp_2020_constrained.tif"

python ./ZoomDiff/datasets/data_preparation/Grid_meta.py \
  --cityname "${CITYNAME}" \
  --province "${PROVINCE}" \
  --data_path "${DATA_PATH}" \
  --shp_path "${SHP_PATH}" \
  --pop_path "${POP_PATH}"
