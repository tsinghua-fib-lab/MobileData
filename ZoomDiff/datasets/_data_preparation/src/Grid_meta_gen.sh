set -e

# =====================================
# Run DATA preparation for TRAIN cities
# =====================================

CITYNAME="Nanchang"
PROVINCE="Jiangxi"
DATATYPE="traffic"

SHP_PATH="./ZoomDiff/datasets/_shared_geographic_data/China_city_boundaries/china_city_boundaries_2024.shp"
POP_PATH="./ZoomDiff/datasets/_shared_geographic_data/chn_ppp_2020_constrained.tif"

python ./ZoomDiff/datasets/_data_preparation/Grid_meta.py \
  --cityname "${CITYNAME}" \
  --province "${PROVINCE}" \
  --datatype "${DATATYPE}" \
  --shp_path "${SHP_PATH}" \
  --pop_path "${POP_PATH}"
