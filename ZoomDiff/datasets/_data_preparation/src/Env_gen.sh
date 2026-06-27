set -e

# ============================================================
# Run ENVIRONMENT preparation for TRAIN cities and TEST cities
# ============================================================

CITYNAME="Nanchang"
PROVINCE="Jiangxi"

SHP_PATH="./ZoomDiff/datasets/_shared_geographic_data/China_city_boundaries/china_city_boundaries_2024.shp"
POP_PATH="./ZoomDiff/datasets/_shared_geographic_data/chn_ppp_2020_constrained.tif"

python ./ZoomDiff/datasets/_data_preparation/Env_extra.py \
  --cityname "${CITYNAME}" \
  --province "${PROVINCE}" \
  --shp_path "${SHP_PATH}" \
  --pop_path "${POP_PATH}"
