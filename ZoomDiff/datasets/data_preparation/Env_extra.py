import numpy as np
import pandas as pd
import geopandas as gpd
from funcs import (
    build_100m_grid, aggregate_edges_5x5, get_loc_2000m,
    count_poi_by_grid, reshape_4x4_blocks, aggregate_5x5_blocks,
    read_osm_to_proj, make_grid_gdf, ensure_proj, count_by_centroid,
    accumulate_length_in_cells, accumulate_area_in_cells
)
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import Polygon
from pyproj import CRS
import glob
import os
from tqdm.auto import tqdm, trange
from contextlib import contextmanager
import argparse

@contextmanager
def step(desc: str):
    bar = tqdm(total=1, desc=desc)
    try:
        yield
    finally:
        bar.update(1)
        bar.close()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Urban grid data preparation pipeline"
    )
    parser.add_argument(
        "--cityname",
        type=str,
        default="成都",
        help="City name in Chinese"
    )
    parser.add_argument(
        "--province",
        type=str,
        default="四川省",
        help="Province name in Chinese"
    )
    parser.add_argument(
        "--shp_path",
        type=str,
        default="../../公共地理数据/地级市边界/2024年初地级.shp",
        help="Path to prefecture-level city boundary shapefile"
    )
    parser.add_argument(
        "--pop_path",
        type=str,
        default="../../公共地理数据/chn_ppp_2020_constrained.tif",
        help="Path to population raster (.tif)"
    )
    parser.add_argument(
        "--osm_dir",
        type=str,
        default=f"./地理数据/OSM_四川省/",
        help="Directory containing OSM shapefiles"
    )
    parser.add_argument(
        "--poi_dir",
        type=str,
        default=f"./地理数据/POI_成都/",
        help="Directory containing POI CSV files"
    )

    return parser.parse_args()

args = parse_args()

# City Selection =============================================================
cityname = args.cityname
os.makedirs(f'./results/{cityname}/', exist_ok=True)

SHP_pth = args.shp_path
with step("Reading prefecture-level city boundary shapefile"):
    gdf = gpd.read_file(SHP_pth)

city_boundary = gdf[gdf["地名"].str.contains(cityname)].geometry.values[0]
long_min, lat_min, long_max, lat_max = city_boundary.bounds
# long_min, lat_min, long_max, lat_max = 115.80, 28.68, 115.90, 28.73

# Grid Partition =============================================================
with step("Constructing 100m grid boundaries"):
    lon_edges, lat_edges, x_edges, y_edges, crs_proj, to_proj, to_geo, epsg = \
        build_100m_grid(long_min, lat_min, long_max, lat_max)

Nx = len(lon_edges) - 1  # longitude direction (columns)
Ny = len(lat_edges) - 1  # latitude direction (rows)
shape = (Ny, Nx)
print("Region size (100m grid): rows (latitude)", Ny, ", columns (longitude)", Nx)


# Population Statistics ======================================================
Pop_pth = args.pop_path
with rasterio.open(Pop_pth) as src:
    nodata = src.nodata
    pop_100m = np.zeros((Ny, Nx), dtype=np.float32)  # rows=latitude (y), cols=longitude (x)

    for i in trange(Ny, desc=f"Reading population raster: rows (Ny={Ny})"):
        lat_bottom = lat_edges[i]
        lat_top = lat_edges[i + 1]
        for j in range(Nx):
            lon_left = lon_edges[j]
            lon_right = lon_edges[j + 1]
            try:
                win = from_bounds(
                    lon_left, lat_bottom, lon_right, lat_top,
                    transform=src.transform
                )
                arr = src.read(1, window=win, boundless=True, fill_value=nodata)

                # Handle nodata and aggregate (mean)
                if nodata is not None:
                    valid = ~np.isnan(arr) if np.isnan(nodata) else (arr != nodata)
                    value = float(arr[valid].mean()) if valid.any() else 0.0
                else:
                    value = float(arr.mean())

            except Exception:
                value = 0.0

            pop_100m[i, j] = value


grid = make_grid_gdf(x_edges, y_edges, CRS.from_user_input(crs_proj), Nx, Ny)
grid_sindex = grid.sindex


# Preparing Environmental Landforms =========================================
OSM_dir = args.osm_dir
buildings_a = read_osm_to_proj(OSM_dir + "/gis_osm_buildings_a_free_1.shp", crs_proj)
natural_p   = read_osm_to_proj(OSM_dir + "/gis_osm_natural_free_1.shp", crs_proj)
natural_a   = read_osm_to_proj(OSM_dir + "/gis_osm_natural_a_free_1.shp", crs_proj)
pois_p      = read_osm_to_proj(OSM_dir + "/gis_osm_pois_free_1.shp", crs_proj)
roads_l     = read_osm_to_proj(OSM_dir + "/gis_osm_roads_free_1.shp", crs_proj)
water_a     = read_osm_to_proj(OSM_dir + "/gis_osm_water_a_free_1.shp", crs_proj)
waterways_l = read_osm_to_proj(OSM_dir + "/gis_osm_waterways_free_1.shp", crs_proj)

# Reproject to grid CRS (meters)
crs_proj = grid.crs
buildings_a = ensure_proj(buildings_a, crs_proj)
natural_p   = ensure_proj(natural_p, crs_proj)
natural_a   = ensure_proj(natural_a, crs_proj)
pois_p      = ensure_proj(pois_p, crs_proj)
roads_l     = ensure_proj(roads_l, crs_proj)
water_a     = ensure_proj(water_a, crs_proj)
waterways_l = ensure_proj(waterways_l, crs_proj)


# === Result arrays ===
buildings_cnt = np.zeros(shape, dtype=np.int32)
natural_cnt   = np.zeros(shape, dtype=np.int32)
pois_cnt      = np.zeros(shape, dtype=np.int32)
roads_len_m   = np.zeros(shape, dtype=np.float64)
water_area_m2 = np.zeros(shape, dtype=np.float64)


# === Statistics ===
# Count by centroid
count_by_centroid(buildings_a, buildings_cnt, Nx, Ny, x_edges, y_edges)
count_by_centroid(natural_p,   natural_cnt,   Nx, Ny, x_edges, y_edges)
count_by_centroid(natural_a,   natural_cnt,   Nx, Ny, x_edges, y_edges)
count_by_centroid(pois_p,      pois_cnt,      Nx, Ny, x_edges, y_edges)

# Line length / polygon area accumulation
accumulate_length_in_cells(roads_l,     roads_len_m,   grid, grid_sindex)
accumulate_length_in_cells(waterways_l, roads_len_m,   grid, grid_sindex)
accumulate_area_in_cells(water_a,       water_area_m2, grid, grid_sindex)

building_100m = buildings_cnt
roadlen_100m = roads_len_m
water_area_100m = water_area_m2

# Point of Interests =========================================================
POI_dir = args.poi_dir

# Read and filter POI data
csv_files = glob.glob(os.path.join(POI_dir, '*.csv'))
print("Number of POI files: ", len(csv_files))

all_columns = set()
for csv_file in csv_files:
    df = pd.read_csv(csv_file, nrows=0)
    all_columns.update(df.columns.tolist())
all_columns = sorted(all_columns)

if 'lon_wgs84' in all_columns:
    df_cityname, df_bigType, df_lon_wgs84, df_lat_wgs84 = 'cityname', 'bigType', 'lon_wgs84', 'lat_wgs84'
elif 'wgs84_经度' in all_columns:
    df_cityname, df_bigType, df_lon_wgs84, df_lat_wgs84 = 'cityname', '行业大类', 'wgs84_经度', 'wgs84_纬度'

use_cols = [df_cityname, df_bigType, df_lon_wgs84, df_lat_wgs84]
dfs = [pd.read_csv(f, usecols=use_cols, dtype=str) for f in csv_files]
poi_df = pd.concat(dfs, ignore_index=True)

# Get unique major categories
unique_majors = sorted(poi_df[df_bigType].unique())

# Build label index mapping
label_map = {major: idx for idx, major in enumerate(unique_majors)}

# Keep only POIs within the target city
poi_df = (
    poi_df[
        poi_df[df_cityname].isin([f'{cityname}市'])
    ]
    .dropna(subset=[df_lon_wgs84, df_lat_wgs84, df_bigType])
    .copy()
)

poi_df['lon'] = poi_df[df_lon_wgs84].astype(float)
poi_df['lat'] = poi_df[df_lat_wgs84].astype(float)

grid_counts = count_poi_by_grid(
    poi_df=poi_df,
    lon_edges=lon_edges,
    lat_edges=lat_edges,
    label_map=label_map,
    lon_col="lon",
    lat_col="lat",
    cat_col=df_bigType,
    dtype=np.int32
)
poi_100m = grid_counts


# Upward aggregation =========================================================
with step("Aggregating to coarser resolutions"):
    # 1. Aggregate to 500m
    pop_500m = aggregate_5x5_blocks(pop_100m)
    building_500m = aggregate_5x5_blocks(building_100m)
    roadlen_500m = aggregate_5x5_blocks(roadlen_100m)
    water_area_500m = aggregate_5x5_blocks(water_area_100m)
    poi_500m = aggregate_5x5_blocks(poi_100m)
    lat_edges_500m, lon_edges_500m = aggregate_edges_5x5(lat_edges, lon_edges)

    # 2. Aggregate to 2000m
    pop_2000m = reshape_4x4_blocks(pop_500m)
    building_2000m = reshape_4x4_blocks(building_500m)
    roadlen_2000m = reshape_4x4_blocks(roadlen_500m)
    water_area_2000m = reshape_4x4_blocks(water_area_500m)
    poi_2000m = reshape_4x4_blocks(poi_500m)
    loc_2000m = get_loc_2000m(lat_edges_500m, lon_edges_500m)

    np.savez(
        f'./datasets/cond/{cityname}_cond.npz',
        pop_2000m=pop_2000m,
        building_2000m=building_2000m,
        roadlen_2000m=roadlen_2000m,
        water_area_2000m=water_area_2000m,
        poi_2000m=poi_2000m,
        loc_2000m=loc_2000m
    )