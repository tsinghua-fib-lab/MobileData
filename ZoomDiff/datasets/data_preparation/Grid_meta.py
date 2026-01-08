import numpy as np
import pandas as pd
import geopandas as gpd
from funcs import (
    build_100m_grid, aggregate_edges_5x5, get_loc_2000m,
    reshape_4x4_blocks, aggregate_5x5_blocks, make_grid_gdf,
    allocate_data_to_users, aggregate_data_to_grids, 
    match_users_to_bs, group_users_by_bs, generate_user_points
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
        default="南昌",
        help="City name in Chinese"
    )
    parser.add_argument(
        "--province",
        type=str,
        default="江西省",
        help="Province name in Chinese"
    )
    parser.add_argument(
        "--datatype",
        type=str,
        default="traffic",
        help="traffic or user"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./filtered_data/南昌/filtered_南昌_traffic.npz",
        help="Path to prefecture-level city boundary shapefile"
    )
    parser.add_argument(
        "--shp_path",
        type=str,
        default="./geographic_data/全国地级市边界/2024年初地级.shp",
        help="Path to prefecture-level city boundary shapefile"
    )
    parser.add_argument(
        "--pop_path",
        type=str,
        default="./geographic_data/chn_ppp_2020_constrained.tif",
        help="Path to population raster (.tif)"
    )

    return parser.parse_args()

args = parse_args()

# City Selection =============================================================
cityname = args.cityname
os.makedirs(f'./results/{cityname}/', exist_ok=True)

SHP_pth = args.shp_path
with step("Reading prefecture-level city boundary shapefile"):
    gdf = gpd.read_file(SHP_pth)

city_boundary = gdf[gdf["地名"] == f"{cityname}市"].geometry.values[0]
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

# Traffic& User Data Processing ==============================================
# Data Loading ---------------------------------------------------
datatype = args.datatype
data_pth = args.data_path
with step("Loading data ..."):
    bs_data_file = np.load(data_pth, allow_pickle=True)

filtered_bs_data = bs_data_file[f'{datatype}']
filtered_bs_long = bs_data_file['long']
filtered_bs_lat = bs_data_file['lat']
filtered_bs_cgi = np.array([f'NJBS-{i}' for i in range(len(filtered_bs_lat))])

# Data Filtering ---------------------------------------------------
mask_bbox = (filtered_bs_long >= long_min) & (filtered_bs_long <= long_max) & \
            (filtered_bs_lat  >= lat_min)  & (filtered_bs_lat  <= lat_max)
mask = mask_bbox

# 6) 筛选落在{cityname}市且位于经纬度框内的基站
selected_bs_data = filtered_bs_data[mask]
selected_bs_long    = filtered_bs_long[mask]
selected_bs_lat     = filtered_bs_lat[mask]
selected_bs_cgi     = filtered_bs_cgi[mask]
np.nan_to_num(selected_bs_data, nan=0.0, copy=False)

selected_cgi_data_dict = {}
selected_cgi_lat_dict = {}
selected_cgi_lon_dict = {}
for i in range(len(selected_bs_cgi)):
    selected_cgi_data_dict[selected_bs_cgi[i]] = selected_bs_data[i]
    selected_cgi_lat_dict[selected_bs_cgi[i]] = selected_bs_lat[i]
    selected_cgi_lon_dict[selected_bs_cgi[i]] = selected_bs_long[i]

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

# Anonymized Traffic & User Construction =====================================
# Generate user points within each grid
# (from funcs, no fine-grained instrumentation available, use stage-level progress)
with step("Generating user points based on population (generate_user_points)"):
    user_points = generate_user_points(pop_100m, lon_edges, lat_edges)


# UE–BS Association 
# Determine which grid each base station falls into (vector → index)
with step("Computing base station grid indices"):
    i = np.digitize(selected_bs_long, lon_edges, right=False) - 1
    j = np.digitize(selected_bs_lat,  lat_edges,  right=False) - 1
    valid = (
        (i >= 0) & (i < len(lon_edges) - 1) &
        (j >= 0) & (j < len(lat_edges) - 1)
    )
    selected_bs_long_idx = i[valid]
    selected_bs_lat_idx  = j[valid]


# Match UEs to base stations (from funcs)
with step("Matching UEs to base stations (match_users_to_bs)"):
    user_ids, user_lons, user_lats, user_bs_cgis = match_users_to_bs(
        user_points,
        bs_lons=selected_bs_long,
        bs_lats=selected_bs_lat,
        bs_cgis=selected_bs_cgi
    )
    selected_cgi_data_dict['BS-None'] = np.zeros(168, dtype=np.float64)


with step("Grouping UEs by base station (group_users_by_bs)"):
    bs_user_dict = group_users_by_bs(user_ids, user_bs_cgis)


# Construct statistics (may take some time if many base stations are involved)
data = {
    "CGI": [],
    "User_Count": []
}
for cgi, user_list in tqdm(
    bs_user_dict.items(),
    desc="Counting the number of UEs associated with each base station"
):
    if user_list:
        data["CGI"].append(cgi)
        data["User_Count"].append(len(user_list))

# Build DataFrame table
df_bs_user_count = pd.DataFrame(data)

# Sort by user count in descending order (optional)
df_bs_user_count.sort_values(by="User_Count", ascending=False, inplace=True)

# Base station traffic allocation (from funcs)
with step(f"Allocating {datatype} data to UEs (allocate_data_to_users)"):
    user_data_dict = allocate_data_to_users(
        selected_cgi_data_dict,
        bs_user_dict
    )


# User traffic re-aggregation to 100m grids (from funcs)
with step("Aggregating UE traffic to 100m grids (aggregate_data_to_grids)"):
    grid_data_100m, grid_data_dict = aggregate_data_to_grids(
        pop_100m,
        user_points,
        user_data_dict
    )

# Upward aggregation =========================================================
with step("Aggregating to coarser resolutions"):
    # 1. Aggregate to 500m
    data_500m = aggregate_5x5_blocks(grid_data_100m)
    lat_edges_500m, lon_edges_500m = aggregate_edges_5x5(lat_edges, lon_edges)

    # 2. Aggregate to 2000m
    data_2000m = reshape_4x4_blocks(data_500m)
    loc_2000m = get_loc_2000m(lat_edges_500m, lon_edges_500m)

    np.savez(
        f'./datasets/data/{cityname}_{datatype}_data.npz',
        data_2000m=data_2000m,
        loc_2000m=loc_2000m
    )