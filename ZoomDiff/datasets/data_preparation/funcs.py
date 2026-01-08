import numpy as np
from pyproj import CRS, Transformer
from collections import defaultdict
from typing import Optional
from scipy.spatial import cKDTree
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from shapely.geometry import Polygon

def build_100m_grid(lb_lon, lb_lat, rt_lon, rt_lat, cell_size=100.0):
    lon_c, lat_c = (lb_lon + rt_lon) / 2, (lb_lat + rt_lat) / 2
    zone = int(np.floor((lon_c + 180) / 6) + 1)
    epsg = (32600 if lat_c >= 0 else 32700) + zone

    crs_geo = CRS.from_epsg(4326)
    crs_proj = CRS.from_epsg(epsg)
    to_proj = Transformer.from_crs(crs_geo, crs_proj, always_xy=True)
    to_geo  = Transformer.from_crs(crs_proj, crs_geo, always_xy=True)

    x_min, y_min = to_proj.transform(lb_lon, lb_lat)
    x_max, y_max = to_proj.transform(rt_lon, rt_lat)

    nx = int(np.ceil((x_max - x_min) / cell_size))
    ny = int(np.ceil((y_max - y_min) / cell_size))
    x_edges = x_min + np.arange(nx + 1) * cell_size
    y_edges = y_min + np.arange(ny + 1) * cell_size

    y_mid = (y_min + y_max) / 2
    x_mid = (x_min + x_max) / 2
    lon_edges, _ = to_geo.transform(x_edges, np.full_like(x_edges, y_mid))
    _, lat_edges = to_geo.transform(np.full_like(y_edges, x_mid), y_edges)

    # 现在多返回 crs_proj / 变换器（可选）
    return np.array(lon_edges), np.array(lat_edges), np.array(x_edges), np.array(y_edges), crs_proj, to_proj, to_geo, epsg

def generate_user_points(
    pop_100m: np.ndarray,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    x_edges: Optional[np.ndarray] = None,
    y_edges: Optional[np.ndarray] = None,
    epsg: Optional[int] = None,  # 若提供 x/y 边界但没给 epsg，也能从边界推
) -> dict:
    """
    根据每个栅格中的用户数，在对应栅格内均匀随机撒点。
    返回 dict: key 为 (i, j)，value 为 [[lat, lon], ...]
    
    参数：
    - pop_100m: (Ny, Nx) 每格用户数（可为浮点，内部 round 后取整）
    - lon_edges: 长度 Nx+1，经度方向边界（度）
    - lat_edges: 长度 Ny+1，纬度方向边界（度）
    - x_edges, y_edges: 若提供，则在米坐标内采样再反投影到经纬度（推荐）
    - epsg: 与 x/y 对应的投影 EPSG；若省略且给了 x/y，会尝试不依赖 epsg 的“恒定坐标系”反投影（需要时传入）
    """
    Ny, Nx = pop_100m.shape
    user_points_dict = defaultdict(list)

    use_proj = x_edges is not None and y_edges is not None
    if use_proj:
        # 若给了投影边界，构造投影↔地理的变换（需要 epsg）
        if epsg is None:
            raise ValueError("使用投影边界采样需要提供 epsg（如 UTM 区的 EPSG）。")
        crs_geo = CRS.from_epsg(4326)
        crs_proj = CRS.from_epsg(epsg)
        to_geo = Transformer.from_crs(crs_proj, crs_geo, always_xy=True)

    rng = np.random.default_rng()

    for i in range(Ny):      # 行：纬度（y）
        lat_lo = lat_edges[i]
        lat_hi = lat_edges[i+1]
        for j in range(Nx):  # 列：经度（x）
            lon_lo = lon_edges[j]
            lon_hi = lon_edges[j+1]

            user_count = int(round(pop_100m[i, j]))
            if user_count <= 0:
                continue

            if use_proj:
                # 在投影坐标（米）内均匀采样
                x_lo, x_hi = x_edges[j],   x_edges[j+1]
                y_lo, y_hi = y_edges[i],   y_edges[i+1]

                xs = rng.uniform(x_lo, x_hi, size=user_count)
                ys = rng.uniform(y_lo, y_hi, size=user_count)
                lons, lats = to_geo.transform(xs, ys)

            else:
                # 在经纬度边界内均匀采样（对 100m 级小格足够好）
                lats = rng.uniform(lat_lo, lat_hi, size=user_count)
                lons = rng.uniform(lon_lo, lon_hi, size=user_count)

            coords = [[float(lat), float(lon)] for lat, lon in zip(lats, lons)]
            user_points_dict[(i, j)] = coords

    return user_points_dict

def find_grid_index(lon, lat, lon_edges, lat_edges, *, right=False, clip=False):
    """
    给定点 (lon, lat) 与栅格边界，返回其所在网格索引 (i, j)
    - i: 经度方向（列索引，基于 lon_edges）
    - j: 纬度方向（行索引，基于 lat_edges）
    参数：
      right: False 表示 [edge_k, edge_{k+1})；True 表示 (edge_{k-1}, edge_k]
      clip : True 则越界裁剪到最近格；False 则越界返回 None
    说明：要求 lon_edges、lat_edges 单调递增；网格数 Nx=len(lon_edges)-1, Ny=len(lat_edges)-1
    """
    Nx = len(lon_edges) - 1
    Ny = len(lat_edges) - 1

    i = np.digitize(lon, lon_edges, right=right) - 1
    j = np.digitize(lat, lat_edges, right=right) - 1

    if 0 <= i < Nx and 0 <= j < Ny:
        return i, j
    if clip:
        return np.clip(i, 0, Nx-1), np.clip(j, 0, Ny-1)
    return None

def _choose_utm(lons, lats):
    """根据区域中心选择一个本地 UTM EPSG。"""
    lon_c = float(np.mean(lons))
    lat_c = float(np.mean(lats))
    zone = int(np.floor((lon_c + 180) / 6) + 1)
    epsg = (32600 if lat_c >= 0 else 32700) + zone
    return epsg

def match_users_to_bs(user_points_dict, bs_lons, bs_lats, bs_cgis):
    """
    根据就近原则（欧氏距离，米）为每个用户匹配最近的 1 个基站。
    返回四个平行列表：user_ids, user_lons, user_lats, user_bs_cgis
    """
    bs_lons = np.asarray(bs_lons, dtype=float)
    bs_lats = np.asarray(bs_lats, dtype=float)
    bs_cgis = np.asarray(bs_cgis)

    # 若没有用户或没有基站，直接返回空
    if (not user_points_dict) or (bs_lons.size == 0):
        return [], [], [], []

    # 将用户按“栅格索引升序 + 栅格内生成顺序”展平成列表，顺便编号
    user_lats, user_lons, user_ids = [], [], []
    uid = 0
    for key in sorted(user_points_dict.keys()):  # key=(i,j)
        pts = user_points_dict[key]
        if not pts:
            continue
        for lat, lon in pts:  # 注意：点是 [lat, lon]
            user_lats.append(float(lat))
            user_lons.append(float(lon))
            user_ids.append(uid)
            uid += 1

    if len(user_ids) == 0:
        return [], [], [], []

    user_lats = np.asarray(user_lats, dtype=float)
    user_lons = np.asarray(user_lons, dtype=float)
    user_ids  = np.asarray(user_ids,  dtype=int)

    # 选择本地 UTM（用用户+基站的总体中心更稳）
    all_lons = np.concatenate([user_lons, bs_lons])
    all_lats = np.concatenate([user_lats, bs_lats])
    utm_epsg = _choose_utm(all_lons, all_lats)

    crs_geo = CRS.from_epsg(4326)
    crs_utm = CRS.from_epsg(utm_epsg)
    to_utm  = Transformer.from_crs(crs_geo, crs_utm, always_xy=True)

    # 经纬度 → 米制平面坐标
    bs_x, bs_y       = to_utm.transform(bs_lons,  bs_lats)
    user_x, user_y   = to_utm.transform(user_lons, user_lats)

    # 最近邻查询
    tree = cKDTree(np.column_stack([bs_x, bs_y]))
    _, nn_idx = tree.query(np.column_stack([user_x, user_y]), k=1)

    # 最近基站的 CGI
    user_bs_cgis = bs_cgis[nn_idx].tolist()

    # 输出四个平行列表
    return user_ids.tolist(), user_lons.tolist(), user_lats.tolist(), user_bs_cgis

def group_users_by_bs(user_ids, user_bs_cgis, *, return_counts=False):
    """
    将用户按基站 CGI 分组。
    参数：
      - user_ids:      可迭代，一一对应的用户编号
      - user_bs_cgis:  可迭代，一一对应的基站 CGI（与 user_ids 等长）
      - return_counts: 若为 True，同时返回 (bs_to_users, bs_keys, counts)

    返回：
      - 当 return_counts=False：dict，键=CGI，值=list[uid]
      - 当 return_counts=True： (bs_to_users, bs_keys, counts)
          * bs_keys:  基站键的顺序列表（插入顺序）
          * counts:   与 bs_keys 对齐的用户数数组（np.ndarray, int）
    """
    # 基本校验
    if len(user_ids) != len(user_bs_cgis):
        raise ValueError("user_ids 与 user_bs_cgis 长度不一致。")

    bs_to_users = defaultdict(list)
    for uid, cgi in zip(user_ids, user_bs_cgis):
        bs_to_users[cgi].append(int(uid))

    if not return_counts:
        return dict(bs_to_users)

    # 插入顺序即 dict 的键顺序（Python 3.7+ 保序）
    bs_keys = list(bs_to_users.keys())
    counts = np.array([len(bs_to_users[k]) for k in bs_keys], dtype=int)
    return dict(bs_to_users), bs_keys, counts

def allocate_data_to_users(selected_cgi_data_dict, bs_user_dict):
    """
    将每个基站每时刻的总流量，按正态分布比例分配给其服务用户。
    返回：
    - user_data_dict: {user_id: np.array([...])}, shape: (672,)
    """
    # 初始化所有用户的流量序列容器
    user_data_dict = defaultdict(lambda: np.zeros_like(next(iter(selected_cgi_data_dict.values()))))

    T = len(next(iter(selected_cgi_data_dict.values())))  # 时间长度，如 672

    for cgi, data_seq in selected_cgi_data_dict.items():
        user_list = bs_user_dict.get(cgi, [])
        n_users = len(user_list)

        if n_users == 0:
            continue  # 无用户连接，跳过

        for t in range(T):
            total_flow = data_seq[t]

            # 生成正态随机比例（绝对值，归一化）
            z = np.abs(np.random.randn(n_users))  # 保证非负
            weights = z / z.sum()  # 和为1
            flows = weights * total_flow

            for uid, flow in zip(user_list, flows):
                user_data_dict[uid][t] += flow  # 有可能多个基站服务同一用户时会叠加

    return user_data_dict

def aggregate_data_to_grids(pop_100m, user_points, user_data_dict):

    grid_data = np.zeros((pop_100m.shape[0], pop_100m.shape[1], 168))
    grid_data_dict = defaultdict(lambda: np.zeros_like(next(iter(user_data_dict.values()))))

    uid = 0  # 用户编号按 user_points 的顺序生成

    for (i, j) in sorted(user_points.keys()):
        for _ in user_points[(i, j)]:
            if uid in user_data_dict:
                grid_data[i, j] += user_data_dict[uid]
                grid_data_dict[(i, j)] += user_data_dict[uid]
            uid += 1

    return grid_data, grid_data_dict

# -------- 帮助函数 --------
def _align_by_imsi(info_imsi, info_loc, traf_imsi, traf):
    """
    用共同 IMSI 的同一排序对齐位置与流量；返回 loc, data（形状对齐）
    info_loc: (Ni, Ti, 2)   traf: (Nt, Tt)
    """
    common = np.intersect1d(info_imsi, traf_imsi)
    if common.size == 0:
        return None, None

    # 建映射：把两侧都重排到“common 的排序”
    order = np.argsort(common)
    common_sorted = common[order]

    info_pos = np.argsort(info_imsi)
    traf_pos = np.argsort(traf_imsi)
    # 用 searchsorted 找出 common_sorted 在两侧的位置索引
    info_idx = info_pos[np.searchsorted(info_imsi[info_pos], common_sorted)]
    traf_idx = traf_pos[np.searchsorted(traf_imsi[traf_pos], common_sorted)]

    loc = info_loc[info_idx]        # (M, Ti, 2)
    data = traf[traf_idx]        # (M, Tt)
    return loc, data

def _pad_tail_to_96(arr, target_len=96, axis=1):
    """
    用尾值复制把时间维补齐到 target_len。
    - arr: (..., T, ...)；axis 指时间轴
    """
    T = arr.shape[axis]
    if T == target_len:
        return arr
    if T > target_len:
        # 若意外更长，截断到 target_len
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(0, target_len)
        return arr[tuple(sl)]
    # 需要 pad
    pad_k = target_len - T
    # 取最后一个切片，并重复 pad_k 次
    idx_last = [slice(None)] * arr.ndim
    idx_last[axis] = slice(T-1, T)
    last_slice = arr[tuple(idx_last)]
    pad_block = np.repeat(last_slice, pad_k, axis=axis)
    return np.concatenate([arr, pad_block], axis=axis)

def _accumulate_day_to_grid_hourly(locs96, data96, lon_edges, lat_edges):
    """
    把单日 (M,96) 的 15min 流量按当刻位置落格累加到 (Ny,Nx,96)，
    再按每 4 段求和 -> (Ny,Nx,24)
    locs96: (M,96,2)  (lon,lat)
    data96: (M,96)
    """
    Ny = len(lat_edges) - 1
    Nx = len(lon_edges) - 1

    # 提取 lon/lat；形状 (M,96)
    lons = locs96[..., 0]
    lats = locs96[..., 1]

    # 用边界分箱（左闭右开 [,)）
    ix = np.digitize(lons, lon_edges, right=False) - 1  # 列（经度）
    iy = np.digitize(lats, lat_edges, right=False) - 1  # 行（纬度）

    # 有效格
    valid = (ix >= 0) & (ix < Nx) & (iy >= 0) & (iy < Ny)

    # 时间索引 0..95
    M, T = data96.shape
    t_idx = np.broadcast_to(np.arange(T), (M, T))

    # 累加到 (Ny, Nx, 96)
    grid_15m = np.zeros((Ny, Nx, 96), dtype=np.float32)
    np.add.at(grid_15m, (iy[valid], ix[valid], t_idx[valid]), data96[valid].astype(np.float32))

    # 15min → 1h（每 4 段求和）：(Ny, Nx, 24)
    hourly = grid_15m.reshape(Ny, Nx, 24, 4).sum(axis=-1, dtype=np.float32)
    return hourly  # (Ny, Nx, 24)

# -------- 主函数：一周聚合 --------
def compute_week_grid_traffic_hourly(
    day_tags,  # 例如 ["0908","0909",...,"0914"]
    base_dir="./用户流量/week_1",
    lon_edges=None,
    lat_edges=None
):
    """
    读取 7 天 *单日* 文件，输出：
      grid_dict[(j,i)] = np.ndarray(shape=(168,), dtype=float32)
    说明：j=纬度行索引，i=经度列索引
    """
    assert lon_edges is not None and lat_edges is not None, "请先提供 lon_edges/lat_edges。"
    Ny = len(lat_edges) - 1
    Nx = len(lon_edges) - 1

    # 累加一周（按天拼接）
    week_cube = np.zeros((Ny, Nx, 0), dtype=np.float32)  # 将逐日 (Ny,Nx,24) 在时间维拼接

    for tag in day_tags:
        info_path = f"{base_dir}/model_test{tag}.npz"
        traf_path = f"{base_dir}/test_{tag}.npz"

        # 读单日
        f_info = np.load(info_path, allow_pickle=True)
        f_traf = np.load(traf_path, allow_pickle=True)

        info_imsi_raw = f_info["imsi"]
        loc_raw = f_info["loc"] / 1e7            # 形状可能是 (Ni, 94/95, 2)，经纬顺序为 (lon, lat)
        traf_imsi_raw = f_traf["imsi"]
        traf_raw = f_traf["traffic"]       # 形状可能是 (Nt, 94/95)
        np.nan_to_num(traf_raw, nan=0.0, copy=False)

        # —— IMSI 对齐（稳定排序）——
        loc, traf = _align_by_imsi(info_imsi_raw, loc_raw, traf_imsi_raw, traf_raw)
        if loc is None:
            # 没有共同 IMSI，当天全 0
            day_hourly = np.zeros((Ny, Nx, 24), dtype=np.float32)
            week_cube = np.concatenate([week_cube, day_hourly], axis=2)
            continue

        # —— 用尾值复制补齐到 96（位置与流量都要补）——
        loc96   = _pad_tail_to_96(loc,   target_len=96, axis=1)    # (M,96,2)
        traf96  = _pad_tail_to_96(traf,  target_len=96, axis=1)    # (M,96)

        # —— 当天 15min → 落格累加 → 1h（24）——
        day_hourly = _accumulate_day_to_grid_hourly(loc96, traf96, lon_edges, lat_edges)  # (Ny,Nx,24)

        # —— 拼进一周 —— 
        week_cube = np.concatenate([week_cube, day_hourly], axis=2)

    # 输出字典：仅保留非全零的格，节省空间
    grid_dict = {}
    for j in range(Ny):
        for i in range(Nx):
            series = week_cube[j, i]  # (Tweek,)
            grid_dict[(j, i)] = series.astype(np.float32)

    return grid_dict, week_cube  # grid_dict(稀疏)，以及完整三维 (Ny,Nx,168)（需要时可用）

def assign_grid_AOI(lon_edges, lat_edges, aoi_path, aoi_name_field=None):
    Ny, Nx = len(lat_edges) - 1, len(lon_edges) - 1
    
    # 1. 构造中心点坐标
    lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_grid, lat_grid = np.meshgrid(lon_c, lat_c)

    # 2. 转为 GeoDataFrame 点
    pts = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(lon_grid.ravel(), lat_grid.ravel())],
        crs='EPSG:4326'
    )

    # 3. 读取 AOI 多边形
    aoi = gpd.read_file(aoi_path)
    if aoi.crs != pts.crs:
        aoi = aoi.to_crs(pts.crs)
    
    # print(aoi.columns)
    # print(aoi.head())

    # 4. 空间连接
    joined = gpd.sjoin(pts, aoi, how='left', predicate='within')

    # 5. 获取 AOI 名称字段或索引编号
    if aoi_name_field is None:
        # 自动找第一个非几何字段
        aoi_name_field = next(col for col in aoi.columns if col != 'geometry')
    aoi_ids = joined["Level1"].fillna(-1)

    # 6. 重塑为 (Ny, Nx)
    return np.array(aoi_ids).reshape((Ny, Nx))

def count_poi_by_grid(
    poi_df: pd.DataFrame,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    label_map: dict,
    lon_col: str = "lon",        # 经度列名（float）
    lat_col: str = "lat",        # 纬度列名（float）
    cat_col: str = "行业大类",     # 类别列名
    dtype=np.int32
):
    """
    将 POI 统计到规则网格中，返回 (R, C, num_classes) 计数张量。
    约定分箱为左闭右开 [edge[i], edge[i+1])，但包含最右/最上边界上的点。
    """
    # 基本参数
    R = len(lat_edges) - 1
    C = len(lon_edges) - 1
    num_classes = len(label_map)

    # 取出需要的列
    lon = poi_df[lon_col].to_numpy(dtype=float, copy=False)
    lat = poi_df[lat_col].to_numpy(dtype=float, copy=False)
    cat = poi_df[cat_col].astype(str).to_numpy()

    # 类别映射：不可识别的类别置为 -1（后续过滤）
    # 更高效的做法：先构造一个 vectorized map
    # 注意：如果类别很杂，可以先做 pd.Series.map 再填充 -1
    cat_idx = np.fromiter((label_map.get(x, -1) for x in cat), dtype=int, count=len(cat))

    # 仅保留落在外包矩形范围内的点（含上边界）
    in_lon = (lon >= lon_edges[0]) & (lon <= lon_edges[-1])
    in_lat = (lat >= lat_edges[0]) & (lat <= lat_edges[-1])
    in_cat = cat_idx >= 0
    mask = in_lon & in_lat & in_cat
    if not np.any(mask):
        return np.zeros((R, C, num_classes), dtype=dtype)

    lon = lon[mask]; lat = lat[mask]; cat_idx = cat_idx[mask]

    # 计算每个点所在的列/行索引（左闭右开），边界点（==最大边）并入最后一格
    # searchsorted(..., 'right') - 1 可确保落在左侧区间；对等于边界值的点，落到其左侧格
    cols = np.searchsorted(lon_edges, lon, side='right') - 1
    rows = np.searchsorted(lat_edges, lat, side='right') - 1

    # 特别处理最右/最上边界：等于最大边界的点会得到 index==C 或 R，扣回到最后一格
    cols = np.where(cols == C, C - 1, cols)
    rows = np.where(rows == R, R - 1, rows)

    # 再做一次有效性过滤（防止越界或数值异常）
    valid = (rows >= 0) & (rows < R) & (cols >= 0) & (cols < C)
    if not np.any(valid):
        return np.zeros((R, C, num_classes), dtype=dtype)

    rows = rows[valid]; cols = cols[valid]; cat_idx = cat_idx[valid]

    # 累加计数
    counts = np.zeros((R, C, num_classes), dtype=dtype)
    np.add.at(counts, (rows, cols, cat_idx), 1)
    return counts

def aggregate_5x5_blocks(arr):
    """将100m数据聚合为500m（每5x5个格子加总）"""
    H, W = arr.shape[:2]
    h5, w5 = H // 5 * 5, W // 5 * 5  # 裁剪成5的整数倍
    arr = arr[:h5, :w5]

    if arr.ndim == 2:
        return arr.reshape(h5//5, 5, w5//5, 5).sum(axis=(1, 3))  # → (H//5, W//5)
    
    elif arr.ndim == 3:
        D = arr.shape[2]
        return arr[:h5, :w5, :].reshape(h5//5, 5, w5//5, 5, D).sum(axis=(1, 3))  # → (H//5, W//5, D)
    
    else:
        raise ValueError("Only support 2D or 3D arrays")

def aggregate_edges_5x5(lat_edges, lon_edges):
    """
    根据100m格网边界生成500m格网边界
    lat_edges: (Ny+1,)
    lon_edges: (Nx+1,)
    """
    Ny = len(lat_edges) - 1
    Nx = len(lon_edges) - 1

    # 取每5个格子的边界
    lat_edges_500m = lat_edges[np.arange(0, Ny+1, 5)]
    lon_edges_500m = lon_edges[np.arange(0, Nx+1, 5)]

    # # 若不是5的整数倍，补上最末边界（仅在确实缺时）
    # if lat_edges_500m[-1] < lat_edges[-1] - 1e-10:
    #     lat_edges_500m = np.append(lat_edges_500m, lat_edges[-1])
    # if lon_edges_500m[-1] < lon_edges[-1] - 1e-10:
    #     lon_edges_500m = np.append(lon_edges_500m, lon_edges[-1])

    return lat_edges_500m, lon_edges_500m

def aggregate_aoi_to_vector(aoi_arr):
    """将aoi_100m 聚合为 shape=(H//5, W//5, 25)，每个500m格子是25维向量"""
    H, W = aoi_arr.shape
    h5, w5 = H // 5 * 5, W // 5 * 5
    aoi_arr = aoi_arr[:h5, :w5]
    out = aoi_arr.reshape(h5//5, 5, w5//5, 5).transpose(0, 2, 1, 3).reshape(h5//5, w5//5, 25)
    return out  # shape = (H//5, W//5, 25)

def reshape_4x4_blocks(arr):
    """将500m数据分块成2000m，输出 shape=(H//20, W//20, 4, 4, ...)"""
    H, W = arr.shape[:2]
    h4, w4 = H // 4 * 4, W // 4 * 4
    arr = arr[:h4, :w4]

    if arr.ndim == 2:
        return arr.reshape(h4//4, 4, w4//4, 4).transpose(0, 2, 1, 3)  # → (N1, N2, 4, 4)
    
    elif arr.ndim == 3:
        D = arr.shape[2]
        return arr.reshape(h4//4, 4, w4//4, 4, D).transpose(0, 2, 1, 3, 4)  # → (N1, N2, 4, 4, D)
    
    else:
        raise ValueError("Only support 2D or 3D arrays")

def get_loc_2000m(lat_edges_500m, lon_edges_500m):
    """
    根据500m格网边界生成2000m格网的边界坐标
    返回 loc_2000m: shape (Nx_2000, Ny_2000, 4)
    每个单元为 [lat_min, lat_max, lon_min, lon_max]
    """
    Ny_500 = len(lat_edges_500m) - 1
    Nx_500 = len(lon_edges_500m) - 1

    Ny_2000 = Ny_500 // 4
    Nx_2000 = Nx_500 // 4

    loc_2000m = np.zeros((Ny_2000, Nx_2000, 4), dtype=float)

    for i in range(Ny_2000):
        lat_min = lat_edges_500m[i * 4]
        lat_max = lat_edges_500m[(i + 1) * 4]
        for j in range(Nx_2000):
            lon_min = lon_edges_500m[j * 4]
            lon_max = lon_edges_500m[(j + 1) * 4]
            loc_2000m[i, j] = [lat_min, lat_max, lon_min, lon_max]

    return loc_2000m

def read_osm_to_proj(path: str, crs_proj):
    """
    读取 Geofabrik/OSM shapefile（WGS84 & UTF-8），
    并重投影到 crs_proj（用于栅格统计的米坐标）。
    """
    # 1) 按 UTF-8 读取（即使有 .cpg 也显式声明，避免平台差异）
    gdf = gpd.read_file(path, encoding="utf-8")
    CRS_WGS84 = CRS.from_epsg(4326)

    # 2) 确认/设置 CRS=EPSG:4326（有些数据会缺失 .prj）
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_WGS84, allow_override=True)
    else:
        # 若不是 4326，先转回 4326（理论上 Geofabrik 就是 4326，这里是保险）
        if not CRS(gdf.crs).equals(CRS_WGS84):
            gdf = gdf.to_crs(CRS_WGS84)

    # 3) 转到你的投影坐标（米）
    gdf = gdf.to_crs(crs_proj)
    return gdf

def make_grid_gdf(x_edges, y_edges, crs, Nx, Ny):
    polys, ix_list, iy_list = [], [], []
    for iy in range(Ny):
        y0, y1 = y_edges[iy], y_edges[iy+1]
        for ix in range(Nx):
            x0, x1 = x_edges[ix], x_edges[ix+1]
            polys.append(Polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)]))
            ix_list.append(ix); iy_list.append(iy)
    grid = gpd.GeoDataFrame({"ix": ix_list, "iy": iy_list, "geometry": polys}, crs=crs)
    return grid

def ensure_proj(gdf, crs_target):
    if gdf is None: return None
    if gdf.crs != crs_target:
        gdf = gdf.to_crs(crs_target)
    return gdf

def count_by_centroid(gdf, out_arr, Nx, Ny, x_edges, y_edges):
    # 支持点或面（面用质心）
    if gdf is None or gdf.empty: return
    geom = gdf.geometry
    if not all(geom.geom_type == "Point"):
        geom = geom.centroid
    xs, ys = geom.x.values, geom.y.values
    ix = np.searchsorted(x_edges, xs, side="right") - 1
    iy = np.searchsorted(y_edges, ys, side="right") - 1
    mask = (ix>=0)&(ix<Nx)&(iy>=0)&(iy<Ny)
    np.add.at(out_arr, (iy[mask], ix[mask]), 1)

def accumulate_length_in_cells(gdf_lines, out_arr, grid, grid_sindex):
    # 线要素：求与每个 cell 的交线长度（米）
    if gdf_lines is None or gdf_lines.empty: return
    for geom in gdf_lines.geometry:
        if geom is None or geom.is_empty: continue
        # 候选网格（用空间索引按包围盒粗筛）
        hits = list(grid_sindex.intersection(geom.bounds))
        for h in hits:
            cell = grid.geometry.iloc[h]
            inter = geom.intersection(cell)
            if inter.is_empty: continue
            out_arr[grid.iy.iloc[h], grid.ix.iloc[h]] += inter.length

def accumulate_area_in_cells(gdf_polys, out_arr, grid, grid_sindex):
    # 面要素：求与每个 cell 的交面面积（㎡）
    if gdf_polys is None or gdf_polys.empty: return
    for geom in gdf_polys.geometry:
        if geom is None or geom.is_empty: continue
        hits = list(grid_sindex.intersection(geom.bounds))
        for h in hits:
            cell = grid.geometry.iloc[h]
            inter = geom.intersection(cell)
            if inter.is_empty: continue
            out_arr[grid.iy.iloc[h], grid.ix.iloc[h]] += inter.area