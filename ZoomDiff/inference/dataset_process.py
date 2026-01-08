import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# 基本参数
is_normalize = True
time_length = 168

def scale_preprocess_spatial(data, spatial_scale):
    if len(data.shape) == 4:
        B, D, H, _ = data.shape
        # 空间粒度拆解
        s_division_level = 2 ** (spatial_scale - 1) # 对边长几等分
        s_granularity = H // s_division_level # 空间粒度
        X = data.reshape(B, D, s_division_level, s_granularity, H)
        X = X.mean(axis=-2) # (B, D, s_division_level, H)
        X = X.reshape(B, D, s_division_level, s_division_level, s_granularity)
        X = X.mean(axis=-1) # (B, D, s_division_level, s_division_level)

    elif len(data.shape) == 5:
        B, D, L, H, _ = data.shape
        # 空间粒度拆解
        s_division_level = 2 ** (spatial_scale - 1) # 对边长几等分
        s_granularity = H // s_division_level # 空间粒度
        X = data.reshape(B, D, L, s_division_level, s_granularity, H)
        X = X.mean(axis=-2) # (B, D, L, s_division_level, H)
        X = X.reshape(B, D, L, s_division_level, s_division_level, s_granularity)
        X = X.mean(axis=-1) # (B, D, L, s_division_level, s_division_level)
    
    return X

class Traffic_Dataset(Dataset):
    def __init__(self, config, city, eval_length=time_length, data_num=100, use_index_list=None, seed=0):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        # 数据集加载 ————————————————————————————————————————————————————————————————————————————
        data, st_info, poi_emb, loc, self.shape_2000 = raw_load(city, data_num)

        spatial_scale = config["diffusion"]["multi_scale"]["spatial_scale"]
        time_scale = config["diffusion"]["multi_scale"]["time_scale"]
        spatial_scale_coarse = config["diffusion"]["multi_scale"]["spatial_scale_coarse"]

        data = scale_preprocess_spatial(data, spatial_scale_coarse)
        st_info = scale_preprocess_spatial(st_info, spatial_scale_coarse) # 静态条件不需要时间分解
        poi_emb = scale_preprocess_spatial(poi_emb.transpose(0, 3, 1, 2), spatial_scale_coarse) # (N, 21, H, H)

        data = np.clip(data, a_min=None, a_max=np.percentile(data, 99))

        N, K, L, H, _ = data.shape
        self.observed_series = data # (N, K, L, H, H)
        self.observed_cond_static = st_info.reshape(N, -1, H, H) # (N, Ds, H, H)
        self.observed_cond_poi = poi_emb.reshape(N, -1, H, H) # (N, 48, H, H)
        self.observed_loc = loc

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_series))
            
            # 计算细分时空粒度
            if time_scale == 5: # (B, K, L, H, H)
                time_granularity = 168
            elif time_scale == 4:
                time_granularity = 24
            else:
                time_granularity = 2 ** (time_scale - 1)

            print('最粗时间粒度：', time_granularity, 'h')
            print('最粗空间粒度：', 50 * (2 ** (4 - spatial_scale_coarse)) * (2 ** (spatial_scale - 1)), 'm')
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_series": self.observed_series[index],
            "observed_cond_static": self.observed_cond_static[index],
            "observed_cond_poi": self.observed_cond_poi[index],
            "observed_loc": self.observed_loc[index],
            "timepoints": np.arange(self.eval_length),
            "idex": index,
        }

        return s

    def __len__(self):
        return len(self.use_index_list)

def get_dataloader(args, batch_size=16):

    # train_loader_all = []
    # valid_loader_all = []
    test_loader_all = []
    shape_2000_all = {}

    seed = args.seed

    for city in args.dataset.split('*'):

        # 创建数据集实例
        dataset = Traffic_Dataset(config=args.config, city=city, seed=seed)
        indlist = dataset.__len__()
        print("数据集规模：", indlist)

        if args.task_state in ['test']:
       
            test_index = np.arange(dataset.__len__())

            test_dataset = Traffic_Dataset(config=args.config, city=city, use_index_list=test_index, seed=seed)
            print("测试集规模：", test_dataset.__len__())

            shape_2000_all[city] = test_dataset.shape_2000
            print(f"{city}2km网格规模：{shape_2000_all[city]}")

            args.H = test_dataset.observed_cond_static.shape[-1]
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        test_loader_all.append([city, test_loader])

    test_loader_all = [(name, i) for name, data in test_loader_all for i in data]

    return args, test_loader_all, shape_2000_all

def raw_load(city, data_num):
    
    file = np.load(f'./datasets/cond/{city}_cond.npz', allow_pickle=True)

    pop = file['pop_2000m']
    building = file['pop_2000m']
    roadlen = file['roadlen_2000m']
    water_area = file['water_area_2000m']
    poi = file['poi_2000m']
    loc = file['loc_2000m']
   
    H_area, W_area, H, W =  pop.shape
    data = np.random.rand(H_area*W_area, 168, H, H) #, dtype=np.float32) # (N, L, H, H)
    pop = np.array(pop, dtype=np.float32).reshape(-1, 4, 4) # (N, H, H)
    building = np.array(building, dtype=np.float32).reshape(-1, 4, 4) # (N, H, H)
    roadlen = np.array(roadlen, dtype=np.float32).reshape(-1, 4, 4) # (N, H, H)
    water_area = np.array(water_area, dtype=np.float32).reshape(-1, 4, 4) # (N, H, H)
    poi_emb = np.array(poi, dtype=np.float32).reshape(-1, 4, 4, 21) # (N, H, H, 21)
    loc = np.array(loc, dtype=np.float32).reshape(-1, 4) # (N, 4)

    # 时不变筛选
    is_constant_per_pixel = np.all(np.var(data, axis=-1) == 0, axis=(1, 2))

    # 取出“时变”的样本
    valid_idx = ~is_constant_per_pixel
    data = data[valid_idx]
    pop = pop[valid_idx]
    building = building[valid_idx]
    roadlen = roadlen[valid_idx]
    water_area = water_area[valid_idx]
    poi_emb = poi_emb[valid_idx]
    loc = loc[valid_idx]
    
    data = np.expand_dims(data, axis=1) # (N, K, L, H, H)
    st_info = np.concatenate([np.expand_dims(pop, axis=1), np.expand_dims(building, axis=1), np.expand_dims(roadlen, axis=1), np.expand_dims(water_area, axis=1)], axis=1) # (N, Ds, H, H), 静态条件信息

    shape_2000 = [H_area, W_area]
    return data[:data_num], st_info[:data_num], poi_emb[:data_num], loc[:data_num], shape_2000