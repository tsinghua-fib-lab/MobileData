import argparse
import numpy as np
import torch
import datetime
import json
import yaml
import os
try:
    import setproctitle
except ImportError:
    setproctitle = None
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass
from main_model import CSDI_Value
from inference.dataset_process import get_dataloader
from inference.utils import train, evaluate
from joblib import load
import joblib
import pickle
import shapefile
from matplotlib.path import Path
from paths import CITY_EN, ZOOMDIFF_DIR, city_cond_file

dataset_list = 'Nanchang*Nanjing'
datatype = 'traffic' # user
task_state = 'test' #'train', 'test', 'zero-shot', 'few-shot'
fewshot_rate = 0.1
modelfolder = ''

parser = argparse.ArgumentParser(description="Multi-scale CSDI")

parser.add_argument("--dataset", type=str, default=dataset_list)
parser.add_argument("--datatype", type=str, default=datatype)
parser.add_argument('--device', default='cuda:0', help='Device for Attack')

parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)
parser.add_argument(
    "--shp_path",
    type=str,
    default=os.path.join(
        ZOOMDIFF_DIR,
        "datasets",
        "_shared_geographic_data",
        "China_city_boundaries",
        "china_city_boundaries_2024.shp",
    ),
    help="Path to the city boundary shapefile used to mask 500m outputs",
)

args = parser.parse_args()
args.task_state = task_state
args.fewshot_rate = fewshot_rate
if setproctitle is not None:
    setproctitle.setproctitle("Multi-scale CSDI-" + args.dataset + "@qxq")
print(args)

path = os.path.join(ZOOMDIFF_DIR, "config", args.config)
with open(path, "r") as f:
    config = yaml.safe_load(f)

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = os.path.join(
    ZOOMDIFF_DIR,
    "save",
    f"{args.datatype}_{args.dataset}_t{str(config['diffusion']['multi_scale']['time_scale'])}s{str(config['diffusion']['multi_scale']['spatial_scale'])}_{current_time}",
)
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(os.path.join(foldername, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

args.config = config
args, test_loader, shape_2000_all = get_dataloader(
    args=args,
    batch_size=config["train"]["batch_size"],
)

model = CSDI_Value(args).to(args.device)

if args.task_state in ['test', 'zero-shot', 'few-shot']:
    args.modelfolder = modelfolder
    model.load_state_dict(torch.load(os.path.join(ZOOMDIFF_DIR, "save", args.modelfolder, "model.pth")))
    scaler = load(os.path.join(ZOOMDIFF_DIR, "datasets", f"template_scaler_{datatype}.pkl"))

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total Parameters:", total_params)

writer = SummaryWriter(log_dir=os.path.join(foldername, "logs"), flush_secs=5)

evaluate(args, model, scaler, shape_2000_all, test_loader, foldername=foldername)

# Sort out the data from the output results
result_path = os.path.join(foldername, "generated_outputs_nsample" + str(args.nsample) + ".pk")
with open(result_path, 'rb') as f:
    all_datatype, all_samples, all_samples_scale, _, all_observed_time, all_observed_loc, all_scalers = pickle.load(f)

typelist = np.unique(all_datatype)
print("Generated City List: ", typelist)

# Load Template Data (to correct the inverse normalization value)
file_TP = np.load(os.path.join(ZOOMDIFF_DIR, "datasets", f"template_{datatype}.npz"), allow_pickle=True)
pop_TP = file_TP["pop_2000m"]
data_TP = file_TP["data_2000m"]
mean_TP = data_TP.mean()
max_TP = data_TP.max()
scaler_TP = joblib.load(os.path.join(ZOOMDIFF_DIR, "datasets", f"template_scaler_{datatype}.pkl"))

CITY_CN = {v: k for k, v in CITY_EN.items()}

def load_city_boundaries(shp_path):
    sf = shapefile.Reader(shp_path, encoding="utf-8")
    field_names = [field[0] for field in sf.fields[1:]]
    return field_names, sf.records(), sf.shapes()


def get_city_boundary_path(city, field_names, records, shapes):
    city = str(city)
    city_cn = CITY_CN.get(city, city)
    candidates = {city, city_cn, f"{city_cn}市"}

    for record, shape in zip(records, shapes):
        values = dict(zip(field_names, record))
        searchable = [
            str(values.get("地名", "")),
            str(values.get("地级", "")),
            str(values.get("ENG_NAME", "")),
            str(values.get("NAME_2", "")),
        ]
        if any(
            candidate == value or candidate in value or value in candidate
            for candidate in candidates
            for value in searchable
            if candidate and value
        ):
            vertices = []
            codes = []
            parts = list(shape.parts) + [len(shape.points)]
            for i in range(len(parts) - 1):
                pts = shape.points[parts[i]:parts[i + 1]]
                if not pts:
                    continue
                vertices.append(pts[0])
                codes.append(Path.MOVETO)
                for point in pts[1:]:
                    vertices.append(point)
                    codes.append(Path.LINETO)
                if pts[-1] != pts[0]:
                    vertices.append(pts[0])
                    codes.append(Path.CLOSEPOLY)
                else:
                    codes[-1] = Path.CLOSEPOLY
            if vertices:
                return Path(vertices, codes)

    raise ValueError(f"Could not find city boundary in shapefile for {city}")


def convert_2000m_to_masked_500m(data_dict, city, field_names, records, shapes):
    loc_2000m = data_dict["loc_2000m"]
    data_2000m = data_dict["data_2000m"]

    H_area, W_area = data_2000m.shape[:2]
    if loc_2000m.ndim == 3 and loc_2000m.shape[:2] == (W_area, H_area):
        loc_2000m = loc_2000m.transpose(1, 0, 2)
    if loc_2000m.shape[:2] != (H_area, W_area):
        H_area, W_area = loc_2000m.shape[:2]

    H_sub, W_sub = 4, 4
    H_total = H_area * H_sub
    W_total = W_area * W_sub

    lats_1d = np.zeros(H_total)
    for i in range(H_area):
        rmin = loc_2000m[i, :, 0].min()
        rmax = loc_2000m[i, :, 1].max()
        step = (rmax - rmin) / H_sub
        lats_1d[i * H_sub:(i + 1) * H_sub] = np.linspace(
            rmin + step / 2, rmax - step / 2, H_sub
        )

    lons_1d = np.zeros(W_total)
    for j in range(W_area):
        cmin = loc_2000m[:, j, 2].min()
        cmax = loc_2000m[:, j, 3].max()
        step = (cmax - cmin) / W_sub
        lons_1d[j * W_sub:(j + 1) * W_sub] = np.linspace(
            cmin + step / 2, cmax - step / 2, W_sub
        )

    xx, yy = np.meshgrid(lons_1d, lats_1d)
    points = np.vstack((xx.ravel(), yy.ravel())).T
    city_path = get_city_boundary_path(city, field_names, records, shapes)
    mask_2d = city_path.contains_points(points).reshape(H_total, W_total)

    output_data = {}
    for key, arr in data_dict.items():
        if key == "loc_2000m" or "2000m" not in key:
            continue

        arr_500m = None
        if arr.ndim == 5:
            if arr.shape[2] == 4 and arr.shape[3] == 4:
                temp = arr
            elif arr.shape[3] == 4 and arr.shape[4] == 4:
                temp = arr.transpose(0, 1, 3, 4, 2)
            else:
                temp = None
            if temp is not None:
                arr_500m = temp.transpose(0, 2, 1, 3, 4).reshape(H_total, W_total, -1)
        elif arr.ndim == 4 and arr.shape[2] == 4 and arr.shape[3] == 4:
            arr_500m = arr.transpose(0, 2, 1, 3).reshape(H_total, W_total)
        elif arr.ndim in (2, 3) and arr.shape[:2] == (H_area, W_area):
            arr_500m = np.repeat(np.repeat(arr, H_sub, axis=0), W_sub, axis=1)

        if arr_500m is not None:
            output_data[key.replace("2000m", "500m")] = arr_500m[mask_2d]

    output_data["lat"] = yy[mask_2d]
    output_data["lon"] = xx[mask_2d]
    return output_data

boundary_field_names, boundary_records, boundary_shapes = load_city_boundaries(args.shp_path)

for city in typelist:

    print(city)
    samples = all_samples[city]
    H_areas, M_areas, M, L = samples.shape[:4]
    H = samples.shape[-1]

    save_gen = samples.reshape(H_areas, M_areas, M, L, H, H)
    save_gen = np.clip(save_gen, a_min=0, a_max=None)

    data_file = np.load(city_cond_file(city), allow_pickle=True)
    data_dict = dict(data_file)

    data_save = save_gen.mean(axis=2)
    pop_save = data_dict['pop_2000m']
    data_norm = scaler_TP.transform(data_save.reshape(-1, 1)).reshape(data_save.shape)

    # Population Ratio
    ratio = pop_save.mean() / pop_TP.mean()
    print("pop_ratio = ", ratio)

    data_denorm = scaler_TP.inverse_transform(data_norm.reshape(-1, 1)).reshape(data_save.shape)
    mean_denorm = data_denorm.mean()
    k = ratio * mean_TP / mean_denorm
    print("Correction Factor k = ", k)

    data_rescaled = data_denorm * k

    def adjust_peak_to_mean(data, target_mean, target_pmr, verbose=True):
        """
        data: 已 rescale 的南京数据（均值已正确）
        target_mean: 目标均值（南昌均值 × 人口比例）
        target_pmr: 南昌的 peak:mean 比例
        """

        # 原始数据
        x = data.copy()
        x_mean0 = x.mean()
        x_max0 = x.max()
        pmr0 = x_max0 / x_mean0

        if verbose:
            print(f"[Before] mean={x_mean0:.4f}, max={x_max0:.4f}, PMR={pmr0:.4f}")

        # 搜索 gamma
        gammas = np.linspace(0.3, 3.5, 200)
        best_gamma = None
        best_diff = 1e9

        for g in gammas:
            xg = x ** g
            xg_mean = xg.mean()
            xg_max = xg.max()
            pmr_g = xg_max / xg_mean
            diff = abs(pmr_g - target_pmr)
            if diff < best_diff:
                best_diff = diff
                best_gamma = g

        if verbose:
            print(f"Best gamma = {best_gamma:.4f}, PMR diff = {best_diff:.6f}")

        # 使用最佳 gamma
        xg = x ** best_gamma
        
        # 计算线性系数 c，使均值回到 target_mean
        c = target_mean / xg.mean()
        
        x_final = xg * c

        if verbose:
            print(f"[After]  mean={x_final.mean():.4f}, "
                f"max={x_final.max():.4f}, "
                f"PMR={x_final.max()/x_final.mean():.4f}")

        return x_final, best_gamma, c

    target_mean = mean_TP * ratio
    target_pmr = max_TP / mean_TP
    data_final, gamma_used, c_used = adjust_peak_to_mean(
        data_rescaled, 
        target_mean=target_mean,
        target_pmr=target_pmr,
        verbose=True
    )
    data_dict["data_2000m"] = data_final
    data_500m = convert_2000m_to_masked_500m(
        data_dict,
        city,
        boundary_field_names,
        boundary_records,
        boundary_shapes,
    )
    results_dir = os.path.join(ZOOMDIFF_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"{city}_500m_{datatype}.npz")
    np.savez(result_file, **data_500m)
    print(f"Final Results have saved at {result_file}")
