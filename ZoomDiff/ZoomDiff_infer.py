import argparse
import numpy as np
import torch
import datetime
import json
import yaml
import os
import setproctitle
from torch.utils.tensorboard import SummaryWriter
from main_model import CSDI_Value
from inference.dataset_process import get_dataloader
from inference.utils import train, evaluate
from joblib import load
import joblib
import pickle

dataset_list = '成都' # 成都*呼和浩特*南阳*唐山*烟台*阳江*长春*珠海*驻马店
# 澳门*重庆*福州*广州*贵阳*哈尔滨*海口*合肥*昆明*拉萨*兰州*沈阳*石家庄*太原*天津*乌鲁木齐*武汉*西安*西宁*香港*银川*长沙*郑州
datatype = 'traffic' # user
task_state = 'test' #'train', 'test', 'zero-shot', 'few-shot'
fewshot_rate = 0.1
modelfolder = ''

parser = argparse.ArgumentParser(description="Multi-scale CSDI")

parser.add_argument("--dataset", type=str, default=dataset_list)
parser.add_argument("--datatype", type=str, default=datatype)
parser.add_argument('--device', default='cuda:3', help='Device for Attack')

parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)

args = parser.parse_args()
args.task_state = task_state
args.fewshot_rate = fewshot_rate
setproctitle.setproctitle("Multi-scale CSDI-" + args.dataset + "@qxq")
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = f"./save/{args.datatype}_{args.dataset}_t{str(config['diffusion']['multi_scale']['time_scale'])}s{str(config['diffusion']['multi_scale']['spatial_scale'])}_{current_time}/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

args.config = config
args, test_loader, shape_2000_all = get_dataloader(
    args=args,
    batch_size=config["train"]["batch_size"],
)

model = CSDI_Value(args).to(args.device)

if args.task_state in ['test', 'zero-shot', 'few-shot']:
    args.modelfolder = modelfolder
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
    scaler = load(f'./datasets/template_scaler_{datatype}.pkl')

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total Parameters:", total_params)

writer = SummaryWriter(log_dir=foldername + 'logs/', flush_secs=5)

evaluate(args, model, scaler, shape_2000_all, test_loader, foldername=foldername)

# Sort out the data from the output results
result_path = foldername + '/generated_outputs_nsample' + str(args.nsample) + '.pk' 
with open(result_path, 'rb') as f:
    all_datatype, all_samples, all_samples_scale, _, all_observed_time, all_observed_loc, all_scalers = pickle.load(f)

typelist = np.unique(all_datatype)
print("Generated City List: ", typelist)

# Load Template Data (to correct the inverse normalization value)
file_TP = np.load(f"./datasets/template_{datatype}.npz", allow_pickle=True)
pop_TP = file_TP["pop_2000m"]
data_TP = file_TP["data_2000m"]
mean_TP = data_TP.mean()
max_TP = data_TP.max()
scaler_TP = joblib.load(f'./datasets/template_scaler_{datatype}.pkl')

for city in typelist:

    print(city)
    samples = all_samples[city]
    H_areas, M_areas, M, L = samples.shape[:4]
    H = samples.shape[-1]

    save_gen = samples.reshape(H_areas, M_areas, M, L, H, H)
    save_gen = np.clip(save_gen, a_min=0, a_max=None)

    data_file = np.load(f"./datasets/cond/{city}_cond.npz", allow_pickle=True)
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
    np.savez(f"./results/{city}_{datatype}.npz", **data_dict)
    print(f"Final Results have saved at ./results/{city}_{args.datatype}.npz")