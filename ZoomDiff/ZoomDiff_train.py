import argparse
import torch
import datetime
import json
import yaml
import os
import setproctitle
from torch.utils.tensorboard import SummaryWriter
from main_model import CSDI_Value
from dataset_process import get_dataloader
from utils import train, evaluate
from joblib import dump
from model_freeze import freeze_all, unfreeze_fewshot_modules

dataset_list = '南昌' # 北京*富阳*济南*南昌*南京*南宁*上海
datatype = 'traffic' # user
task_state = 'train' #'train', 'test', 'zero-shot', 'few-shot'
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
parser.add_argument("--nsample", type=int, default=5)

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
args, train_loader, valid_loader, test_loader, scaler_all, _ = get_dataloader(
    args=args,
    batch_size=config["train"]["batch_size"],
)
# dump(scaler, f'{foldername}global_scaler.joblib')
# print(f"全局Scaler已保存至: {foldername}global_scaler.joblib")
os.makedirs(f'{foldername}scalers/', exist_ok=True)
for city, scaler in scaler_all.items():
    
    dump(scaler, f'{foldername}scalers/scaler_{city}_{datatype}.pkl')

model = CSDI_Value(args).to(args.device)

if args.task_state in ['test', 'zero-shot', 'few-shot']:
    args.modelfolder = modelfolder
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total Parameters:", total_params)

writer = SummaryWriter(log_dir=foldername + 'logs/', flush_secs=5)

if args.task_state in ['train', 'few-shot']:
    if args.task_state == 'few-shot':
        config["train"]["epochs"] = 50

        freeze_all(model)
        unfreeze_fewshot_modules(model)
    
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
        writer=writer,
    )

evaluate(args, model, scaler_all, test_loader, foldername=foldername)