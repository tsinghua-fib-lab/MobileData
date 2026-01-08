import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
from scipy.spatial import distance
import math
import torch.nn as nn
import torch.nn.utils as utils

# 梯度裁剪阈值
MAX_NORM = 1.0

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=100,
    foldername="",
    writer=None,
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)

    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[p1, p2],
                                                        gamma=0.1)

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()

                if writer is not None:
                    writer.add_scalar('train/loss', loss.item(), epoch_no)

                # 梯度裁剪
                utils.clip_grad_norm_(model.parameters(), MAX_NORM)

                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0,
                          maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        if writer is not None:
                            writer.add_scalar('valid/loss', loss.item(), epoch_no)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss":
                                avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )

            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))

def evaluate(args, model, scaler_all, test_loader, foldername=""):
    nsample = args.nsample
    
    print('testnsample=',nsample)
    with torch.no_grad(): # 临时禁用梯度计算
        model.eval()
        all_target = {name:[] for name in args.dataset.split('*')}
        all_observed_time = {name:[] for name in args.dataset.split('*')}
        all_observed_loc = {name:[] for name in args.dataset.split('*')}
        all_generated_samples = {name:[] for name in args.dataset.split('*')}
        all_generated_samples_scale = {name:[] for name in args.dataset.split('*')}
        all_names = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                name, samples, samples_scale, c_targets, observed_time, observed_loc = model.evaluate(test_batch, nsample)

                samples = samples.permute(0, 1, 3, 2, 4, 5)  # (B,nsample,L,K,H,H)
                c_targets = c_targets.permute(0, 2, 1, 3, 4)  # (B,L,K,H,H)
                samples_scale = samples_scale.permute(0, 1, 2, 4, 3, 5, 6) # (B,nsample,S,L,K,H,H,H)

                all_target[name].append(c_targets)
                all_observed_time[name].append(observed_time)
                all_observed_loc[name].append(observed_loc)
                all_generated_samples[name].append(samples)
                all_generated_samples_scale[name].append(samples_scale)
                all_names.append(name)
                # ----------------------------generated Metric------------------------------------

                it.set_postfix(
                    ordered_dict={
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            for name in args.dataset.split('*'): 

                if len(all_target[name]) != 0:
                    all_target[name] = torch.cat(all_target[name], dim=0).squeeze(-1).cpu().numpy()
                    all_observed_time[name] = torch.cat(all_observed_time[name], dim=0).cpu().numpy()
                    all_observed_loc[name] = torch.cat(all_observed_loc[name], dim=0).cpu().numpy()
                    all_generated_samples[name] = torch.cat(all_generated_samples[name], dim=0).squeeze(-1).cpu().numpy()
                    all_generated_samples_scale[name] = torch.cat(all_generated_samples_scale[name], dim=0).squeeze(-1).cpu().numpy()

                    scaler = scaler_all[name] # scaler_all['global']
                    all_target[name] = scaler.inverse_transform(all_target[name].reshape(-1,1)).reshape(all_target[name].shape)
                    all_generated_samples[name] = scaler.inverse_transform(all_generated_samples[name].reshape(-1,1)).reshape(all_generated_samples[name].shape)
                    all_generated_samples_scale[name] = scaler.inverse_transform(all_generated_samples_scale[name].reshape(-1,1)).reshape(all_generated_samples_scale[name].shape)

            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:

                pickle.dump(
                    [
                        all_names,
                        all_generated_samples,
                        all_generated_samples_scale,
                        all_target,
                        all_observed_time,
                        all_observed_loc,
                        scaler_all,
                    ],
                    f,  
                )