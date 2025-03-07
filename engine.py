# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py 混合精度
"""
import gc
import math
import sys
from typing import Iterable

import torch
import torch.distributed as dist
from torch.amp import autocast
from tqdm import tqdm

import utils.eval_utils as eval_utils
import utils.loss_utils as loss_utils
import utils.misc as utils


def train_one_epoch(
    args,
    model: torch.nn.Module,
    scaler,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    alpha: float = 1.0,
    is_collect=False,
    focal_loss=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = "Epoch:[{}]".format(epoch)
    print_freq = 100

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        if args.is_res:
            img_data, text_data, target, gt_mask = batch
        else:
            img_data, text_data, target = batch
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)  #  cx,cy,w,h
        if args.is_res:
            gt_mask = gt_mask.to(device)
        else:
            gt_mask = None

        with autocast(device_type="cuda"):
            # model forward
            output = model(img_data, text_data)
            # target: x1y1x2y2
            loss_dict = loss_utils.trans_vg_convhead_loss(args, output, target, gt_mask, focal_loss, args.patch_size)
            losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v for k, v in loss_dict_reduced.items()}
        losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
        loss_value = losses_reduced_unscaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Eval:"

    for batch in metric_logger.log_every(data_loader, 100, header):
        if args.is_res:
            img_data, text_data, target, gt_mask = batch
        else:
            img_data, text_data, target = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        if args.is_res:
            gt_mask = gt_mask.to(device)
        else:
            gt_mask = None

        output = model(img_data, text_data)

        if args.is_rec:
            miou, accu = eval_utils.trans_vg_eval_val(output["pred_boxes"].squeeze(1), target)
            metric_logger.update_v2("accu", accu, batch_size)
            metric_logger.update_v2("miou", torch.mean(miou), batch_size)

        if args.is_res:
            mask_miou = eval_utils.trans_vg_eval_miou(output["pred_mask"], gt_mask)
            metric_logger.update_v2("mask_miou", mask_miou, batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    print("Averaged stats:", metric_logger)
    return stats


@torch.no_grad()
def evaluate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []
    tot_num = 0
    tot_iou = 0
    print_profile = True
    for _, batch in enumerate(tqdm(data_loader)):
        if args.is_res:
            img_data, text_data, target, gt_mask = batch
        else:
            img_data, text_data, target = batch

        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        output = model(img_data, text_data)

        tot_num += output["pred_mask"].shape[0]

        if args.is_res:
            mask_miou = eval_utils.trans_vg_eval_miou(output["pred_mask"], gt_mask)
            tot_iou += mask_miou * output["pred_mask"].shape[0]

        if args.is_rec:
            pred_box_list.append(output["pred_boxes"].squeeze(1).cpu())
            gt_box_list.append(target.cpu())

        if print_profile:
            from thop import profile

            flops, params = profile(model, inputs=(img_data, text_data))
            print_profile = False
            print(f"GFLOPs: {flops/1e9:.3f} G")
            print(f"PARAMs: {params/1e6:.3f} M")

    # print(sum(tot_time) / len(tot_time))
    if args.is_rec:
        pred_boxes = torch.cat(pred_box_list, dim=0)
        gt_boxes = torch.cat(gt_box_list, dim=0)
        total_num = gt_boxes.shape[0]
        accu_num = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)
    else:
        total_num = 1
        accu_num = 0

    result_tensor = torch.tensor([accu_num, total_num, tot_iou, tot_num]).to(device)

    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)

    if args.is_rec:
        accuracy = float(result_tensor[0]) / float(result_tensor[1])
    else:
        accuracy = 0

    if args.is_res:
        miou = float(result_tensor[2]) / float(result_tensor[3])
    else:
        miou = 0
    return accuracy, miou


def clean_cache():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
