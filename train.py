import argparse
import datetime
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.amp import GradScaler
from torch.utils.data import DataLoader, DistributedSampler

import utils.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, validate
from models import build_model
from utils.loss_utils import FocalLoss


def get_args_parser():
    parser = argparse.ArgumentParser("Set PLVL", add_help=False)
    parser.add_argument("--lr", default=2.5e-5, type=float)
    parser.add_argument("--lr_bert", default=5e-6, type=float)
    parser.add_argument("--lr_visual", default=1e-5, type=float)
    parser.add_argument("--loss_alpha", default=0.1, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--lr_power", default=0.9, type=float, help="lr poly power")
    parser.add_argument("--clip_max_norm", default=0.0, type=float, help="gradient clipping max norm")
    parser.add_argument("--eval", dest="eval", default=False, action="store_true", help="if evaluation only")
    parser.add_argument("--optimizer", default="adamw", type=str)
    parser.add_argument("--lr_scheduler", default="poly", type=str)
    parser.add_argument("--lr_drop", default=120, type=int)
    # Augmentation options
    parser.add_argument("--aug_blur", action="store_true", help="If true, use gaussian blur augmentation")
    parser.add_argument("--aug_crop", action="store_true", help="If true, use random crop augmentation")
    parser.add_argument("--aug_scale", action="store_true", help="If true, use multi-scale augmentation")
    parser.add_argument("--aug_translate", action="store_true", help="If true, use random translate augmentation")
    # Model parameters
    parser.add_argument("--is_res", action="store_true", help="if use segmentation")
    parser.add_argument("--is_rec", action="store_true", help="if use location")
    parser.add_argument("--model_name", type=str, default="PLVL")
    parser.add_argument("--backbone", type=str, default="ViTDet_Dec")
    parser.add_argument("--position_embedding", default="sine", type=str, choices=("sine", "learned"), help="Type of positional embedding to use on top of the image features")
    parser.add_argument("--imsize", default=448, type=int, help="image size")
    parser.add_argument("--patch_size", default=16, type=int, help="image size")
    parser.add_argument("--ca_block_indexes", default=[], type=int, nargs="*")
    parser.add_argument("--bert_model", default="checkpoints/bert-base-uncased", type=str, help="bert model")
    parser.add_argument("--bert_enc_num", default=12, type=int)
    parser.add_argument("--vl_dropout", default=0.1, type=float, help="Dropout applied in the vision-language transformer")
    parser.add_argument("--vl_nheads", default=8, type=int, help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument("--vl_hidden_dim", default=512, type=int, help="Size of the embeddings (dimension of the vision-language transformer)")
    parser.add_argument("--vl_dim_feedforward", default=1024, type=int, help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument("--vl_enc_layers", default=3, type=int, help="Number of encoders in the vision-language transformer")
    parser.add_argument("--pretrain", default=None, type=str, help="pretrained checkpoint")
    # Dataset parameters
    parser.add_argument("--data_root", type=str, default="./image_data")
    parser.add_argument("--split_root", type=str, default="./mask_data")
    parser.add_argument("--dataset", default="gref_umd", type=str, help="referit/unc/unc+/gref/gref_umd")
    parser.add_argument("--max_query_len", default=40, type=int, help="maximum time steps (lang length) per batch")

    # train parameters
    parser.add_argument("--output_dir", default="./outputs/test", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cpu", help="device to use for training / testing")
    parser.add_argument("--seed", default=13, type=int)
    parser.add_argument("--resume", default=None, help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_workers", default=10, type=int)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # build tensorboard
    writer = SummaryWriter(args.output_dir)
    # build model
    model = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"all_n_params:{n_parameters}")

    visu_param = [p for n, p in model_without_ddp.named_parameters() if (("visumodel" in n) and p.requires_grad)]
    text_param = [p for n, p in model_without_ddp.named_parameters() if (("textmodel" in n) and p.requires_grad)]
    rest_param = [p for n, p in model_without_ddp.named_parameters() if (("visumodel" not in n) and ("textmodel" not in n) and p.requires_grad)]
    print(f"visu_para_train_num:{len(visu_param)}")
    print(f"text_para_train_num:{len(text_param)}")
    print(f"rest_para_train_num:{len(rest_param)}")

    param_list = [
        {"params": rest_param},
        {"params": visu_param, "lr": args.lr_visual},
        {"params": text_param, "lr": args.lr_bert},
    ]

    if args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError("Lr scheduler type not supportted ")

    if args.lr_scheduler == "poly":

        def lr_func(epoch):
            return (1 - epoch / args.epochs) ** args.lr_power

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == "halfdecay":
        lr_func = lambda epoch: 0.5 ** (epoch // (args.epochs // 10))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == "cosine":

        def lr_func(epoch):
            if epoch < 3:
                return epoch / 10
            else:
                return 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    else:
        raise ValueError("Lr scheduler type not supportted ")

    # build dataset
    dataset_train = build_dataset("train", args)
    dataset_val = build_dataset("val", args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    best_accu = 0
    best_mask_iou = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.eval and "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            best_accu = checkpoint["val_accu"]
            best_mask_iou = checkpoint["val_mask_miou"]

    elif args.pretrain is not None:
        checkpoint = torch.load(args.pretrain, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        print("Loading pretrain model from {}".format(args.pretrain))

    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(str(args) + "\n")

    print("Start training")
    scaler = GradScaler()
    start_time = time.time()
    focal_loss = FocalLoss()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(args, model, scaler, data_loader_train, optimizer, device, epoch, args.clip_max_norm, args.loss_alpha, epoch == args.start_epoch, focal_loss=focal_loss)
        lr_scheduler.step()
        val_stats = validate(args, model, data_loader_val, device)
        log_stats = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"validation_{k}": v for k, v in val_stats.items()},
            "best_accu": best_accu,
            "best_mask_iou": best_mask_iou,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.output_dir:
            if "accu" in val_stats and val_stats["accu"] > best_accu:
                best_accu = val_stats["accu"]
                utils.save_on_master({"model": model_without_ddp.state_dict(), "epoch": epoch}, output_dir / "best_checkpoint.pth")
            if "mask_miou" in val_stats and val_stats["mask_miou"] > best_mask_iou:
                best_mask_iou = val_stats["mask_miou"]
                utils.save_on_master({"model": model_without_ddp.state_dict(), "epoch": epoch}, output_dir / "best_mask_checkpoint.pth")

            utils.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "val_accu": val_stats["accu"] if "accu" in val_stats else 0,
                    "val_mask_miou": val_stats["mask_miou"] if "mask_miou" in val_stats else 0,
                },
                output_dir / "checkpoint.pth",
            )
        if utils.is_main_process:
            # train info
            writer.add_scalar("train/total_loss", train_stats["loss"], epoch)
            if args.is_rec:
                writer.add_scalar("train/giou_loss", train_stats["giou_loss"], epoch)
                writer.add_scalar("train/l1_loss", train_stats["l1_loss"], epoch)
                writer.add_scalar("train/focal_loss_box", train_stats["focal_loss_box"], epoch)
            if args.is_res:
                writer.add_scalar("train/focal_loss_seg", train_stats["focal_loss_seg"], epoch)
                writer.add_scalar("train/dice_loss", train_stats["dice_loss"], epoch)
            # val info
            if args.is_rec:
                writer.add_scalar("val/accu", val_stats["accu"], epoch)
            if args.is_res:
                writer.add_scalar("val/mask_miou", val_stats["mask_miou"], epoch)
            # learning rate
            writer.add_scalar("lr/lr_other", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("lr/lr_visu", optimizer.param_groups[1]["lr"], epoch)
            writer.add_scalar("lr/lr_text", optimizer.param_groups[2]["lr"], epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PLVL training script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
