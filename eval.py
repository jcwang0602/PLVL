import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import utils.misc as utils
from datasets import build_dataset
from engine import evaluate
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=2.5e-5, type=float)
    parser.add_argument("--lr_bert", default=5e-6, type=float)
    parser.add_argument("--lr_visual", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--lr_power", default=0.9, type=float, help="lr poly power")
    parser.add_argument("--clip_max_norm", default=0.0, type=float, help="gradient clipping max norm")
    parser.add_argument("--eval", dest="eval", default=False, action="store_true", help="if evaluation only")
    parser.add_argument("--optimizer", default="rmsprop", type=str)
    parser.add_argument("--lr_scheduler", default="poly", type=str)
    parser.add_argument("--lr_drop", default=80, type=int)
    # Model parameters
    parser.add_argument("--is_res", action="store_true", help="if use segmentation")
    parser.add_argument("--is_rec", action="store_true", help="if use location")
    parser.add_argument("--model_name", type=str, default="PLVL", help="EEVG_ConvH, EEVG_ConvH_NVL")
    parser.add_argument("--backbone", type=str, default="ViTDet_Dec", help="ViTDet, ViTDet_Dec, ViTDet_Dec_Adapter_NR, ViTDet_Dec_CTM")
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
    parser.add_argument("--data_root", type=str, default="/share/wangjingchao/vg_data/image_data")
    parser.add_argument("--split_root", type=str, default="/share/wangjingchao/vg_data/mask_data")
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

    # evalutaion options
    parser.add_argument("--eval_set", default="val", type=str)
    parser.add_argument("--eval_model", default="", type=str)

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # build dataset
    dataset_test = build_dataset(args.eval_set, args)

    if args.distributed:
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = DataLoader(
        dataset_test,
        args.batch_size,
        sampler=sampler_test,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    checkpoint = torch.load(args.eval_model, map_location="cpu")
    try:
        model_without_ddp.load_state_dict(checkpoint["model"])
    except:
        pass

    # output log
    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "eval_log.txt").open("a") as f:
            f.write(str(args) + "\n")

    start_time = time.time()

    if args.is_res:
        # perform evaluation
        accuracy, mIoU = evaluate(args, model, data_loader_test, device)
    else:
        accuracy = evaluate(args, model, data_loader_test, device)

    if utils.is_main_process():
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

        log_stats = {"test_model:": args.eval_model}
        if args.is_rec:
            log_stats["%s_set_accuracy" % args.eval_set] = accuracy
        if args.is_res:
            log_stats["%s_set_mIoU" % args.eval_set] = mIoU
        print(log_stats)
        if args.output_dir and utils.is_main_process():
            with (output_dir / "eval_log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("EEVG evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
