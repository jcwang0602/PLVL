# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torchvision.transforms import Compose, Normalize, ToTensor

import datasets.transforms as T

from .data_loader import EEVGDataset


def make_transforms(args, image_set, is_onestage=False):
    if is_onestage:
        normalize = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return normalize

    imsize = args.imsize

    if image_set == "train":
        scales = []
        if args.aug_scale:
            for i in range(7):
                scales.append(imsize - 32 * i)
        else:
            scales = [imsize]

        if args.aug_crop:
            crop_prob = 0.5
        else:
            crop_prob = 0.0
        # 随机裁剪
        return T.Compose(
            [
                T.RandomSelect(
                    T.RandomResize(scales),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600], with_long_side=False),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales),
                        ]
                    ),
                    p=crop_prob,
                ),
                T.ColorJitter(0.4, 0.4, 0.4),
                T.GaussianBlur(aug_blur=args.aug_blur),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.NormalizeAndPad(size=imsize, aug_translate=args.aug_translate, model_name=args.model_name),
            ]
        )

    if image_set in ["val", "test", "testA", "testB"]:
        return T.Compose(
            [
                T.RandomResize([imsize]),
                T.ToTensor(),
                T.NormalizeAndPad(size=imsize, model_name=args.model_name),
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build_dataset(split, args):
    return EEVGDataset(
        data_root=args.data_root,
        split_root=args.split_root,
        dataset=args.dataset,
        split=split,
        transform=make_transforms(args, split),
        max_query_len=args.max_query_len,
        is_segment=args.is_res,
        bert_model=args.bert_model,
    )


def build_dataset_vis(split, args):
    return EEVGDataset_Vis(
        data_root=args.data_root,
        split_root=args.split_root,
        dataset=args.dataset,
        split=split,
        transform=make_transforms(args, split),
        max_query_len=args.max_query_len,
        is_segment=args.is_segment,
        bert_model=args.bert_model,
    )
