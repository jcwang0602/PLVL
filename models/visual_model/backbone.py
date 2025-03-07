# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from utils.misc import NestedTensor

from .position_encoding import build_position_encoding
from .ViTDet_Dec import build_ViTDet_Dec


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, name: str, backbone: nn.Module, num_channels: int):
        super().__init__()
        self.body = backbone
        self.num_channels = num_channels
        self.size = (28, 28)

    def forward(self, img_data: NestedTensor, txt_tokens):
        xs = self.body(img_data.tensors, txt_tokens)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = img_data.mask
            assert m is not None  # mask [8,448,448]->[8,28,28]
            mask = F.interpolate(m[None].float(), size=self.size).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, args):
        name = args.backbone

        if name == "ViTDet_Dec":
            backbone = build_ViTDet_Dec(args.ca_block_indexes)
            import pickle as pkl

            with open("./checkpoints/model_final_435fa9.pkl", "rb") as f:
                info_dict = pkl.load(f)

            new_dict = {}
            for k, v in info_dict["model"].items():
                if "backbone.net." in k:
                    k = k.replace("backbone.net.", "")
                new_dict[k] = torch.from_numpy(v)
            backbone.load_state_dict(new_dict, strict=False)

        num_channels = 768

        super().__init__(name, backbone, num_channels)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, img_data: NestedTensor, txt_tokens):
        xs = self[0](img_data, txt_tokens)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():

            out.append(x)
            # position encoding
            if name == "0":
                pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone(args)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
