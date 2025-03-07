import torch
from torch import nn

from utils.misc import NestedTensor, nested_tensor_from_tensor_list

from .backbone import build_backbone


class VisualBackbone(nn.Module):

    def __init__(self, backbone, train_backbone, backbone_name):
        super().__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name

        hidden_dim = backbone.num_channels

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.num_channels = hidden_dim

    def forward(self, img_data: NestedTensor, txt_tokens=None):
        if isinstance(img_data, (list, torch.Tensor)):
            img_data = nested_tensor_from_tensor_list(img_data)
        features, pos = self.backbone(img_data, txt_tokens)

        src, mask = features[0].decompose()
        assert mask is not None
        # src [8, 784, 768]   mask [8, 28, 28]
        try:
            out = [mask.flatten(1), src.permute(1, 0, 2), features[1].tensors, features[2].tensors]
        except:
            out = [mask.flatten(1), src.permute(1, 0, 2), None]
        return out


def build_visual(args):
    backbone = build_backbone(args)
    try:
        train_backbone = args.lr_visual > 0
    except:
        train_backbone = False

    model = VisualBackbone(
        backbone,
        train_backbone=train_backbone,
        backbone_name=args.backbone,
    )
    return model
