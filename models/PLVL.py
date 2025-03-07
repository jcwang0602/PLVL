from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .language_model.bert import build_bert
from .visual_model.visual_backbone import build_visual


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class PLVL(nn.Module):
    def __init__(self, args):
        super(PLVL, self).__init__()
        hidden_dim = args.vl_hidden_dim
        # visual backbone
        self.visumodel = build_visual(args)
        # language backbone
        self.textmodel = build_bert(args)
        # 是否定位和分割任务并行训练
        self.is_rec = args.is_rec
        self.is_res = args.is_res
        self.is_eliminate = False

        print("is_rec", self.is_rec)
        print("is_res", self.is_res)

        self.patch_length = 28
        self.imsize = args.imsize

        decoder = dict(
            num_layers=args.vl_enc_layers,
            layer=dict(d_model=hidden_dim, nhead=8, dim_feedforward=args.vl_dim_feedforward, dropout=0.1, activation="relu"),
            is_eliminate=self.is_eliminate,
        )
        print("vl_dim_feedforward", args.vl_dim_feedforward)

        from .decoder import TransformerDecoder

        self.decoder = TransformerDecoder(decoder, patch_length=self.patch_length)

        self.mask_head = MLP(hidden_dim, hidden_dim, 256, 2)
        self.mask_cnn = nn.Conv2d(1, 1, 5, padding=2)

        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

        self.bbox_embed = CenterPredictor(hidden_dim, hidden_dim, 28, 16)

    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]

        # language bert
        text_fea = self.textmodel(text_data)  # [8,40]
        text_src, text_mask = text_fea.decompose()  # text_src [8, 40, 768] text_mask [8,40]

        # visual backbone
        visu_mask, visu_src, vit_inter_features, vis_inter_attns = self.visumodel(img_data, text_src)
        visu_src = self.visu_proj(visu_src)  # (N*B)xC   C 768->768

        assert text_mask is not None
        text_src = self.text_proj(text_src)  # C 768->768

        # permute BxLenxC to LenxBxC
        text_src = text_src.permute(1, 0, 2)
        text_mask = text_mask.flatten(1)

        # 视觉-语言解码器
        tgt, dec_inter_features, dec_inter_attns = self.decoder(tgt=visu_src, memory=text_src, tgt_key_padding_mask=visu_mask, memory_key_padding_mask=text_mask)

        # [782, 8, 768]
        L, B, C = tgt.shape
        visu_src_bbox = tgt.permute(1, 2, 0).reshape(bs, C, 28, 28).contiguous()
        score_map_ctr, score_map_seg, bbox, size_map, offset_map = self.bbox_embed(visu_src_bbox)
        # [782, 8, 768]
        outputs_coord = bbox
        outputs_coord_new = outputs_coord.view(bs, 1, 4)

        out = {
            "pred_boxes": outputs_coord_new,
            "score_map": score_map_ctr,
            "size_map": size_map,
            "offset_map": offset_map,
            "pred_mask": score_map_seg,
            "vit_inter_features": vit_inter_features,
            "dec_inter_features": dec_inter_features,
            "vit_inter_attns": vis_inter_attns,
            "dec_inter_attns": dec_inter_attns,
        }
        return out


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class CenterPredictor(
    nn.Module,
):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(CenterPredictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        # corner predict
        self.conv1_ctr = conv(256, 256, freeze_bn=freeze_bn, kernel_size=3, padding=1)

        self.conv2_ctr = conv(256, 128, freeze_bn=freeze_bn, kernel_size=3, padding=1)
        self.conv3_ctr = nn.Conv2d(128, 1, kernel_size=1)

        # segment mask
        self.conv2_seg = conv(256, 256, freeze_bn=freeze_bn, kernel_size=1, padding=0)
        self.conv3_seg = nn.Conv2d(1, 1, 5, padding=2)

        # size regress
        self.conv1_offset = conv(256, 256, freeze_bn=freeze_bn, kernel_size=3, padding=1)

        self.conv2_offset = conv(256, 128, freeze_bn=freeze_bn, kernel_size=3, padding=1)
        self.conv3_offset = nn.Conv2d(128, 2, kernel_size=1)

        # size regress
        self.conv2_size = conv(256, 128, freeze_bn=freeze_bn, kernel_size=3, padding=1)
        self.conv3_size = nn.Conv2d(128, 2, kernel_size=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """Forward pass with input x."""
        score_map_ctr, score_map_seg, size_map, offset_map = self.get_score_map(x)

        bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)

        return score_map_ctr, score_map_seg, bbox, size_map, offset_map

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz, (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz, size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)
        return size * self.feat_sz, offset

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x[:, :256, ...])
        x_ctr2 = self.conv2_ctr(x_ctr1)
        score_map_ctr = self.conv3_ctr(x_ctr2)

        # seg mask
        x_seg2 = self.conv2_seg(x_ctr1)
        x_seg2 = x_seg2.permute(0, 2, 3, 1).reshape(-1, 28, 28, 16, 16).permute(0, 1, 3, 2, 4).reshape(-1, 1, 448, 448).contiguous()
        score_map_seg = self.conv3_seg(x_seg2).sigmoid()

        # offset branch
        x_offset1 = self.conv1_offset(x[:, 256:, ...])
        x_offset2 = self.conv2_offset(x_offset1)
        score_map_offset = self.conv3_offset(x_offset2)

        # size branch
        x_size2 = self.conv2_size(x_offset1)
        score_map_size = self.conv3_size(x_size2)
        return _sigmoid(score_map_ctr), score_map_seg, _sigmoid(score_map_size), score_map_offset


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True), FrozenBatchNorm2d(out_planes), nn.ReLU(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))


class FrozenBatchNorm2d(torch.nn.Module):
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
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
