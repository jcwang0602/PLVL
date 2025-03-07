from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import _get_clones

from .decoder_layer.decoder_layer import TransformerDecoderLayer


def with_pos_embed(tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos


class TransformerDecoderLayerWithPositionEmbedding(TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super(TransformerDecoderLayerWithPositionEmbedding, self).__init__(*args, **kwargs)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        need_weights: bool = False,
    ):
        # 视觉自注意力
        tgt2 = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=need_weights)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # 视觉语言交叉注意力
        # tgt [784-783-782, 8, 768]  memory [40, 8, 768]
        tgt2, attn = self.multihead_attn(
            query=with_pos_embed(tgt, query_pos),
            key=with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn


class TransformerDecoderWithPositionEmbedding(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm, is_eliminate=False):
        super(TransformerDecoderWithPositionEmbedding, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.is_eliminate = is_eliminate
        if self.is_eliminate:
            self.init_idx()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        self.tgt_key_padding_mask = tgt_key_padding_mask  # [8, 785]
        self.query_pos = query_pos  # [785, 8, 768]
        # 3 层解码器
        features = []
        attns = []
        for layer_num, layer in enumerate(self.layers):
            output, attn = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=self.tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=self.query_pos,
                need_weights=False,
            )
            features.append(output)
            attns.append(attn)
        output = self.norm(output)
        return output, features, attns

    def pad_patch(self, attn_weight, cur_idx, min_num):
        cur_idx = cur_idx[:, 1:] - 1
        new_weight = min_num.clone().unsqueeze(-1).repeat(1, PATCH_LEN**2)
        new_weight.scatter_(1, cur_idx, attn_weight)
        return new_weight


class TransformerDecoder(nn.Module):
    def __init__(self, decoder, patch_length):
        super(TransformerDecoder, self).__init__()
        self.d_model = decoder["layer"]["d_model"]
        self.decoder = TransformerDecoderWithPositionEmbedding(TransformerDecoderLayerWithPositionEmbedding(**decoder.pop("layer")), **decoder, norm=nn.LayerNorm(self.d_model))
        global PATCH_LEN
        PATCH_LEN = patch_length
        self._reset_parameters()

    def _reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):

        tgt, features, attns = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos
        )

        return tgt, features, attns
