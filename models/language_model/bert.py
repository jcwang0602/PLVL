# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import torch
from pytorch_pretrained_bert.modeling import BertModel
from torch import nn

from utils.misc import NestedTensor


class BERT(nn.Module):
    def __init__(self, name: str, train_bert: bool, hidden_dim: int, max_len: int, enc_num):
        super().__init__()
        if "bert-base-uncased" in name:
            self.num_channels = 768
        else:
            self.num_channels = 1024
        self.enc_num = enc_num

        self.bert = BertModel.from_pretrained(name)

        if not train_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)

        cur_bert_layer_num = len(self.bert.encoder.layer)
        for ind in range(cur_bert_layer_num, 0, -1):
            if ind > self.enc_num:
                del self.bert.encoder.layer[ind - 1]
            else:
                break

    def forward(self, tensor_list: NestedTensor):
        if self.enc_num > 0:
            all_encoder_layers, _ = self.bert(tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask)
            # use the output of the X-th transformer encoder layers
            xs = all_encoder_layers[self.enc_num - 1]
        else:
            xs = self.bert.embeddings.word_embeddings(tensor_list.tensors)

        mask = tensor_list.mask.to(torch.bool)
        mask = ~mask
        out = NestedTensor(xs, mask)

        return out


def build_bert(args):
    try:
        train_bert = args.lr_bert > 0
    except:
        train_bert = False
    bert = BERT(args.bert_model, train_bert, None, None, args.bert_enc_num)
    return bert
