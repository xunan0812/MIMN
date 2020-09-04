# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import torch
import torch.nn as nn


class AELSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(AELSTM, self).__init__()
        self.opt = opt
        self.n_head = 1
        self.embed_dim = opt.embed_dim
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)

        self.lstm = DynamicLSTM(opt.embed_dim*2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention = Attention(opt.hidden_dim, score_function='mlp')
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        x = self.embed(text_raw_indices)
        aspect = self.embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))
        asp_squ = aspect.unsqueeze(dim=1)

        asp_re = asp_squ.repeat(1, x.size()[1], 1)
        asp_x = torch.cat((x, asp_re), dim=-1)
        text_memory, (_, _) = self.lstm(asp_x, x_len)
        out_at = self.attention(text_memory, asp_squ).squeeze(dim=1)
        out_at = out_at.view(out_at.size(0), -1)
        out = self.dense(out_at)
        return out
