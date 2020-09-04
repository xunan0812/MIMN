# -*- coding: utf-8 -*-
# file: ram.py
# author: xunna <xunan2015@ia.ac.cn>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import torch
import torch.nn as nn


class RAM(nn.Module):
    def locationed_memory(self, memory, memory_len):
        # here we just simply calculate the location vector in Model2's manner
        for i in range(memory.size(0)):
            for idx in range(memory_len[i]):
                memory[i][idx] *= (1-float(idx)/int(memory_len[i]))
        return memory

    def __init__(self, embedding_matrix, opt):
        super(RAM, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        self.bi_lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.bi_lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = Attention(opt.hidden_dim*2, score_function='mlp')
        self.gru_cell = nn.GRUCell(opt.hidden_dim*2, opt.hidden_dim*2)
        self.dense = nn.Linear(opt.hidden_dim*2, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        memory_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)

        memory = self.embed(text_raw_indices)
        memory, (_, _) = self.bi_lstm_context(memory, memory_len)
        # memory = self.locationed_memory(memory, memory_len)
        aspect = self.embed(aspect_indices)
        aspect, (_, _) = self.bi_lstm_aspect(aspect, aspect_len)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))

        et = aspect
        for _ in range(self.opt.hops):
            it_al = self.attention(memory, et).squeeze(dim=1)
            et = self.gru_cell(it_al, et)
        out = self.dense(et)
        return out

class RAM2m(nn.Module):
    def locationed_memory(self, memory, memory_len):
        # here we just simply calculate the location vector in Model2's manner
        for i in range(memory.size(0)):
            for idx in range(memory_len[i]):
                memory[i][idx] *= (1-float(idx)/int(memory_len[i]))
        return memory

    def __init__(self, embedding_matrix, opt):
        super(RAM2m, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        self.bi_lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.bi_lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.bi_lstm_img = DynamicLSTM(opt.embed_dim_img, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.attention_text = Attention(opt.hidden_dim*2, score_function='mlp')
        self.attention_img = Attention(opt.hidden_dim * 2, score_function='mlp')

        self.gru_cell_text = nn.GRUCell(opt.hidden_dim*2, opt.hidden_dim*2)
        self.gru_cell_img = nn.GRUCell(opt.hidden_dim*2, opt.hidden_dim*2)

        self.bn = nn.BatchNorm1d(opt.hidden_dim*2, affine=False)
        self.fc = nn.Linear(opt.hidden_dim * 4, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, aspect_indices, imgs, num_imgs = inputs[0], inputs[1], inputs[2], inputs[3]
        text_memory_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        imgs_memory_len = torch.tensor(num_imgs).to(self.opt.device)
        nonzeros_imgs = torch.tensor(num_imgs, dtype=torch.float).to(self.opt.device)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)

        text_raw = self.embed(text_raw_indices)
        aspect = self.embed(aspect_indices)

        text_memory, (_, _) = self.bi_lstm_context(text_raw, text_memory_len)
        # memory = self.locationed_memory(memory, memory_len)

        aspect, (_, _) = self.bi_lstm_aspect(aspect, aspect_len)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))
        et_text = aspect

        img_memory, (_, _) = self.bi_lstm_img(imgs, imgs_memory_len)
        img_memory = torch.sum(img_memory, dim=1)
        img_memory = torch.div(img_memory, nonzeros_imgs.view(nonzeros_imgs.size(0), 1))
        et_img = img_memory




        for _ in range(self.opt.hops):
            it_al_text = self.attention_text(text_memory, et_img).squeeze(dim=1)
            et_text = self.gru_cell_text(it_al_text, et_text)
        et = torch.cat((et_text, et_img), dim=-1)
        out = self.fc(et)
        return out