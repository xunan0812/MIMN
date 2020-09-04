# -*- coding: utf-8 -*-
# file: ian.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import torch
import torch.nn as nn


class IAN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(IAN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention_aspect = Attention(opt.hidden_dim, score_function='bi_linear')
        self.attention_context = Attention(opt.hidden_dim, score_function='bi_linear')
        self.dense = nn.Linear(opt.hidden_dim*2, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        text_raw_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)

        context = self.embed(text_raw_indices)
        aspect = self.embed(aspect_indices)
        context, (_, _) = self.lstm_context(context, text_raw_len)
        aspect, (_, _) = self.lstm_aspect(aspect, aspect_len)

        aspect_len = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        aspect_pool = torch.sum(aspect, dim=1)
        aspect_pool = torch.div(aspect_pool, aspect_len.view(aspect_len.size(0), 1))

        text_raw_len = torch.tensor(text_raw_len, dtype=torch.float).to(self.opt.device)
        context_pool = torch.sum(context, dim=1)
        context_pool = torch.div(context_pool, text_raw_len.view(text_raw_len.size(0), 1))

        aspect_final = self.attention_aspect(aspect, context_pool).squeeze(dim=1)
        context_final = self.attention_context(context, aspect_pool).squeeze(dim=1)

        x = torch.cat((aspect_final, context_final), dim=-1)
        out = self.dense(x)
        return out

class IAN2m(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(IAN2m, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.bi_lstm_img = DynamicLSTM(opt.embed_dim_img, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.attention_aspect = Attention(opt.hidden_dim, score_function='bi_linear')
        self.attention_context = Attention(opt.hidden_dim, score_function='bi_linear')
        self.fc = nn.Linear(opt.hidden_dim * 4, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, aspect_indices, imgs, num_imgs = inputs[0], inputs[1], inputs[2], inputs[3]

        text_raw_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        imgs_memory_len = torch.tensor(num_imgs).to(self.opt.device)
        nonzeros_imgs = torch.tensor(num_imgs, dtype=torch.float).to(self.opt.device)


        context = self.embed(text_raw_indices)
        aspect = self.embed(aspect_indices)
        context, (_, _) = self.lstm_context(context, text_raw_len)
        aspect, (_, _) = self.lstm_aspect(aspect, aspect_len)

        aspect_len = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        aspect_pool = torch.sum(aspect, dim=1)
        aspect_pool = torch.div(aspect_pool, aspect_len.view(aspect_len.size(0), 1))

        text_raw_len = torch.tensor(text_raw_len, dtype=torch.float).to(self.opt.device)
        context_pool = torch.sum(context, dim=1)
        context_pool = torch.div(context_pool, text_raw_len.view(text_raw_len.size(0), 1))

        aspect_final = self.attention_aspect(aspect, context_pool).squeeze(dim=1)
        context_final = self.attention_context(context, aspect_pool).squeeze(dim=1)

        img_memory, (_, _) = self.bi_lstm_img(imgs, imgs_memory_len)
        img_memory = torch.sum(img_memory, dim=1)
        img_memory = torch.div(img_memory, nonzeros_imgs.view(nonzeros_imgs.size(0), 1))
        et_img = img_memory

        x = torch.cat((aspect_final, context_final, et_img), dim=-1)
        out = self.fc(x)
        return out
