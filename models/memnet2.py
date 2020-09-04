# -*- coding: utf-8 -*-
# file: memnet.py
# author: xunna <xunan2015@ia.ac.cn>
# Copyright (C) 2018. All Rights Reserved.

from layers.attention import Attention
from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn

from layers.squeeze_embedding import SqueezeEmbedding


class MemNet2(nn.Module):
    
    def locationed_memory(self, memory, memory_len, left_len, aspect_len):
        # here we just simply calculate the location vector in Model2's manner
        '''
        Updated to calculate location as the absolute diference between context word and aspect
        '''
        for i in range(memory.size(0)):
            for idx in range(memory_len[i]):
                aspect_start = left_len[i] - aspect_len[i]
                if idx < aspect_start: l = aspect_start.item() - idx                   # l = absolute distance to the aspect
                else: l = idx +1 - aspect_start.item()
                memory[i][idx] *= (1-float(l)/int(memory_len[i]))
               
        return memory

    def __init__(self, embedding_matrix, opt):
        super(MemNet2, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.bi_lstm_img = DynamicLSTM(opt.embed_dim_img, opt.embed_dim, num_layers=1, batch_first=True, bidirectional=False)

        self.attention = Attention(opt.embed_dim, score_function='mlp')
        self.x_linear = nn.Linear(opt.embed_dim, opt.embed_dim)
        self.dense = nn.Linear(opt.embed_dim*2, opt.polarities_dim)

    def forward(self, inputs):
        # text_raw_without_aspect_indices, aspect_indices, left_with_aspect_indices = inputs[0], inputs[1], inputs[2]

        text_raw_without_aspect_indices, aspect_indices, imgs, num_imgs = inputs[0], inputs[1], inputs[2], inputs[3]

        memory_len = torch.sum(text_raw_without_aspect_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        imgs_memory_len = torch.tensor(num_imgs).to(self.opt.device)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)

        text_memory = self.embed(text_raw_without_aspect_indices)
        text_memory = self.squeeze_embedding(text_memory, memory_len)

        img_memory, (_, _) = self.bi_lstm_img(imgs, imgs_memory_len)

        # memory = self.locationed_memory(memory, memory_len, left_len, aspect_len)
        aspect = self.embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))
        x = aspect.unsqueeze(dim=1)

        et_text = x
        et_img = x

        for _ in range(self.opt.hops):
            et_text = self.x_linear(et_text)
            out_at_text = self.attention(text_memory, et_img)
            et_text = out_at_text + et_text

            et_img = self.x_linear(et_img)
            out_at_img = self.attention(img_memory, et_text)
            et_img = out_at_img + et_img

        et_text = et_text.view(et_text.size(0), -1)
        et_img = et_img.view(et_img.size(0), -1)
        et = torch.cat((et_text, et_img), dim=-1)
        out = self.dense(et)
        # out = self.dense(x)
        return out

