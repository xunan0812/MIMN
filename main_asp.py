# -*- coding: utf-8 -*-
# file: main_asp.py
# author: xunna <xunan2015@ia.ac.cn>
# Copyright (C) 2018. All Rights Reserved.
# Requirement: torch 0.4.0

from data_utils import ABSADatesetReader, ZOLDatesetReader
import time
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from visdom import Visdom
from sklearn.metrics import accuracy_score
from sklearn import metrics
from torchvision.models import alexnet, resnet18, resnet50, inception_v3
from models.lstm import LSTM
from models.ian import IAN, IAN2m
from models.memnet import MemNet
from models.ram import RAM, RAM2m
from models.mimn import MIMN
from models.td_lstm import TD_LSTM
from models.cabasc import Cabasc
from models.memnet2 import MemNet2
from models.ae_lstm import AELSTM

np.random.seed(1337)  # for reproducibility

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        print('> training arguments:')
        for arg in vars(opt):
            print('>>> {0}: {1}'.format(arg, getattr(opt, arg)))

        if opt.dataset in ['restaurant', 'laptop']:
            self.my_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim, max_seq_len=opt.max_seq_len)
            self.train_data_loader = DataLoader(dataset=self.my_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
            self.dev_data_loader = DataLoader(dataset=self.my_dataset.test_data, batch_size=len(self.my_dataset.test_data), shuffle=False)
            self.test_data_loader = DataLoader(dataset=self.my_dataset.test_data, batch_size=len(self.my_dataset.test_data), shuffle=False)

        elif opt.dataset in ['zol_cellphone']:
            self.my_dataset = ZOLDatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim, max_seq_len=opt.max_seq_len, cnn_model_name=opt.cnn_model_name)
            self.train_data_loader = DataLoader(dataset=self.my_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
            self.dev_data_loader = DataLoader(dataset=self.my_dataset.dev_data, batch_size=len(self.my_dataset.dev_data), shuffle=False)
            self.test_data_loader = DataLoader(dataset=self.my_dataset.test_data, batch_size=len(self.my_dataset.test_data), shuffle=False)

        self.idx2word = self.my_dataset.idx2word
        self.writer = SummaryWriter(log_dir=opt.logdir)
        self.model = opt.model_class(self.my_dataset.embedding_matrix, opt).to(opt.device)
        self.reset_parameters()

    def reset_parameters(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
                if len(p.shape) > 1:
                    self.opt.initializer(p)
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))

    def get_metrics(self, truth, pred):
        assert len(truth) == len(pred)
        y_true = truth
        y_pred = pred
        acc = accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='weighted')
        return acc * 100, f1 * 100

    def get_accuracy(self, truth, pred):
        assert len(truth) == len(pred)
        y_true = truth
        y_pred = pred
        acc = accuracy_score(y_true, y_pred)
        return acc * 100

    def findword(self, idxs):
        text = ""
        idxs = idxs.cpu().numpy()
        for id in idxs:
            if id in self.idx2word:
                text += self.idx2word[id]+" "
            else:
                text += str(id) + " "
        return text

    def run(self):
        # Loss and Optimizer
        dtw = time.strftime("%Y-%m-%d-%H-%M", time.localtime(int(time.time())))
        viz = Visdom(env='xunan')
        best_dev_acc = 0.0
        best_dev_f1 = 0.0
        no_up = 0
        loss_line = viz.line(
            X=np.array([0]),
            Y=np.array([2]),
            opts=dict(title=dtw+'_loss_'+opt.model_name))

        acc_line = viz.line(
            X=np.array([0]),
            Y=np.column_stack((np.array([0]), np.array([0]))),
            opts=dict(title=dtw+'_acc_'+opt.model_name))

        loss_function = nn.CrossEntropyLoss()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(params, lr=self.opt.learning_rate)

        for epoch in range(self. opt.num_epoch):
            start = time.time()
            # print('>' * 100)
            tra_loss, tra_acc, tra_f1 = self.train_epoch(loss_function, optimizer, epoch)
            # print('now best dev f1:', best_dev_f1)
            dev_loss, dev_acc, dev_f1 = self.evaluate(loss_function, 'dev')

            end = time.time()
            epoch_dt = end - start

            print('epoch: %d done, %d s! Train avg_loss:%g , acc:%g, f1:%g, Dev loss:%g acc:%g f1:%g' % (epoch, epoch_dt, tra_loss, tra_acc, tra_f1, dev_loss, dev_acc, dev_f1))

            # visdom
            viz.line(X=np.array([epoch]),
                     Y=np.array([tra_loss]),
                     win=loss_line,
                     opts=dict(legend=["loss"]),
                     update='append')
            viz.line(X=np.array([epoch]),
                     Y=np.column_stack((np.array([tra_acc]), np.array([dev_acc]))),
                     win=acc_line,
                     opts=dict(legend=["train_acc", "dev_acc"]),
                     update='append')

            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                os.system('rm ./checkpoint/'+self.opt.dataset+'_'+self.opt.model_name+'_best_acc_*'+dtw+'.models')
                test_loss, test_acc, test_f1 = self.evaluate(loss_function, 'test')
                print('New Best Test acc， f1:', test_acc, test_f1)
                torch.save(self.model.state_dict(), './checkpoint/'+self.opt.dataset+'_'+self.opt.model_name+'_best_acc_%.4g_f1_%.4g' % (test_acc, test_f1) +'_time_'+dtw+ '.models')
                no_up = 0
            else:
                no_up += 1
                if no_up >= self.opt.early_stop:
                    exit()




    def test1(self):
        # Loss and Optimizer
        loss_function = nn.CrossEntropyLoss()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        # switch models to evaluation mode
        self.model.eval()
        print(self.model)
        avg_loss = 0.0
        truth_res = []
        pred_res = []

        with torch.no_grad():
            for batch, sample_batched in enumerate(self.dev_data_loader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                truth_res += list(targets.data)
                outputs = self.model(inputs)
                pred_label = outputs.data.max(1)[1].cpu().numpy()
                pred_res += [x for x in pred_label]
                loss = loss_function(outputs, targets)
                avg_loss += loss.item()
                break

            avg_loss /= len(self.dev_data_loader)
            acc = self.get_accuracy(truth_res, pred_res)
            return avg_loss, acc


    def test2(self, modelname):
        # Loss and Optimizer
        loss_function = nn.CrossEntropyLoss()
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        self.model.load_state_dict(torch.load('./checkpoint/'+modelname))

        # switch models to evaluation mode
        self.model.eval()
        avg_loss = 0.0
        truth_res = []
        pred_res = []
        text_data = []
        img_data = []
        aspect_data = []
        score_texts = []
        score_imgs = []

        with torch.no_grad():
            for batch, sample_batched in enumerate(self.dev_data_loader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                truth_res += list(targets.data)
                outputs, score_text, score_img = self.model(inputs)
                pred_label = outputs.data.max(1)[1].cpu().numpy()
                pred_res += [x for x in pred_label]

                text_data += list(inputs[0].data)
                aspect_data += list(inputs[1].data)
                img_data += list(inputs[2].data)
                score_texts += list(score_text.data)
                score_imgs += list(score_img.data)

                break


            for i in range(len(truth_res)):
                print('sample======',i )

                print(self.findword(text_data[i]) )
                print(self.findword(aspect_data[i]) )
                # print(score_texts[i])
                # print(score_imgs[i])
                # print(truth_res[i], pred_res[i])



            acc = self.get_accuracy(truth_res, pred_res)
            return avg_loss, acc


    def evaluate(self, loss_function, name='dev'):
        # switch models to evaluation mode
        self.model.eval()
        avg_loss = 0.0
        truth_res = []
        pred_res = []

        if name == 'dev':
            evaluate_data = self.dev_data_loader
        elif name == 'test':
            evaluate_data = self.test_data_loader

        with torch.no_grad():
            for batch, sample_batched in enumerate(evaluate_data):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                truth_res += list(targets.data)
                outputs = self.model(inputs)
                pred_label = outputs.data.max(1)[1].cpu().numpy()
                pred_res += [x for x in pred_label]
                loss = loss_function(outputs, targets)
                avg_loss += loss.item()

            avg_loss /= len(self.dev_data_loader)
            acc, f1 = self.get_metrics(truth_res, pred_res)
            return avg_loss, acc, f1


    def train_epoch(self, loss_function, optimizer, i):
        # switch models to training mode, clear gradient accumulators
        # print('epoch:', i)
        self.model.train()
        avg_loss = 0.0
        count = 0
        truth_res = []
        pred_res = []
        for i_batch, sample_batched in enumerate(self.train_data_loader):
            optimizer.zero_grad()
            inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            targets = sample_batched['polarity'].to(self.opt.device)
            truth_res += list(targets.data)
            outputs = self.model(inputs)
            pred_label = outputs.data.max(1)[1].cpu().numpy()
            pred_res += ([x for x in pred_label])
            loss = loss_function(outputs, targets)
            avg_loss += loss.item()
            count += 1
            if count % self.opt.log_step == 0:
                dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
                # print('"%s" iteration: [%d/%d] loss: %g' % (dt, count * self.opt.batch_size, len(self.my_dataset.train_data), loss.item()))
            loss.backward()
            optimizer.step()

        avg_loss /= len(self.train_data_loader)
        acc, f1 = self.get_metrics(truth_res, pred_res)
        return avg_loss, acc, f1



if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='mimn', type=str, help='ian, ram, lstm, ae_lstm, memnet, mimn')
    parser.add_argument('--cnn_model_name', default='resnet50', type=str, help='resnet50, resnet18, vgg')
    parser.add_argument('--dataset', default='zol_cellphone', type=str, help='restaurant, laptop, zol_cellphone')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--logdir', default='log', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--embed_dim_img', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--max_seq_len', default=300, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=0, type=str)
    parser.add_argument('--early_stop', default=10, type=int)
    parser.add_argument('--test_mode', default=False, type=bool)

    opt = parser.parse_args()
    if opt.dataset == 'zol_cellphone':
        opt.polarities_dim = 8

    model_classes = {
        'lstm': LSTM,
        'ae_lstm': AELSTM,
        'td_lstm': TD_LSTM,
        'ian': IAN,
        'ian2m': IAN2m,
        'memnet': MemNet,
        'memnet2': MemNet2,
        'ram': RAM,
        'ram2': MIMN,
        'ram2m': RAM2m,
        'cabasc': Cabasc
    }

    input_colses = {
        'lstm': ['text_raw_indices'],
        'ae_lstm': ['text_raw_indices', 'aspect_indices'],
        'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'ian': ['text_raw_indices', 'aspect_indices'],
        'ian2m': ['text_raw_indices', 'aspect_indices', 'imgs', 'num_imgs'],
        'memnet': ['text_raw_indices', 'aspect_indices'],
        'memnet2': ['text_raw_indices', 'aspect_indices', 'imgs', 'num_imgs'],
        'ram': ['text_raw_indices', 'aspect_indices'],
        'mimn': ['text_raw_indices', 'aspect_indices', 'imgs', 'num_imgs'],
        'ram2m': ['text_raw_indices', 'aspect_indices', 'imgs', 'num_imgs'],
        'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    ins = Instructor(opt)
    # ins.test2('zol_cellphone_mimn_best_acc_61.59_f1_60.51_time_2018-09-05-17-11.models')
    ins.run()
