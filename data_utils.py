# -*- coding: utf-8 -*-
# file: data_utils_ai.py
# author: xunan <xunan0812@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import jieba
import re
import torch
import time
import pickle
import os
from PIL import Image
from PIL import ImageFile
from torch import nn
import numpy as np
from torchvision import transforms
from collections import Counter
from torch.utils.data import Dataset
from torchvision.models import alexnet, resnet18, resnet50, inception_v3

np.random.seed(1337)  # for reproducibility

def dp_txt(txt):
    http_pattern = re.compile(
        "((http|ftp|https)://)(([a-zA-Z0-9\._-]+\.[a-zA-Z]{2,6})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(:[0-9]{1,4})*(/[a-zA-Z0-9\&%_\./-~-]*)?")
    url_pattern = re.compile('(www|WWW)\\.[0-9%a-zA-Z\\.]+\\.(com|cn|org)')
    txt = txt.strip()
    txt = re.sub(http_pattern, '', txt)
    txt = re.sub(url_pattern, '', txt)
    return txt

def jieba_cut(text):
    text = dp_txt(text)
    stopwords = {}.fromkeys([line.rstrip() for line in open('./datasets/stopwords.txt', encoding='utf-8')])
    segs = jieba.cut(text, cut_all=False)

    final = ''
    for seg in segs:
        seg = str(seg)
        if seg not in stopwords:
            final += seg
    seg_list = jieba.cut(final, cut_all=False)
    text_cut = ' '.join(seg_list)
    return text_cut

def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.dat'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.random.rand(len(word2idx) + 2, embed_dim)  # idx 0 and len(word2idx)+1 are all-zeros
        fname = '../../datasets/GloveData/glove.6B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else '../../datasets/ChineseWordVectors/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim' + str(embed_dim) + '.iter5'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, lower=False, max_seq_len=None, max_aspect_len=None):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.max_aspect_len = max_aspect_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    @staticmethod
    def pad_sequence(sequence, maxlen, dtype='int64', padding='pre', truncating='pre', value=0.):
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def text_to_sequence(self, text, isaspect=False , reverse=False):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        pad_and_trunc = 'post'  # use post padding together with torch.nn.utils.rnn.pack_padded_sequence
        if reverse:
            sequence = sequence[::-1]
        if isaspect:
            return Tokenizer.pad_sequence(sequence, self.max_aspect_len, dtype='int64',
                                          padding=pad_and_trunc, truncating=pad_and_trunc)
        else:
            return Tokenizer.pad_sequence(sequence, self.max_seq_len, dtype='int64',
                                          padding=pad_and_trunc, truncating=pad_and_trunc)

class ABSADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            text_left_indices = tokenizer.text_to_sequence(text_left)
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            polarity = int(polarity)+1

            data = {
                'text_raw_indices': text_raw_indices,
                'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                'text_left_indices': text_left_indices,
                'text_left_with_aspect_indices': text_left_with_aspect_indices,
                'text_right_indices': text_right_indices,
                'text_right_with_aspect_indices': text_right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'polarity': polarity,
            }
            all_data.append(data)
        return all_data

    def __init__(self, dataset='restaurant', embed_dim=100, max_seq_len=40):
        print("preparing {0} datasets...".format(dataset))
        fname = {
            'restaurant': {
                'train': './datasets/semeval14/Restaurants_Train.xml.seg',
                'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
            },
            'laptop': {
                'train': './datasets/semeval14/Laptops_Train.xml.seg',
                'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
            }
        }
        text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
        tokenizer = Tokenizer(max_seq_len=max_seq_len)
        tokenizer.fit_on_text(text.lower())
        self.embedding_matrix, self.word2idx = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer))


class ZOLDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ZOLDatesetReader:
    @staticmethod
    def __data_Counter__(fnames):
        jieba_counter = Counter()
        label_counter = Counter()
        max_length_text = 0
        min_length_text = 1000
        max_length_img = 0
        min_length_img = 1000
        lengths_text = []
        lengths_img = []
        for fname in fnames:
            with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
                lines = fin.readlines()
                for i in range(0, len(lines), 4):
                    text_raw = lines[i].strip()
                    imgs = lines[i + 1].strip()[1:-1].split(',')
                    aspect = lines[i + 2].strip()
                    polarity = lines[i + 3].strip()

                    length_text = len(text_raw)
                    length_img = len(imgs)

                    if length_text >= max_length_text:
                        max_length_text = length_text
                    if (length_text <= min_length_text):
                        min_length_text = length_text
                    lengths_text.append(length_text)

                    if length_img >= max_length_img:
                        max_length_img = length_img
                    if (length_img <= min_length_img):
                        min_length_img = length_img
                    lengths_img.append(length_img)


                    jieba_counter.update(text_raw)
                    label_counter.update([polarity])
        print(
            'data_num:', len(lengths_text),
            'max_length_text:', max_length_text,
            'min_length_text:', min_length_text,
            'ave_length_test:', np.average(np.array(lengths_text)),
            'max_length_img:', max_length_img,
            'min_length_img:', min_length_img,
            'ave_length_img:', np.average(np.array(lengths_img)),
            'jieba_num:', len(jieba_counter)
        )
        print(label_counter)

        # data_num: 28429
        # max_length_text: 8511
        # min_length_text: 5
        # ave_length_test: 315.106651659
        # max_length_img: 111
        # min_length_img: 1
        # ave_length_img: 4.49984171093
        # jieba_num: 3389

        # data_num: 28429
        # max_length_text: 8511
        # min_length_text: 5
        # ave_length_text: 315.106651659
        # max_length_img: 111
        # min_length_img: 1
        # ave_length_img: 4.49984171093
        # jieba_num: 3389

    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
                lines = fin.readlines()
                for i in range(0, len(lines), 4):
                    text_raw = lines[i].strip()
                    text += text_raw + " "
        return text


    def read_img(self, imgs_path):
        imgs = []
        for j in range(len(imgs_path)):
            img_path = imgs_path[j].strip().replace('\'', '')
            try:
                img = Image.open('/home/xunan/code/pytorch/ZOLspider/multidata_zol/img/' + img_path).convert('RGB')
                input = self.transform_img(img).unsqueeze(0)
                output = self.cnn_extractor(input).squeeze()
                imgs.append(output)
                img.close()
            except:
                error = 1
        embed_dim_img = len(imgs[0])
        img_features = torch.zeros(self.max_img_len, embed_dim_img)
        num_imgs = len(imgs)
        if num_imgs >= self.max_img_len:
            for i in range(self.max_img_len):
                img_features[i,:] = imgs[i]
        else:
            for i in range(self.max_img_len):
                if i < num_imgs:
                    # img_features[(self.max_img_len-num_imgs)+i,:] = imgs[i]
                    img_features[i, :] = imgs[i]
                else:
                    break
        return img_features, min(self.max_img_len, num_imgs)
    # @staticmethod
    def read_data(self, fname, tokenizer):
        polarity_dic = {'10.0': 8, '8.0': 7, '6.0': 6, '5.0': 5, '4.0': 4, '3.0': 3, '2.0': 2, '1.0': 1}
        data_path = fname.split('.txt')[0]+'/'
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        data_path = fname.split('.txt')[0]+'/'+self.cnn_model_name+'/'
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            lines = fin.readlines()
            all_data = []
            for i in range(0, len(lines), 4):
                fname_i = data_path + str(int(i/4)) + '.pkl'
                if os.path.exists(fname_i):
                    with open(fname_i, 'rb') as fpkl:
                        data = pickle.load(fpkl)
                else:
                    print(fname_i)
                    text_raw = lines[i].strip()
                    imgs, num_imgs = self.read_img(lines[i + 1].strip()[1:-1].split(','))
                    aspect = lines[i + 2].strip()
                    polarity = int(polarity_dic[(lines[i + 3].strip())]-1)
                    text_raw_indices = tokenizer.text_to_sequence(text_raw, isaspect=False)
                    aspect_indices = tokenizer.text_to_sequence(aspect, isaspect=True)
                    data = {
                        'text_raw_indices': text_raw_indices,
                        'imgs': imgs,
                        'num_imgs': num_imgs,
                        'aspect_indices': aspect_indices,
                        'polarity': int(polarity),
                    }
                    with open(fname_i, 'wb') as fpkl:
                        pickle.dump(data, fpkl)
                all_data.append(data)
        return all_data

    def __init__(self, dataset='zol_cellphone', embed_dim=100, max_seq_len=320, max_aspect_len=2, max_img_len=5, cnn_model_name='resnet50'):
        start = time.time()
        print("Preparing {0} datasets...".format(dataset))
        fname = {
            'zol_cellphone': {
                'train': './datasets/zolDataset/zol_Train_jieba.txt',
                'dev': './datasets/zolDataset/zol_Dev_jieba.txt',
                'test': './datasets/zolDataset/zol_Test_jieba.txt'
            }
        }

        cnn_classes = {
            'resnet18': resnet18(pretrained=True),
            'resnet50': resnet50(pretrained=True),
            'alexnet': alexnet(pretrained=True)
        }

        self.cnn_model_name = cnn_model_name
        self.max_img_len = max_img_len
        self.cnn_extractor = nn.Sequential(*list(cnn_classes[cnn_model_name].children())[:-1])
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            ])

        text = ZOLDatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['dev'], fname[dataset]['test']])
        tokenizer = Tokenizer(max_seq_len=max_seq_len, max_aspect_len=max_aspect_len)
        tokenizer.fit_on_text(text)

        self.word2idx = tokenizer.word2idx
        self.idx2word = tokenizer.idx2word

        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        self.train_data = ZOLDataset(self.read_data(fname[dataset]['train'], tokenizer))
        self.dev_data = ZOLDataset(self.read_data(fname[dataset]['dev'], tokenizer))
        self.test_data = ZOLDataset(self.read_data(fname[dataset]['test'], tokenizer))
        end = time.time()
        m, s = divmod(end-start, 60)
        print('Time to read datasets: %02d:%02d' % (m, s))


if __name__ == '__main__':

    # text_zol = ZOLDatesetReader.__read_text__(['./datasets/zolDataset/zol_Train_jieba.txt',
    #                                            './datasets/zolDataset/zol_Dev_jieba.txt',
    #                                            './datasets/zolDataset/zol_Test_jieba.txt'])
    # counter_zol = ZOLDatesetReader.__data_Counter__(['./datasets/zolDataset/zol_Train_jieba.txt',
    #                                            './datasets/zolDataset/zol_Dev_jieba.txt',
    #                                            './datasets/zolDataset/zol_Test_jieba.txt'])
    zol_dataset = ZOLDatesetReader()

