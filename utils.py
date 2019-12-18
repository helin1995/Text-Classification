# coding: utf-8

import os
import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import time
from datetime import timedelta

MAX_VOCAB_SIZE = 10000       # 词表最大长度
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

def build_dataset(config):
    '''
    :param config: 模型参数配置
    :return: 数据集
    '''
    tokenizer = lambda x: [y for y in x]  # 字符级，对输入句子按字切分
    if os.path.exists(config.vocab_path):
        # 判断词表文件是否存在，如果存在这个词表，那么就加载此词表
        vocab = pkl.load(open(config.vocab_path, 'rb'))  # 加载pkl文件，pkl.load(文件对象，不是文件名)
    else:
        # 如果不存在这个词表，那么就利用训练集生成词表
        vocab = build_vocab(config.train_path, tokenizer, MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    # print(vocab)
    print(f'Vocab size: {len(vocab)}')
    # 生成数据集
    train = load_dataset(config.train_path, tokenizer, vocab, pad_size=32)  # 训练集
    dev = load_dataset(config.dev_path, tokenizer, vocab, pad_size=32)      # 开发集
    test = load_dataset(config.test_path, tokenizer, vocab, pad_size=32)    # 测试集
    return vocab, train, dev, test

def build_vocab(filepath, tokenizer, max_size, min_freq):
    '''
    :param filepath: 训练集路径
    :param tokenizer: 切割函数
    :param max_size: 词表最大长度
    :param min_freq: 词出现的最小频率
    :return: 词表
    '''
    vocab_dic = {}  # 词表字典
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            # tqdm是一个快速，可扩展的Python进度条，可以在Python长循环中添加一个进度提示信息，用户只需要封装任意的迭代器tqdm(iterator)
            line = line.strip()
            if not line:
                # 如果line是空字符串，那么结束本次循环，读取下一行
                continue
            content = line.split('\t')[0]
            # 统计训练集中字符出现的频率
            for word in tokenizer(content):
                # 常见为dict.get(a,b):a是键值key，如果dict存在键值a，则函数返回dict[a]；否则返回b，如果没有定义b参数，则返回None。
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        # 获取出现频率前10000的词构成词表
        # dict.items()返回的是有key，value构成元组的列表，即[(key1, value1), (key2, value2),....]
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        # 生成语料的词表，词与id一一对应
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        # dict.update(dict2)，将dict2添加到dict中
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic

def load_dataset(path, tokenizer, vocab, pad_size=32):
    '''
    :param path: 数据集路径
    :param tokenizer: 切分器
    :param vocab: 词表
    :param pad_size: padding长度
    :return: 字符id，文本label和文本的长度
    '''
    contents = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            content, label = line.split('\t')
            word_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if seq_len < pad_size:
                    token.extend([PAD] * (pad_size - seq_len))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                word_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((word_line, int(label), seq_len))
    return contents  # [([...], 0), ([...], 1), ...]

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def get_time_dif(start_time):
    '''
    :param start_time: 开始时间
    :return: 加载数据所用时间
    '''
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def init_network(model, method='xavier', exclude='embedding', seed=123):
    '''
    :param model: 模型
    :param method: 默认的初始化方法
    :param exclude: embedding
    :param seed: 随机种子
    :return:
    '''
    for name, w in model.named_parameters():
        # print(name)
        if exclude not in name:
            if 'weight' in name:
                # 对权重进行初始化
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                # 对偏差进行初始化
                nn.init.constant_(w, 0)
            else:
                pass