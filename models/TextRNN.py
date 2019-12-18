# coding: utf-8

import torch
import torch.nn as nn
import numpy as np

class Config():
    '''配置模型参数'''
    def __init__(self, dataset, embedding):
        self.model_name = 'TextRNN'
        self.train_path = dataset + '/data/train.txt'  # 训练集路径
        self.dev_path = dataset + '/data/dev.txt'      # 开发集路径
        self.test_path = dataset + '/data/test.txt'    # 测试集路径
        self.class_path = dataset + '/data/class.txt'  # 文本类别列表路径
        self.class_list = [x.strip() for x in open(self.class_path, 'r', encoding='utf-8').readlines()]  # 文本类别列表
        self.vocab_path = dataset + '/data/vocab.pkl'  # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name  # 模型训练日志文件
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 训练设备
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype(np.float64))\
            if embedding != 'random' else None                                       # 预训练词向量
        # 以下是模型的参数配置
        self.dropout = 0.5                       # 丢弃率
        self.require_improvement = 1000          # 若超过1000batch性能还没有提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 文本类别数量
        self.n_vocab = 0                         # 词表大小，在运行时赋值
        self.num_epochs = 10                     # 训练轮次
        self.batch_size = 128                    # mini-batch的大小
        self.pad_size = 32                       # 每句话处理成的长度（短填长切）
        self.learning_rate = 1e-3                # 学习率
        self.hidden_size = 128                   # lstm隐藏层特征数
        self.num_layers = 2                      # lstm层数
        self.embed = 300                         # 词向量的维度

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        x, _ = x
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out