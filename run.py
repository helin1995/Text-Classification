# coding: utf-8

import argparse
import time
import torch
import numpy as np
from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif, init_network
from train_eval import train

parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument用于添加参数，default - 不指定参数时的默认值，type - 命令行参数应该被转换成的类型，
# required - 可选参数是否可以省略 (仅针对可选参数)，help - 参数的帮助信息，当指定为 argparse.SUPPRESS 时表示不显示该参数的帮助信息
parser.add_argument('--model', type=str, default='TextRNN', help='模型的名称')
parser.add_argument('--embedding', type=str, default='pre_trained', help='是否采用预训练词向量，pre_trained-采用预训练词向量，random-不采用预训练词向量')
args = parser.parse_args()  # 解析添加的参数，如果要使用参数的话，直接写args.参数名称，比如：args.embedding，它的值就是False

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'  # 要采用的预训练词向量
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model
    print(model_name)
    x = import_module('models.' + model_name)  # 导入模型TextRNN
    config = x.Config(dataset, embedding)  # 生成模型的参数配置
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样，用以保证实验的可重复性

    start_time = time.time()  # 计时开始
    print('加载数据中...')
    vocab, train_data, dev_data, test_data = build_dataset(config)  # 生成数据集
    # print(len(train_data), len(dev_data), len(test_data))
    # 创建数据集迭代器
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print('Time usage: ', time_dif)

    # 更新模型参数配置中的词表长度
    config.n_vocab = len(vocab)
    # 创建模型
    model = x.Model(config).to(config.device)
    # 初始化模型参数
    init_network(model)
    # 打印模型参数
    print(model.parameters)
    # 训练模型
    train(config, model, train_iter, dev_iter, test_iter)