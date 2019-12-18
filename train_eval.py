# coding: utf-8

import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
from utils import get_time_dif

def train(config, model, train_iter, dev_iter, test_iter):
    '''
    :param config: 模型参数配置
    :param model: 模型
    :param train_iter: 训练集迭代器
    :param dev_iter: 开发集迭代器
    :param test_iter: 测试集迭代器
    :return:
    '''
    start_time = time.time()
    model.train()  # 训练模式
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)  # 优化器
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            # 计算loss
            loss = F.cross_entropy(outputs, labels)
            # 反向传播
            model.zero_grad()
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每100个batch输出在训练集和开发集上的效果
                true = labels.data.cpu().tolist()
                predict = torch.max(outputs.data, 1)[1].cpu().tolist()
                train_acc = metrics.accuracy_score(true, predict)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)  # 保存当前模型参数
                    improve = '*'  # *代表性能有提升
                    last_improve = total_batch
                else:
                    improve = ''   # 性能没有提升
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()

            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 开发集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)

def test(config, model, test_iter):
    '''
    :param config: 模型的参数配置
    :param model: 模型
    :param test_iter: 测试数据集
    :return:
    '''
    # test
    model.load_state_dict(torch.load(config.save_path))  # 加载当前性能最好的模型
    model.eval()  # 评价模式
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    '''
    :param config: 模型的参数配置
    :param model: 模型
    :param test_iter: 测试数据集
    :return:
    '''
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)