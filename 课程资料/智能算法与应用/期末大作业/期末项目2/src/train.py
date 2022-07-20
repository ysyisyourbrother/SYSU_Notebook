# 训练模型
import os
import pandas as pd
path = "/content/drive/My Drive/Movie Sentimental Analysis"
os.chdir(path)
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random
import pickle
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn.metrics import accuracy_score
from torch.autograd import Variable

from data_process import *
from model import *

labels = 5  # 要划分类别数
num_epochs = 5
embed_dim = 100
hidden_dim = 128
linear_dim = 64
n_layers = 2
batch_size = 64
lr = 0.001
device = torch.device('cuda:0')


# 读取glove模型
weight = torch.zeros(len(vocab_to_int), embed_dim)
count = 0 
for i in range(len(vocab_to_int)):
        # 若该词语不存在，通<unk>的值
        if i not in glove.keys():
            count+=1
        weight[i, :] = torch.Tensor(glove.get(i,np.random.rand(1,100)))
    
# 整理训练数据
train_features = torch.LongTensor(Pad_sequences(phrase_to_int,30))

train_set = torch.utils.data.TensorDataset(train_features, train_labels)
#train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True)
train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

net = LSTM(hidden_dim, linear_dim, len(vocab_to_int),n_layers,embed_dim, weight, labels)

net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)
# 动态调整学习率，每隔step_size就将学习率调整为0.1倍
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.1, last_epoch=-1)


global_batch_size = 0 # 全局的batchnum统计
for epoch in range(num_epochs):
    start = time.time()
    train_loss, test_losses = 0, 0
    train_acc, test_acc = 0, 0
    n, m = 0, 0
    batch_num = 0
    for feature, label in train_iter:
        n += 1
        batch_num +=1
        global_batch_size+=1
        net.zero_grad()
        feature = Variable(feature.cuda())
        label = Variable(label.cuda())
        score = net(feature)
        loss = loss_function(score, label)
        loss.backward()
        optimizer.step()
        train_acc += accuracy_score(torch.argmax(score.data.cpu(), dim=1), label.cpu())
        train_loss += loss
        
        if batch_num %100 ==0:
            print('batch_num: %d, train loss: %.4f, train acc: %.2f'%
                        (batch_num, train_loss / n, train_acc / n))
            
    scheduler.step()    # 更新学习率
    cur_lr = optimizer.param_groups[0]['lr']    # 查看当前学习率
    end = time.time()
    runtime = end - start
    print('epoch: %d, train loss: %.4f, train acc: %.2f, time: %.2f, learning rate: %f' %
          (epoch, train_loss / n, train_acc / n, runtime, cur_lr))

torch.save(net, 'network.pkl')