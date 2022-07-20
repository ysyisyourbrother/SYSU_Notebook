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

# 构建model
class LSTM(nn.Module):
    def __init__(self,hidden_dim, linear_dim, n_word, n_layers,embed_dim, weight, labels):
        super().__init__()
        self.n_layers = n_layers    # lstm层数
        self.hidden_dim = hidden_dim    # 隐藏层节点数
        self.embedd_dim = embed_dim   # 使用glove的数据
        self.lables = labels # softmax要划分的类别数
        self.linear_dim = linear_dim # 线形层的节点数
        self.n_word = n_word # 词语总数
        
        # 定义词嵌入层，使用预训练glove模型来进行
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False

        # # 不使用glove，直接进行词语嵌入
        self.embedding = nn.Embedding(self.n_word, 100)

        # 定义双层的LSTM层
        self.lstm = nn.LSTM(self.embedd_dim, self.hidden_dim,num_layers = self.n_layers, bidirectional=True)

        # 定义线形层 使用relu激活函数 因为对lstm的输出进行了拼接，因此这里输入维度是hidden_dim*2
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.linear_dim),
            nn.ReLU(True),
        )

        # 定义全连接层
        self.fc = nn.Linear(self.linear_dim, self.lables)

        
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        # permute [batch_size, seq_len, embedded_size] -> [seq_len, batch_size, embedded_size]
        lstm_out, hidden = self.lstm(embeddings.permute([1, 0, 2]))
        output = self.linear(lstm_out[-1])
        output = self.fc(output)
        return output
