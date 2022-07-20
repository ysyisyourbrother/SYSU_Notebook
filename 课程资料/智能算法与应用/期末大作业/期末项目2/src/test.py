import csv
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

# 测试数据集
f = open('submission.csv', 'w', newline='') # 要写入的测试文件
writer = csv.writer(f)
writer.writerow(['PhraseId', 'Sentiment'])
net = torch.load('network.pkl').cuda()  # 加载模型

# 读取测试数据
df = pd.read_csv('test.tsv', sep='\t')
test_phrase_to_int = []
# 将phrase的词语列表替换为int列表   如果词语不存在则用0代替
for i in range(len(df)):
    test_phrase_to_int.append([vocab_to_int.get(word, 0) for word in df.Phrase.values[i].lower().split()])
# 对数据进行padding
test_phrase_to_int = Pad_sequences(test_phrase_to_int,30)

# 遍历测数据计算结果
id = 156061
for i in range(len(test_phrase_to_int)):
    if i % 3000 ==0:
        print("======>",i)
    feature = torch.LongTensor(test_phrase_to_int[i])
    feature = torch.unsqueeze(feature, dim=0)
    feature = Variable(feature.cuda())
    score = net(feature)
    res = torch.argmax(score.data.cpu())
    writer.writerow([id, res.item()])
    id += 1
f.close()
    