# 更改当前路径
import os
import pandas as pd
path = "/content/drive/My Drive/Movie Sentimental Analysis"
os.chdir(path)

# Any results you write to the current directory are saved as output.
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

# 判断是否使用了gpu
import tensorflow as tf
tf.test.gpu_device_name()

# 读取数据
train = pd.read_csv('train.tsv', sep='\t')

# 标签数据
train_labels = torch.tensor(train['Sentiment'].values)   # 训练集标签

# 读取预训练的语言模型  glove-100
def read_glove():
    print("reading glove...")
    glove = {}
    with open("glove.txt", "r") as f:
        for line in f.readlines():
            line=line.split()
            word = line[0]
            vector = np.array(line[1:]).astype(np.float)
            # 如果该词语在词典中存在
            if vocab_to_int.get(word, -1) != -1:
                glove[vocab_to_int[word]] = vector
    # 定义glove[0] <unk>为0
    glove[0] = [0]*100
    return glove
# 读取glove模型
glove = read_glove()

# 将phrase padding 为相同的长度，多删少补
def Pad_sequences(phrase_to_int,seq_length):
    print("padding phrase")
    pad_sequences = np.zeros((len(phrase_to_int), seq_length),dtype=int)
    for idx,row in tqdm(enumerate(phrase_to_int),total=len(phrase_to_int)):
        pad_sequences[idx, :len(row)] = np.array(row)[:seq_length]
    return pad_sequences


def Corpus_Extr(df):
    print('Construct Corpus...')
    corpus = []
    # 将phrase分为一个个词语
    for i in tqdm(range(len(df))):
        corpus.extend(df.Phrase[i].lower().split())
    corpus = Counter(corpus)
    corpus2 = sorted(corpus,key=corpus.get,reverse=True)
    print('Convert Corpus to Integers')
    # 构建word:int字典
    vocab_to_int = {word: idx for idx,word in enumerate(corpus2,1)}
    # 构建int:word字典
    int_to_vocab = {idx : word for idx,word in enumerate(corpus2,1)}
    print('Convert Phrase to Integers')
    phrase_to_int = []
    # 将phrase的词语列表替换为int列表
    for i in tqdm(range(len(df))):
        phrase_to_int.append([vocab_to_int[word] for word in df.Phrase.values[i].lower().split()])
    return corpus, vocab_to_int, int_to_vocab, phrase_to_int
corpus, vocab_to_int,int_to_vocab, phrase_to_int = Corpus_Extr(train)


# 对于不存在的词语用'<unk>'来表示
vocab_to_int['<unk>'] = 0
int_to_vocab[0] = '<unk>'