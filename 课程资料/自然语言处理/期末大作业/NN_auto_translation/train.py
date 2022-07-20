# -*- coding: utf-8 -*
import math

import torch
from torch import nn, optim
import random
import numpy as np
import matplotlib.pyplot as plt
from model import *
from nltk.translate.bleu_score import sentence_bleu
import sys
# import os
# os.chdir(r'drive/My Drive/NN_auto_translation')

BOS=0
EOS=1
UKN=2
PAD=3

def get_data(filename, length=8000):
    data = list()
    f = open(filename, 'r', encoding='utf-8')
    for line in f:
        line = line.strip().split()
        data.append(line)
        if len(data) == length:
            break
    f.close()
    return data

def countword(filename):
    f = open(filename, 'r', encoding='utf-8')
    return len(f.readlines())

def padding(batch_data, pad):
    #在词表中默认<PAD> 为3 <EOS> 为1
    padding_data = list()
    # 统计最长的句子长度
    max_length = 0
    for data in batch_data:
        max_length = len(data) if len(data) > max_length else max_length
    for data in batch_data:
        # 如果没到最长的长度，就进行padding
        if len(data) < max_length:
            data = data + (max_length-len(data))*[pad]
        padding_data.append(data)
    return np.array(padding_data, dtype='int64'),max_length

def next_batch(batch_num,training_data,batch_size,training_label):
    # 分批喂数据
    i = batch_num % (math.ceil(len(training_data) / batch_size))
    if (i + 1) * batch_size >= len(training_data):
        train_data = training_data[i * batch_size:]+training_data[:((i+1)*batch_size)-len(training_data)]
        train_label = training_label[i * batch_size:]+training_label[:((i+1)*batch_size)-len(training_data)]
        flag = False
    else:
        train_data = training_data[i * batch_size:(i + 1) * batch_size]
        train_label = training_label[i * batch_size:(i + 1) * batch_size]
        flag = True
    return train_data,train_label,flag

def train():
    # 设置超参数
    batch_size = 500
    epochs = 2000
    encoder_lr = 1e-3
    decoder_lr = 1e-3
    Curriculum_Learning_ratio = 0.5

    # 读取预处理后的数据集
    training_data = get_data('./preprocessing/train_source_8000.txt')
    training_label = get_data('./preprocessing/train_target_8000.txt')
    source_num_word = countword('./preprocessing/word_dic_train_source_8000.txt')
    target_num_word = countword('./preprocessing/word_dic_train_target_8000.txt')

    # 定义编码器和解码器的网络
    encoder = Encoder(source_num_word, 50, 100).cuda()
    decoder = Decoder(target_num_word, 50, 100).cuda()
    # encoder = Encoder(source_num_word, 600, 300).cuda()
    # decoder = Decoder(target_num_word, 600, 300).cuda()

    # 定义优化器
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=encoder_lr)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=decoder_lr)

    # 定义交叉熵
    criterion = nn.CrossEntropyLoss(reduce=True)

    # 统计epoch的误差
    epoch_loss_list = list()


    for epoch in range(epochs):
        batch_num=0
        flag = True # 判断是否全部使用了一次数据集
        epoch_loss = 0.0 # 每epoch记录loss
        while flag:
            # 拿一批数据喂入
            train_data,train_label,flag = next_batch(batch_num,training_data,batch_size,training_label)

            # padding 训练数据和测试数据到和最长句子一样的长度
            train_data, _ = padding(train_data, PAD)
            train_label,max_length = padding(train_label, BOS)


            # 应用编码器输出结果
            train_data = torch.LongTensor(train_data).transpose(0, 1).cuda() # 因为batchsize不是第一维度，进行transpose->(seq_len, bs)
            train_label = torch.LongTensor(train_label).transpose(0, 1).cuda()
            encoder_out, (h_n, h_c) = encoder(train_data) # 将编码器最后一个时刻的输出结果作为init输入到解码器中
            decoder_init = (h_n.unsqueeze(0), h_c.unsqueeze(0)) # 因为decoder只有一个lstm节点，所以在第一维加个为1的维度符合init的输入

            # 应用解码器，并计算损失并优化
            loss = 0.0  # 统计总的误差
            first_input = torch.full(size=[1,batch_size],fill_value=BOS).long().cuda() # 第一步都是<BOS>开始 没有loss
            decoder_output, decoder_hc = decoder(first_input, decoder_init, encoder_out) # 将句子编码的结果放入解码器的初始状态
            # values, word_index = torch.max(decoder_output, 2)
            for step in range(1,max_length):
                Curriculum_Learning = True if random.random() < Curriculum_Learning_ratio else False
                ### 使用Curriculum Learning的学习方法 一定概率输入ground truth 或前一步的decoder输出
                if Curriculum_Learning:
                    ground_truth = train_label[step].unsqueeze(0) # 选出正确答案中的词语的index
                    decoder_output, decoder_hc = decoder(ground_truth, decoder_hc, encoder_out)
                else:
                    _, last_word = torch.max(decoder_output, 2)  # 选出概率最高的一个词语的下标作为解码器的输入
                    decoder_output, decoder_hc = decoder(last_word, decoder_hc, encoder_out)
                # 对解码器每一个输出求loss
                loss += criterion(decoder_output.squeeze(), train_label[step])
            print("batch_num:",batch_num,"loss:",loss.item()/batch_size) # 平均每个样本max_length时间片总误差
            batch_num += 1  # 当前一共训练了多少个batch

            # 反向传播更新公式
            encoder_optimizer.zero_grad()  # 清楚梯度累积
            decoder_optimizer.zero_grad()
            loss.backward(retain_graph=True) # 反向传播计算梯度
            encoder_optimizer.step()  # 应用优化器更新参数
            decoder_optimizer.step()
            epoch_loss += loss.item() # 计算每一次epoch平均的loss

        # epoch的loss
        epoch_loss_list.append(epoch_loss)
        print('epoch: {} of {}'.format(epoch + 1, epochs))
        print('loss: {}'.format(epoch_loss))
        print()
        if (epoch%5==0):
            # save the model
            torch.save(encoder, 'encoder.pkl')
            torch.save(decoder, 'decoder.pkl')

    # plot the variation of loss
    plt.figure()
    plt.plot(epoch_loss_list, label='training loss')
    plt.xlabel('epoch')
    # plt.xlabel('batch_num')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss.png', format='png')
    plt.show()



if __name__=="__main__":
    train()