# -*- coding: utf-8 -*
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocb_size, vocb_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.vocb_size = vocb_size
        self.hidden_dim = hidden_dim
        self.vocb_dim = vocb_dim
        # 把所有的词语embedding到较低维空间中表示
        self.embedding = nn.Embedding(self.vocb_size, self.vocb_dim)   # 对中文词嵌入
        # 建立一个双向lstm 每个time_step会有两个状态的输出部分
        self.lstm = nn.LSTM(self.vocb_dim, self.hidden_dim, bidirectional=True)

    def forward(self, x):
        seq_len, batch_size = x.size()
        # h0 = x.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        # c0 = x.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        # 用正态分布初始化lstm状态 双向单层的lstm需要大小为 2*batch*hidden_dim 的h和c
        h0=torch.normal(0,0.01,[2,batch_size, self.hidden_dim]).cuda()
        c0=torch.normal(0,0.01,[2,batch_size, self.hidden_dim]).cuda()
        input_word = self.embedding(x) # 获取embedding后的句子 embedding需要是long类型的tensor
        # 双向lstm输入数据格式为input(seq_len, batch, input_size)
        # 输出数据格式为output(seq_len, batch, hidden_dim*2) hidden是最后个time_step的输出结果 hidden(batch, hidden_dim*2)
        output, hidden = self.lstm(input_word, (h0, c0))
        # 将前向和后向的输出加在一起输出
        h_n=hidden[0][0]+hidden[0][1]
        c_n=hidden[1][0]+hidden[1][1]
        output = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:] # 将前向和后向的hidden_dim加起来,方便作为输入放入解码器中
        return output, (h_n,c_n) # 返回所有的输出output 和最后一个隐藏层的输出


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_layer = nn.Linear(self.hidden_dim, self.hidden_dim) #经过linear相当于 Whs

    def forward(self, ht, hs):
        ht = ht.transpose(0, 1) # 转成(batch, 1, hidden_dim)
        hs = hs.transpose(0, 1) # 转成 (batch, seq_len, hidden_dim)
        Whs = self.attention_layer(hs).transpose(1, 2) # (batch seq_len, hidden_dim) -> (batch, hidden_dim, seq_len)
        score = torch.bmm(ht, Whs)  # bmm是带了batch的，对后两维度做矩阵乘法(batch, 1, hidden_dim) * (batch, hidden_dim, seq_len) = (batch, 1, seq_len)
        weight = F.softmax(score, dim=2) # 0-batch  2-score 对第二维进行softmax
        # ct = torch.bmm(weight,hs) # 计算contex vector ct(batch,1,hidden_dim)
        # return ct
        return weight


class Decoder(nn.Module):
    def __init__(self, vocb_size, n_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.n_word = vocb_size
        self.hidden_dim = hidden_dim
        self.n_dim = n_dim

        self.attention = Attention(self.hidden_dim) # 调用上面的Attention计算注意力机制的context
        self.embedding = nn.Embedding(self.n_word, self.n_dim)  # 对英文词嵌入
        self.lstm = nn.LSTM(self.n_dim, self.hidden_dim)
        self.wc = nn.Linear(2*self.hidden_dim, hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.n_word)
        self.sm = torch.nn.Softmax(2)

    def forward(self, x, init_state, encoder_output):
        ### decoder每次将上一次输出的词语，在作为输入到下一次
        ### decoder 的输入x是(1*batchsize)  1为输入词语的下标
        ### 因此lstm每次只有一个时间片，并且h初始化为上一次输出结果的h
        input_word = self.embedding(x)
        # output_lstm size: (1, batch_size, hidden_dim)
        output_lstm, hidden = self.lstm(input_word, init_state)

        # attention_weight size: (batch_size, 1, seq_len)
        attention_weight = self.attention(output_lstm, encoder_output)
        # 将batch转到开头(seq_len, batch_size, hidden_dim) -> (batch_size, seq_len, hidden_dim)
        encoder_output = encoder_output.transpose(0, 1)

        # 进行带batch的矩阵乘法
        # (batch_size, 1, seq_len) * (batch_size, seq_len, hidden_dim) = (batch_size, 1, hidden_dim)
        context = torch.bmm(attention_weight, encoder_output)

        # 因为是batchsize不是优先的，所以昨晚乘法有要再把batchsize转回去
        # (batch_size, 1, hidden_dim) -> (1, batch_size, hidden_dim)
        context = context.transpose(0, 1)

        # 将attention的context和decoder输出向量拼接
        # (1, batch_size, hidden_dim) + (1, batch_size, hidden_dim) = (1, batch_size, 2*hidden_dim)
        context = torch.cat((context, output_lstm), 2)

        # 用一个全连接层将attention的和decoder输出结合的向量投影到和词语数量维度匹配的
        # (1, batch_size, 2*hidden_dim) -> (1, batch_size, hidden_dim)
        # # 最后将结果做一个softmax映射到0-1的概率上去
        context = self.wc(context)
        ht = F.torch.tanh(context)
        output = self.out(ht)
        # output = self.sm(self.out(ht))
        return output, hidden
