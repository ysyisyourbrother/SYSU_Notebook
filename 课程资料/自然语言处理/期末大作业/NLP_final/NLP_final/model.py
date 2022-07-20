# -*- coding: utf-8 -*
"""
Author: Junkai-Yan
File: model.py
Finished in 2019/12/16
This file is a head file, including NN model of Encoder and Decoder with attention.
Encoder is a easy network consists of Embedding and LSTM.
Decoder is a complicated network:
Firstly, there are Embedding layer and LSTM.
Secondly, calculate attention weight according to the output of LSTM and the output of Encoder.
Thirdly, calculate context vector using a Linear layer.
Fourthly, calculate attention vector using a Linear layer with a tanh.
Lastly, calculate output using a Linear layer.
The principle of how to calculate attention vector is in page 18~19 of PPT.
**********
What you should focus on is:
The dimension transformation after every layer.
What's more, using transpose function to adjust dimension of Tensor is convenient.
The initial hidden state of Decoder is from the last hidden state of Encoder.(It will be shown in train.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """Encoder network"""
    def __init__(self, vocb_size, n_dim, hidden_dim):
        """
        :param vocb_size: number of non repeating words
        :param n_dim: the dimension of embedded word
        :param hidden_dim: the dimension of hidden layer
        """
        super(Encoder, self).__init__()
        self.n_word = vocb_size
        self.hidden_dim = hidden_dim
        self.n_dim = n_dim
        # layer
        self.embedding = nn.Embedding(self.n_word, self.n_dim)
        self.lstm = nn.LSTM(self.n_dim, self.hidden_dim, bidirectional=True)

    def forward(self, x, hidden=None):
        """
        :param x: input sequence
        :param hidden: hidden state
        :return: output and the last hidden state
        """
        seq_len, batch_size = x.size()
        # init hidden state with zero
        if hidden is None:
            h_0 = x.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = x.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # (seq_len, bs) -> (seq_len, bs, n_dim)
        output = self.embedding(x)
        # output size:(seq_len, bs, n_dim) -> (seq_len, bs, hidden_dim*2)
        output, hidden = self.lstm(output, (h_0, c_0))
        # (seq_len, bs, hidden_dim*2) -> (seq_len, bs, hidden_dim)
        output = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]
        return output, hidden


class Attention(nn.Module):
    """Calculate attention weight"""
    def __init__(self, hidden_dim):
        """
        :param hidden_dim: hidden layer's dimension for encoder and decoder
        """
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_layer = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, ht, hs):
        """
        :param ht: current target hidden state, size is (1, bs, hidden_dim)
        :param hs: the source hidden state, that is, the hidden state of encoder, size is(seq_len, bs, hidden_dim)
        :return: attention weight
        """
        # (1, bs, hidden_dim) -> (bs, 1, hidden_dim)
        ht = ht.transpose(0, 1)
        # (seq_len, bs, hidden_dim) -> (bs, seq_len, hidden_dim)
        hs = hs.transpose(0, 1)
        # (bs, seq_len, hidden_dim) -> (bs, hidden_dim, seq_len)
        whs = self.attention_layer(hs).transpose(1, 2)
        # (bs, 1, hidden_dim) * (bs, hidden_dim, seq_len) = (bs, 1, seq_len)
        score = torch.bmm(ht, whs)
        weight = F.softmax(score, dim=2)
        return weight


class Decoder(nn.Module):
    """Decoder with attention"""
    def __init__(self, vocb_size, n_dim, hidden_dim):
        """
        :param vocb_size: number of non repeating words
        :param n_dim: the dimension of embedded word
        :param hidden_dim: the dimension of hidden layer
        """
        super(Decoder, self).__init__()
        self.n_word = vocb_size
        self.hidden_dim = hidden_dim
        self.n_dim = n_dim

        # this layer calculate attention weight
        self.attention = Attention(self.hidden_dim)
        self.embedding = nn.Embedding(self.n_word, self.n_dim)
        self.lstm = nn.LSTM(self.n_dim, self.hidden_dim)
        self.wc = nn.Linear(2*self.hidden_dim, hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.n_word)

    def forward(self, x, hidden_state, encoder_output):
        """
        :param x: target sequence for partly input
        :param hidden_state: the last hidden state of encoder
        :param encoder_output: total output of encoder, size is (seq_len, bs, hidden_dim)
        :return: output and the last hidden state
        """
        embedded = self.embedding(x)
        # output_lstm size: (1, bs, hidden_dim)
        output_lstm, hidden = self.lstm(embedded, hidden_state)

        # the principle of the code behind comes from PPT page 18
        # attention_weight size: (bs, 1, seq_len)
        attention_weight = self.attention(output_lstm, encoder_output)
        # (seq_len, bs, hidden_dim) -> (bs, seq_len, hidden_dim)
        encoder_output = encoder_output.transpose(0, 1)
        # (bs, 1, seq_len) * (bs, seq_len, hidden_dim) = (bs, 1, hidden_dim)
        context = torch.bmm(attention_weight, encoder_output)
        # (bs, 1, hidden_dim) -> (1, bs, hidden_dim)
        context = context.transpose(0, 1)
        # (1, bs, hidden_dim) + (1, bs, hidden_dim) = (1, bs, 2*hidden_dim)
        context = torch.cat((context, output_lstm), 2)
        # (1, bs, 2*hidden_dim) -> (1, bs, hidden_dim)
        context = self.wc(context)
        ht = F.torch.tanh(context)
        output = self.out(ht)
        return output, hidden
