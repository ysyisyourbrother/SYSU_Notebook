# -*- coding: utf-8 -*
"""
Author: Junkai-Yan
File: train.py
Finished in 2019/12/17
This file trains the model using dataset_10000(after pre-processing).
Model is in model.py.
Firstly, read in training data, training target and their dict.
Secondly, build the network, pay attention to the vocabulary size of encoder and decoder.
Thirdly, divided data set into some batches.
    For each batches, padding training data and label,
    such that its length equal to the longest sequence of this batch.
Fourthly, throw the batch training data into Encoder, and get the output and the last hidden state.
    In this part, because the LSTM is bidirectional, the output size of it is (seq_len, bs, 2*hidden_dim),
    It show be reshaped to (seq_len, bs, hidden_dim), the way I choose is to add the forward output and
    the backward output, so the output size becomes (seq_len, bs, hidden_dim).
Fifthly, init the initial hidden state of Decoder by the last hidden state of Encoder.
    In this part, the dimension problem discussed before takes place again, my operation is the same.
    Then for each time step, using random.random() to judge if we choose teacher forcing.
    If so, this time step we input the training target to Decoder, else, input the output of previous time step.
    For each time step, calculate the loss between output and target.
    For each batch, update the parameters.
    For each epoch, note down the average loss.
Lastly, save the total model as a 'pkl' file and show the variation of loss.
**********
What you should focus on is:
The dimension of input is not batch first,
so I transpose the input before throw it into Encoder.
What's more, the learning rate should not be too large,
otherwise, it will make the loss oscillate near the minimum.
**********
Hyper parameters:
    learning_rate = 1e-4
    decoder_learning_rate = 5e-4
    batch_size = 100
    epochs = 1000
    teaching_force_ratio = 0.5

Optimizer:
    Encoder: SGD
    Decoder: SGD

Loss function:
    Cross Entropy
"""

import torch
from torch import nn, optim
import random
import numpy as np
import matplotlib.pyplot as plt
from model import *

def get_data(filename, length=8000):
    """
    Get data from the first length rows of the file.
    :param filename: file name
    :param length: the length of data set
    (use it to choose a smaller size to train, default is 8000, the whole data set)
    :return: list, the first length rows of the file
    """
    data = list()
    f = open(filename, 'r', encoding='utf-8')
    for line in f:
        line = line.strip().split()
        data.append(line)
        if len(data) == length:
            break
    f.close()
    return data

def get_num2word(filename):
    """
    Get the number to word dict, stored in a list, index is the number.
    :param filename: file name
    :return: list, index is the number, value is the word
    """
    data = list()
    f = open(filename, 'r', encoding='utf-8')
    for line in f:
        line = line.strip().split()
        data.append(line[1])
    f.close()
    return data

def padding(batch_data, num2word, is_data):
    """
    Padding data until its length equal to the longest one in this batch
    :param batch_data: a batch of data
    :param num2word: list, the dict number to word
    :param is_data: True for training data, False for training label
    :return: np.array, the data after padding
    """
    if is_data:
        pad = num2word.index('<PAD>')
    else:
        pad = num2word.index('<EOS>')
    new_batch_data = list()
    length = 0
    for data in batch_data:
        length = len(data) if len(data) > length else length
    for data in batch_data:
        if len(data) < length:
            data = data + [pad]*(length-len(data))
        new_batch_data.append(data)
    return np.array(new_batch_data, dtype='int64')

if __name__=="__main__":
    # generate data set and dict
    training_data = get_data('./predata_10000/train_source_8000.txt')
    training_label = get_data('./predata_10000/train_target_8000.txt')
    source_num2word = get_num2word('./predata_10000/num2word_train_source_8000.txt')
    target_num2word = get_num2word('./predata_10000/num2word_train_target_8000.txt')

    # hyper parameter
    learning_rate = 1e-4
    decoder_learning_rate = 5e-4
    batch_size = 100
    epochs = 1000
    teaching_force_ratio = 0.5

    # generate NN
    encoder = Encoder(len(source_num2word), 600, 300).cuda()
    decoder = Decoder(len(target_num2word), 600, 300).cuda()
    if encoder.hidden_dim != decoder.hidden_dim:
        raise RuntimeError('Encoder and Decoder should have the same hidden dimension!')

    # Optimizer is SGD
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate,)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_learning_rate,)

    # Loss function is cross entropy
    criterion = nn.CrossEntropyLoss()

    # list to note down average loss of each epoch
    epoch_loss_list = list()

    # training part
    for epoch in range(epochs):
        i = 0
        flag = False
        epoch_loss = 0.0
        # loop for batch
        while 1:
            loss = 0.0
            # divided data set into batch
            # if the rest data less than a batch, take all of them
            if (i + 1)*batch_size >= len(training_data):
                train_data = training_data[i*batch_size:]
                train_label = training_label[i*batch_size:]
                flag = True
            # else, take a batch
            else:
                train_data = training_data[i*batch_size:(i + 1)*batch_size]
                train_label = training_label[i*batch_size:(i + 1)*batch_size]

            # padding
            train_data = padding(train_data, source_num2word, True)
            train_label = padding(train_label, target_num2word, False)

            max_length = train_label.shape[1]
            # transpose, (bs, seq_len) -> (seq_len, bs) because batch_first is false
            train_data = torch.LongTensor(train_data).transpose(0, 1).cuda()
            train_label = torch.LongTensor(train_label).transpose(0, 1).cuda()

            # forward of encoder
            encoder_out, (h_n, h_c) = encoder(train_data)
            # add the forward direction and backward direction to reduce dimension
            h_n = (h_n[0] + h_n[1]).unsqueeze(0)
            h_c = (h_c[0] + h_c[1]).unsqueeze(0)
            decoder_hidden = (h_n, h_c)

            # loop for each time step, feeding Decoder
            for time_step in range(max_length):
                # the first step, input '<BOS>' (which is also the first element of train_label)
                if time_step == 0:
                    begin_input = train_label[0].unsqueeze(0)
                    decoder_out, decoder_hidden = decoder(begin_input, decoder_hidden, encoder_out)
                # the rest time steps, using teacher forcing:
                else:
                    teacher_forcing = True if random.random() < teaching_force_ratio else False
                    if teacher_forcing:
                        time_step_input = train_label[time_step].unsqueeze(0)
                        decoder_out, decoder_hidden = decoder(time_step_input, decoder_hidden, encoder_out)
                    else:
                        _, time_step_input = torch.max(decoder_out, 2)
                        decoder_out, decoder_hidden = decoder(time_step_input, decoder_hidden, encoder_out)
                # for each time step, calculate loss
                loss += criterion(decoder_out.squeeze(), train_label[time_step])
            print(loss/(batch_size))
            # backward and update parameters
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            encoder_optimizer.step()
            decoder_optimizer.step()
            # calculate total loss for this epoch
            epoch_loss += loss.item()/max_length
            # When flag is True, the data is run out
            if flag:
                break
            i += 1
        # note down the loss of the batch
        epoch_loss_list.append(epoch_loss)
        print('*' * 10)
        print('epoch: {} of {}'.format(epoch + 1, epochs))
        print('average loss: {:.6f}'.format(epoch_loss))
        if epoch%5==0：
            # save the model
            torch.save(encoder, 'encoder.pkl')
            torch.save(decoder, 'decoder.pkl')

    # plot the variation of loss
    plt.plot(epoch_loss_list, label='training loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss.png', format='png')