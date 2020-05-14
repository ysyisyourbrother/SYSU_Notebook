# -*- coding: utf-8 -*
"""
Author: Junkai-Yan
File: test.py
Finished in 2019/12/18
This file test the model and evaluate it by using BLEU-4.
Firstly, read in testing data, developing data and the dict of target language.
Secondly, load in the model have saved in training part.
Thirdly, for test and dev data, convert the sentences to sequences and throw them into model.
    In this part, I throw it one by one, surely you can throw a batch, but it need padding.(best not)
    We must not use teaching forcing in testing part.
    The input to the next time step is the output of previous time step, using beam search but not greedy.
    Choose the best sentence, that is, the sentence which has the highest probability to be the finally output,
    note it down in a file.
    Calculate the BLEU-4 score between target and output, also note the score down in the file.
**********
What you should focus on is:
The whole process is similar to training process, the differences are:
    1. when training, we use teaching forcing, but testing, no!
    2. when training, we use greedy to be the next time step's input, but testing, we use beam search.
    3. when training, we calculate loss, but testing, we calculate BLEU-4 score.
"""

import torch
import numpy as np
from model import *
from nltk.translate.bleu_score import sentence_bleu

def get_data(filename):
    """
    Get data from the file.
    :param filename: file name
    :return: list, the data of the file
    """
    data = list()
    f = open(filename, 'r', encoding='utf-8')
    for line in f:
        line = line.strip().split()
        line = list(map(int, line))
        data.append(line)
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

def test(name, data_set, label_set, target_num2word, beam_value=3, max_length=60):
    """
    This function tests data using beam search, output the best one and it's BLEU-4
    score to a txt file.
    :param name: name of file to write
    :param data_set: testing data
    :param label_set: testing label, use it to calculate BLEU-4 score
    :param target_num2word: target language dictionary
    :param beam_value: how many branch beam search choose, default is 3
    :param max_length: max length of output, default is 60
    :return: None
    """
    f = open('./output_data_10000/'+name+'_output.txt', 'w')
    for i in range(len(data_set)):
        # get target sentence for ready to calculate BLEU-4
        target_sentence = [target_num2word[word] for word in label_set[i]]

        # get data and label
        data = torch.LongTensor(np.array(data_set[i], dtype='int64')).unsqueeze(0)
        label = torch.LongTensor(np.array(label_set[i], dtype='int64')).unsqueeze(0)

        # transpose, (bs, seq_len) -> (seq_len, bs) because batch_first is false
        data = torch.LongTensor(data).transpose(0, 1).cuda()
        label = torch.LongTensor(label).transpose(0, 1).cuda()

        # forward of encoder
        encoder_out, (h_n, h_c) = encoder(data)
        # add the forward direction and backward direction to reduce dimension
        h_n = (h_n[0] + h_n[1]).unsqueeze(0)
        h_c = (h_c[0] + h_c[1]).unsqueeze(0)
        decoder_hidden = (h_n, h_c)

        # list to note all the output and their probability for each word
        output_list = [['<BOS>'] for i in range(beam_value)]
        probability_list = [[1] for i in range(beam_value)]

        # loop for each time step, feeding Decoder
        for time_step in range(max_length):
            # the first step, input '<BOS>' (which is also the first element of label)
            if time_step == 0:
                begin_input = label[0].unsqueeze(0)
                decoder_out, decoder_hidden = decoder(begin_input, decoder_hidden, encoder_out)
            # the second time step, beam search:
            elif time_step == 1:
                values, indices = torch.topk(decoder_out, beam_value)
                indices = indices.squeeze(0).transpose(0, 1)
                for k in range(beam_value):
                    output_list[k].append(target_num2word[indices[k][0].item()])
                    probability_list[k].append(values[0][0][k].item())
                decoder_out, decoder_hidden = decoder(indices, decoder_hidden, encoder_out)
            # the rest time steps, greedy search:
            else:
                values, indices = torch.max(decoder_out, 2)
                for k in range(beam_value):
                    output_list[k].append(target_num2word[indices[k][0].item()])
                    probability_list[k].append(values[k][0].item())
                decoder_out, decoder_hidden = decoder(indices, decoder_hidden, encoder_out)

        # choose the best output
        best_output = list()
        best_pro = 0
        # traverse all the possible output and update best one
        for k in range(beam_value):
            end = output_list[k].index('<EOS>') if '<EOS>' in output_list[k] else -1
            total_pro = sum(probability_list[k][0:end+1])/(end+1) if end != -1 else sum(probability_list[k])/max_length
            if total_pro > best_pro:
                best_pro = total_pro
                best_output = output_list[k][0:end+1] if end != -1 else output_list[k]

        # write in to a file
        f.write('Predict sentence: '+' '.join(best_output)+'\n')
        f.write('BLEU-4 score:'+str(sentence_bleu(target_sentence, best_output))+'\n\n')
    f.close()


if __name__=="__main__":
    # get testing data, label
    testing_data = get_data('./predata_10000/test_source_1000.txt')
    testing_label = get_data('./predata_10000/test_target_1000.txt')

    # get developing data, label
    dev_data = get_data('./predata_10000/dev_source_1000.txt')
    dev_label = get_data('./predata_10000/dev_target_1000.txt')

    # get target language dictionary
    t_num2word = get_num2word('./predata_10000/num2word_train_target_8000.txt')

    # load in model have trained, convert it to eval
    encoder = torch.load('encoder.pkl').cuda().eval()
    decoder = torch.load('decoder.pkl').cuda().eval()

    # test for testing set and developing set
    test('test', testing_data, testing_label, t_num2word, 3, 80)
    test('dev', dev_data, dev_label, t_num2word, 3, 80)