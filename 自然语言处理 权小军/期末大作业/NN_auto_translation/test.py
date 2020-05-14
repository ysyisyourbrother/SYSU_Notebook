# -*- coding: utf-8 -*
BOS=0
EOS=1
UKN=2
PAD=3

import torch
import numpy as np
from model import *
from nltk.translate.bleu_score import sentence_bleu

def get_data(filename):
    data = list()
    f = open(filename, 'r', encoding='utf-8')
    for line in f:
        line = line.strip().split()
        line = list(map(int, line))
        data.append(line)
    f.close()
    return data

def get_worddic(filename):
    word_dic = {}
    f = open(filename, 'r', encoding='utf-8')
    for line in f:
        line = line.strip().split()
        word_dic[int(line[0])] = line[1]
    f.close()
    return word_dic

def test(beam_value=3, max_length=60):
    # 读取预处理好的数据集
    data_set = get_data('./preprocessing/test_source_1000.txt')
    label_set = get_data('./preprocessing/test_target_1000.txt')

    # 读取词语字典
    word_dic = get_worddic('./preprocessing/word_dic_train_target_8000.txt')

    # 读取模型的训练结果
    encoder = torch.load('encoder.pkl').cuda().eval()
    decoder = torch.load('decoder.pkl').cuda().eval()

    f = open('./output_data_10000/test_output.txt', 'w',encoding="utf-8")
    for i in range(len(data_set)):
        # 一句话一句话进行翻译
        input_data = torch.LongTensor(np.array(data_set[i], dtype='int64')).unsqueeze(0)
        label = torch.LongTensor(np.array(label_set[i], dtype='int64')).unsqueeze(0)

        # (batchsize, seq_len) -> (seq_len, batchsize) 将bs转到另外的维度
        input_data = torch.LongTensor(input_data).transpose(0, 1).cuda()
        label = torch.LongTensor(label).transpose(0, 1).cuda()

        # 对输入句子编码
        encoder_out, (h_n, h_c) = encoder(input_data)
        decoder_init = (h_n.unsqueeze(0), h_c.unsqueeze(0))

        # 定义收集不同beam的列表，初始化bos和概率1，因为bos出现概率为1
        output_list = [[BOS] for i in range(beam_value)]
        probability_list = [[1] for i in range(beam_value)]


        # 第一步先输入bos <1 bs >
        first_input = torch.full(size=[1,1],fill_value=BOS).long().cuda()
        decoder_out, decoder_hidden = decoder(first_input, decoder_init, encoder_out)
        # 计算第一步beam输出
        values, word_index = torch.topk(decoder_out, beam_value)  # 取出beam个最大值
        # 遍历每个beam 保存搜索结果
        for k in range(beam_value):
            output_list[k].append(word_index[0][0][k].item())
            probability_list[k].append(values[0][0][k].item())

        decoder_out_beam = [[] for _ in range(beam_value)]
        decoder_hidden_beam = [[] for _ in range(beam_value)]
        for k in range(beam_value):
            word_input = torch.tensor([[output_list[k][-1]]]).long().cuda()
            decoder_out_beam[k], decoder_hidden_beam[k] = decoder(word_input, decoder_hidden, encoder_out)


        for k in range(beam_value):
            for step in range(2,max_length):
                # 后面的用贪心算法，最后选概率最高的
                values, word_index = torch.max(decoder_out_beam[k], 2)
                output_list[k].append(word_index[0][0].item())
                probability_list[k].append(values[0][0].item())
                decoder_out_beam[k], decoder_hidden_beam[k] = decoder(word_index, decoder_hidden_beam[k], encoder_out)

        # 将它换成字符串  并统计概率最高的一句话
        best_sen=0
        probability_max=-1
        for k in range(beam_value):
            probability_=0
            for index,word in enumerate(output_list[k]):
                output_list[k][index] = word_dic.get(word,'<UKN>')
                probability_+=probability_list[k][index]  # 计算概率和
                if word==EOS:  # 如果翻译到了句子结尾就直接退出
                    output_list[k]=output_list[k][:index+1]
                    break
            if probability_>probability_max:
                best_sen=k
                probability_max=probability_
        best_output=output_list[best_sen]

        target_sentence=[]
        for word in label_set[i]:
            target_sentence.append(word_dic.get(word,'<UKN>'))
        print("正确的句子为："+' '.join(target_sentence))
        print('预测出的句子为 '+' '.join(best_output))
        print('BLEU-4 评分为:'+str(sentence_bleu(target_sentence, best_output))+'\n\n')
        f.write("正确的句子为："+' '.join(target_sentence)+'\n')
        f.write('预测出的句子为 '+' '.join(best_output)+'\n')
        f.write('BLEU-4 评分为:'+str(sentence_bleu(target_sentence, best_output))+'\n\n')
    f.close()


if __name__=="__main__":
    test(3, 80)