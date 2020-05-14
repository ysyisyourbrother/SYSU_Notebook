import numpy as np
import jieba
import re

Train_File= './train_data/{}.txt'
Test_File = './test/questions.txt'
Answer_Path = './test/answer.txt'
StopWord_Path = 'stopword.txt'
Word_Table_Path = './train_data/word_table.txt'

class build_trainData:
    def __init__(self, seq_len):
        self.seq_len = seq_len  # 一个句子的最大长度
        self.input = []  # 网络的输入
        self.output = []  # 网络的输出
        self.length = []  # 输入向量的实际长度
        self.mask = []  # 长度跟输入一样，有实际输入的位置置一，padding的位置上置零
        self.cur = 0  # 开始读取的位置
        self.train_num = 0  # 训练数据即句子的数量

        self.build_vocab()
        self.read_file()  # 将训练文本转成向量

    def build_vocab(self):
        # 建立词到id的映射
        self.word2Id = {}
        self.Id2word = {}
        with open(Word_Table_Path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.split()
                self.word2Id[line[1]] = int(line[0])+1 # 下标从0开始
        self.word2Id['???'] = 0
        for key, value in self.word2Id.items():
            self.Id2word[value] = key

        return self.word2Id,self.Id2word


    def read_file(self):
        self.inputdata=[]
        self.outputdata=[]
        self.mask=[]
        for i in range(1, 1000):
            with open(Train_File.format(i), 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.split()
                    ##### 将句子每个词语用它在词表中位置
                    sen=[]
                    for word in line:
                        if word in self.word2Id.keys():
                            sen.append(self.word2Id[word])
                        else:
                            sen.append(0)

                    if len(sen) == 1 or len(sen) == 0:  # 句子只有一个词或者是空的句子
                        continue
                    elif  len(sen) < self.seq_len:
                        self.length.append(len(sen) - 1)  # 记录输入的时候的句子的真实长度
                        sen += [0] * (self.seq_len -  len(sen))
                    else:
                        sen = sen[:self.seq_len] # 截断
                        self.length.append(self.seq_len - 1)
                    self.inputdata.append(sen[: -1])  # 句子的前l-1个词为输入
                    self.outputdata.append(sen[1: ])  # 句子的后l-1个词为输出

        self.train_num = len(self.inputdata)
        for i in range(self.train_num):
            t = [1] * self.length[i] + [0] * (self.seq_len - 1 - self.length[i])
            self.mask.append(t)

    def next_batch(self, batch_size):
        '''
        :param batch_size: 批量读取大小
        :return: 批量数据
        '''
        end = (self.cur + batch_size) % self.train_num  # 结束的位置
        if self.cur > end:
            input_batch = self.inputdata[self.cur:] + self.inputdata[: end]
            output_batch = self.outputdata[self.cur:] + self.outputdata[: end]
            length_batch = self.length[self.cur:] + self.length[: end]
            mask_batch = self.mask[self.cur:] + self.mask[: end]
        else:
            input_batch = self.inputdata[self.cur: end]
            output_batch = self.outputdata[self.cur: end]
            length_batch = self.length[self.cur: end]
            mask_batch = self.mask[self.cur: end]
        self.cur = end  # 更新下次读取的位置
        return input_batch, output_batch, length_batch, mask_batch


class build_testData:
    def __init__(self,seq_len,word2Id,Id2word):
        self.input = []                                 # 网络的输出
        self.length = []                                # 句向量实际长度
        self.seq_len = seq_len                        # 一个句子的最大长度
        self.test_num = 0                               # 测试数据即句子的数量

        self.word2Id = word2Id
        self.Id2word = Id2word
        self.stop_word=[]
        with open(StopWord_Path, 'r', encoding='utf-8') as f:
            for line in f:
                self.stop_word.append(line.strip())
        self.readfile(seq_len)  # 将测试文本转成向量
        self.get_test_answer()

    def readfile(self, seq_len):
        jieba.add_word('MASK') # 将MASk添加到词表
        with open(Test_File, 'r', encoding='utf-8') as f:
            for line in f:
                # 分词、定位MASK位置
                sentence = jieba.lcut(line)
                mask_pos = sentence.index('MASK')
                sentence = sentence[: mask_pos]

                # 去非中文、去停止词
                for i in range(mask_pos):
                    sentence[i] = re.sub('[^\u4e00-\u9fa5]', '', sentence[i])
                while '' in sentence:
                    sentence.remove('')

                ##### 去停用词 将句子每个词语用它在词表中位置
                sen = []
                for word in sentence:
                    if word in self.stop_word:
                        if word in self.word2Id.keys():
                            sen.append(self.word2Id[word])
                        else:
                            sen.append(0)
                l = len(sen)
                if l > seq_len:
                    sen = sen[-seq_len:]
                    self.length.append(seq_len)
                else:
                    sen += [0] * (seq_len - l)
                    self.length.append(l)

                self.input.append(sen)

        self.test_num = len(self.length)

    def get_test_data(self):
        return self.input, self.length

    def get_test_answer(self):
        # 加载答案
        self.Answer=[]
        with open(Answer_Path, 'r', encoding='utf-8') as f:
            for line in f:
                self.Answer.append(line.split()[0])

