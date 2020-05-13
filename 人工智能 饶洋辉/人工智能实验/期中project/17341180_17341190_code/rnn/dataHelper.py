import numpy as np
import re
import os
import random

glove_path = os.path.dirname(__file__)

class HelperClass():
    def __init__(self,filename='',step=0,word_dim=0,embedding_exist={}):
        self.file_name=filename


        self.embeddings_index = embedding_exist  # 创建一个空的字典用来装词向量
        self.paddingSequence=step # lstm允许最长句子长度
        self.word_dim = word_dim # glove选取的词向量长度

        self.pad_word = [0] * self.word_dim  # 用0来padding

        self.len_pairs1=[] # 原来句子长度
        self.len_pairs2=[] # 原来句子长度
        self.padding_sen1 = []  # 记录padding后的句子对
        self.padding_sen2 = []  # 记录padding后的句子对

        self.similarity_list = [] # 装了句子相似度的列表

        self.cursor=0 # batch输入的时候的指针

        self.TFRecord_path = './data/train.tfrecords'

    def initial(self):
        # 初始化 读取数据并生成对应格式列表
        if len(self.embeddings_index)==0:
            self.convert_embedding_file()
        self.readdata()
        self.washData()
        self.combineGlove()
        self.padding()



    def readdata(self):
        '''
        读取训练的txt文件
        '''
        f = open("./stsbenchmark/{}.txt".format(self.file_name),encoding='utf-8')
        self.sentence_pair=[]
        for line in f.readlines():
            sentences = line.split(sep='\t')[4:7] # 获取句子和相似度 用tab分割
            # print(sentences)
            # os.system("pause")
            sentences[2]=sentences[2].strip()
            sentences[1]=sentences[1].strip()
            self.similarity_list.append(float(sentences[0].strip()))
            self.sentence_pair.append(sentences[1:])
        # print(len(self.sentence_pair))

    def washData(self):
        '''
        清洗句子并去除标点符号并分词
        :return:
        '''
        for pairs in self.sentence_pair:
            # 只保留中文、大小写字母和阿拉伯数字
            reg = '''[!"#$%&'"()*+,-./:;<=>?@`’“]'''

            for i in range(len(pairs)):
                tem = [re.sub(reg, '', word).lower() for word in pairs[i].split()]
                pairs[i]=tem
        # print(self.sentence_pair)

    def combineGlove(self):
        '''
        将词语序列和Glove的词语向量结合
        删除掉glove词向量模型中不存在的词语
        :return:
        '''
        for i1,pairs in enumerate(self.sentence_pair[:]):
            for i2 in range(2):
                for i3,word in enumerate(pairs[i2]):
                    # 遍历每一个词语
                    self.sentence_pair[i1][i2][i3] = self.embeddings_index.get(word, None)
                self.sentence_pair[i1][i2] = list(filter(None, self.sentence_pair[i1][i2])) # 过滤掉None的元素

    def padding(self):
        '''
        para:
            sentence_pairs:输入一个列表，列表的每一个元素是一个二元组；二元组的元素是一个句子的单词组成的列表
            paddingSequence: 处理后的统一长度;

        return:
            len_pairs:原句子的长度，整数二元组的列表
            padding_pairs:padding后的句子，字符串二元组的列表
        '''

        for pair in self.sentence_pair:
            wordNum_a, wordNum_b = len(pair[0]), len(pair[1])
            if wordNum_a>self.paddingSequence:
                self.len_pairs1.append(self.paddingSequence)
            else:
                self.len_pairs1.append(wordNum_a)

            if wordNum_b>self.paddingSequence:
                self.len_pairs2.append(self.paddingSequence)
            else:
                self.len_pairs2.append(wordNum_b)

            list_a, list_b = pair[0], pair[1]
            if wordNum_a <= self.paddingSequence:
                for i in range(self.paddingSequence - wordNum_a):
                    list_a.append(self.pad_word)
            else:
                for i in range(wordNum_a - self.paddingSequence):
                    list_a.pop()

            if wordNum_b <= self.paddingSequence:
                for i in range(self.paddingSequence - wordNum_b):
                    list_b.append(self.pad_word)
            else:
                for i in range(wordNum_b - self.paddingSequence):
                    list_b.pop()
            self.padding_sen1.append(list_a)
            self.padding_sen2.append((list_b))


    def convert_embedding_file(self):
        '''
            读取glove预训练词向量组成字典
        :return:
        '''
        # 词向量中，第一个是单词，后面的是一个按照空格分割的300维度的向量。
        embedding_file = glove_path+ "/glove/glove.6B.{}d.txt".format(self.word_dim)  # 把训练好的词向量全都变成字典的形式来进行存储。
        rf = open(embedding_file, 'r', encoding='utf-8')  # 打开词向量文件。
        print("reading embedding from " + embedding_file)  # 下面开始读取其中的文件内容。
        count = 0
        for line in rf:
            count += 1
            if count % 100000 == 0:  # 这里用来统计词嵌入矩阵的单词的数量。
                print(str(count))

            values = line.split()  # 每次分割300个元素的内容。  第一个存储的是的单词。意思就是每次只能够读取其中的一行的单词。
            index = len(values) - self.word_dim  # 一般是301-300 = 1,  所以一般的情况下是1.
            if len(values) > (self.word_dim + 1):  # 对于一些词嵌入的单词可能使得不只是一个单词组成，所以用下面的情况进行判断。
                word = ""  # 一个空的字符串。                 # 例如由  Bill Gates组成，，所以单词的长度会超过一般的情况。
                for i in range(len(values) - self.word_dim):
                    word += values[i] + " "
                word = word.strip()  # 去除前后的空格。    # 去除前后的空格的部分内容。
            else:
                word = values[0]  # 单词是values[0]的形式。

            # coefs = np.asarray(values[index:], dtype='float32')  # 作为矩阵存储其中的元素。  一个向量。从索引值后面的内容都是词向量的数字。
            self.embeddings_index[word] = [float(tem) for tem in values[index:]] # 把对应的内容用字典存储起来。
        rf.close()  # 关闭文件。
        print("finish.")  # 完成操作。

    def next_batch(self,batch_size):
        '''
        返回一批数据
        :param batch_size: 一批数据的大小
        '''
        end=self.cursor+batch_size
        if end>len(self.padding_sen1):
            end %= len(self.padding_sen1)
            batch_xs1 = self.padding_sen1[self.cursor:]+self.padding_sen1[:end]
            batch_xs2 = self.padding_sen2[self.cursor:]+self.padding_sen2[:end]
            batch_ys = self.similarity_list[self.cursor:]+self.similarity_list[:end]
            mask_xs1 = self.len_pairs1[self.cursor:]+self.len_pairs1[:end]
            mask_xs2 = self.len_pairs2[self.cursor:]+self.len_pairs2[:end]
            self.cursor = end
            return 0, batch_xs1, batch_xs2, batch_ys, mask_xs1, mask_xs2
        else:
            batch_xs1=self.padding_sen1[self.cursor:end]
            batch_xs2=self.padding_sen2[self.cursor:end]
            batch_ys = self.similarity_list[self.cursor:end]
            mask_xs1=self.len_pairs1[self.cursor:end]
            mask_xs2=self.len_pairs2[self.cursor:end]
            self.cursor = end
            return 1,batch_xs1,batch_xs2,batch_ys,mask_xs1,mask_xs2

    def random_batch(self):
        random.choices(self.padding_sen1)

    def write(self,filename):
        '''
        para:
            simi_list:句子对的相似度组成的列表；
            len1_list:所有句子a的长度组成的列表；
            len2_list:所有句子b的长度组成的列表；
            sent1_list:所有句子a的分词映射结果，一个三维向量，第一维表示所有句子a，第二维表示一个句子，第三维表示单词映射
            sent2_list:同上

        return:
            无。写入文件
        '''
        File = open('./data/'+filename, 'w', encoding='utf-8')

        samp_count = len(self.similarity_list)
        for i in range(samp_count):
            line = str(self.similarity_list[i]) + '\t' + str(self.len_pairs1[i]) + '\t' + str(self.len_pairs2[i]) + '\t' \
                   + str(self.padding_sen1[i]) + '\t' + str(self.padding_sen2[i])
            File.write(line)
            if i != samp_count - 1:
                File.write('\n')
        File.close()

    def str2list(self,s):
        res = []

        each_word, each_dim_num = [], ''
        for i in range(1, len(s) - 1):
            if (s[i] >= '0' and s[i] <= '9') or s[i] == '.':
                each_dim_num += s[i]
            elif s[i] == ',' and s[i - 1] != ']':
                each_word.append(float(each_dim_num))
                each_dim_num = ''
            elif s[i] == ']':
                each_word.append(float(each_dim_num))
                each_dim_num = ''
                res.append(each_word)
                each_word = []
        return res

    def read(self,filename):
        '''
        读文件，按格式读取训练集txt
        return:
            simi_list:句子对的相似度组成的列表；
            len1_list:所有句子a的长度组成的列表；
            len2_list:所有句子b的长度组成的列表；
            sent1_list:所有句子a的分词映射结果，一个三维向量，第一维表示所有句子a，第二维表示一个句子，第三维表示单词映射
            sent2_list:同上
        '''
        File = open('./data/'+filename, 'r', encoding='utf-8')

        simi_list, len1_list, len2_list, sent1_list, sent2_list = [], [], [], [], []

        for i,line in enumerate(File.readlines()):
            print(i)
            line = line.strip().split('\t')
            # print(line[0])
            self.similarity_list.append(float(line[0]))
            self.len_pairs1.append(float(line[1]))
            self.len_pairs2.append(float(line[2]))

            self.padding_sen1.append(self.str2list(line[3]))
            self.padding_sen2.append(self.str2list(line[4]))

        File.close()


if __name__=='__main__':
    train_input = HelperClass('sts-train',25,50)
    train_input.initial()
    _,batch_xs1, batch_xs2, batch_ys, mask_xs1, mask_xs2=train_input.next_batch(32)



