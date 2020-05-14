# encoding=utf-8

import jieba
import os
Path = os.path.abspath('.')

def process_train(filename,lag):
    word_dic = {'<BOS>': 0, '<EOS>': 1, '<UKN>': 2, '<PAD>': 3}
    with open('./dataset_10000/'+filename, 'r', encoding='utf-8') as dataset:
        with open('./preprocessing/' + filename, 'w', encoding='utf-8') as data_encoded:
            n = 4
            for line in dataset.readlines():
                if lag == 0:
                    sentence = jieba.lcut(line.strip())
                elif lag == 1:
                    sentence = jieba.lcut(line.strip().lower())  # 如果是英文的话就要都改成小写
                sentence = '<BOS> ' + " ".join(sentence) + ' <EOS>'
                sentence = sentence.split()
                # 构建训练集上的字典
                for word in sentence:
                    if word not in word_dic:
                        word_dic[word] = n
                        n += 1
                    data_encoded.write(str(word_dic[word])+' ')
                data_encoded.write('\n')
    with open('./preprocessing/word_dic_' + filename, 'w', encoding='utf-8') as num2word_file:
        for word in word_dic:
            num2word_file.write(str(word_dic[word]) + ' ' + word + '\n')
    return word_dic


def process_test(filename, word_dic,lag):
    with open('./dataset_10000/'+filename, 'r', encoding='utf-8') as dataset:
        with open('preprocessing/' + filename, 'w', encoding='utf-8') as data_encoded:
            for line in dataset.readlines():
                if lag==0:
                    sentence = jieba.lcut(line.strip())
                elif lag==1:
                    sentence = jieba.lcut(line.strip().lower())
                sentence = '<BOS> ' + " ".join(sentence) + ' <EOS>'
                sentence = sentence.split()
                for word in sentence:
                    if word not in word_dic:
                        data_encoded.write(str(word_dic['<UKN>']) + ' ')  # 如果词语在训练集上没有出现，就直接删掉
                    else:
                        data_encoded.write(str(word_dic[word]) + ' ')
                data_encoded.write('\n')


if __name__ == "__main__":
    ch_word_dic = process_train('train_source_8000.txt',0)
    en_word_dic = process_train('train_target_8000.txt',1)
    process_test('test_source_1000.txt', ch_word_dic,0) # 将训练集上生成的词典传到测试集上
    process_test('test_target_1000.txt', en_word_dic,1)

