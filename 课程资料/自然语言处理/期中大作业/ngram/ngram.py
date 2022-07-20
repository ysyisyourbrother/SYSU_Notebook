import re
import jieba
import os

N = 3
word_dict = {}
word_dict_2 = {}
word_dict_3 = {}

# 二元词组，三元词组, 多处用到，传参太麻烦了，直接拿到外面来
for index in range(1, 1001):
    # 工作路径和代码路径可能不同，最好不用相对路径
    # cur_path = os.getcwd()
    # file_path = cur_path + r'./train/' + str(index) + '.txt'
    with open('./train1/' + str(index) + '.txt', 'r', encoding='utf-8-sig') as f:
        for sentence in f.readlines():
            words = sentence.strip().split(" ")
            words.insert(0,  "<S>")
            words.append("<E>")
            if len(words) >= 3:
                # 统计词组出现频率
                for i in range(len(words)-1):
                    key = (words[i], words[i+1])
                    word_dict_2[key] = word_dict_2.get(key, 0) + 1

                for j in range(len(words)-2):
                    key = (words[j], words[j+1], words[j+2])
                    word_dict_3[key] = word_dict_3.get(key, 0) + 1

len2 = len(word_dict_2.keys())
len3 = len(word_dict_3.keys())
print(len2, len3)


def read_data(filename):
    """
    读取当前路径下的 filename 文件，要求每一行是单独的数据，不会做其他处理
    :param filename: str, 文件名
    :return:
    """
    path = os.getcwd() + '/' + filename
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(line.strip())
    return data


def cal_prob(w1, w2, w3):
    """
    N = 3 时，计算该句子的概率, 使用加1法平滑数据
    :param w1: str
    :param w2: str
    :param w3: str
    :return: w1,w2出现的情况下同时出现w1，w2，w3的概率
    """
    p1 = word_dict_2.get((w1, w2), 0) + len(word_dict_2.keys())
    p2 = word_dict_3.get((w1, w2, w3), 0) + 1
    return float(p2) / float(p1)


def get_related_words(pre_cut, suf_cut):
    """
    取出 [MASK]附近的几个词
    :param pre_cut: list，原句子mask前的部分分词结果
    :param suf_cut: list，原句子mask后的部分分词结果
    :return: [MASK]附近的几个词，[w1,w2,MASK,w3,w4]
    """
    related_word = []
    if len(pre_cut) > N - 1:
        for j in range(N - 2, -1, -1):
            related_word.append(pre_cut[-1 - j])
    else:
        related_word += pre_cut
    related_word.append('MASK')
    if len(suf_cut) > N - 1:
        for j in range(N - 1):
            related_word.append(suf_cut[j])
    else:
        related_word += suf_cut
    return related_word


def precondition(sentence):
    """
    预处理，去数字、英文、符号，分词，返回MASK前后句子分词结果
    :param sentence: str，原句子
    :return:
    """
    # 注意要包含空格
    punctuation = "A-Za-z0-9;,：.:＃＄％＆＇（）＊＋，－：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠" \
                  "｢｣､、〃《》「」『』【】 〔〕〖〗〘〙〚〛〜〝〞–—‘’‛“”„‟…‧﹏\r。！!?？｡\n"
    pos = sentence.find('[MASK]')
    # mask 前后的词语
    pre = sentence[:pos]
    suf = sentence[pos + 6:]
    # 去数字，符号，停止词，划分词语
    pre = re.sub(r'[{}]+'.format(punctuation), '', pre)
    pre_cut = list(jieba.cut(pre, cut_all=False))
    suf = re.sub(r'[{}]+'.format(punctuation), '', suf)
    suf_cut = list(jieba.cut(suf, cut_all=False))
    for word in pre_cut[::-1]:
        if word in stop_words:
            pre_cut.remove(word)
    for word in suf_cut[::-1]:
        if word in stop_words:
            suf_cut.remove(word)
    return pre_cut, suf_cut


def predict(related_word):
    """
    根据 MASK附近的词来预测MASK
    :param related_word: list，MASK附近的词
    :return: 概率最大的词以及概率
    """
    max_prob = 0.0
    predict_word = ''
    for word in word_dict.keys():
        index_of_mask = related_word.index('MASK')
        related_word[index_of_mask] = word
        # print(related_word)
        prob = 1.0
        for j in range(len(related_word) - 2):
            prob = prob * cal_prob(related_word[j], related_word[j + 1], related_word[j + 2])
        if prob > max_prob:
            max_prob = prob
            predict_word = word
        related_word[index_of_mask] = 'MASK'
    return max_prob, predict_word


if __name__ == '__main__':
    # 读取问题句子和答案以及停止词，位于同一路径下
    questions = read_data('./test/questions.txt')
    answer = read_data('./test/answer.txt')
    stop_words = read_data('停止词.txt')

    # 读取单个词出现的频率，因为之前写入了文件，直接读取
    with open('./train1/word_table.txt', 'r', encoding='utf-8-sig') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            word_dict[line[1]] = int(line[0])

    accuracy = 0
    predictions = []
    for i in range(len(questions)):
        sentence = questions[i]
        # 预处理，去数字、英文、符号，分词
        pre_cut, suf_cut = precondition(sentence)
        # 取出该词附近的词语 [w1, w2, [MASK], w3, w4]
        related_word = get_related_words(pre_cut, suf_cut)
        # 预测，得到最合适的词，以及概率
        max_prob, predict_word = predict(related_word)
        predictions.append(predict_word)
        if predict_word == answer[i]:
            accuracy += 1
            print(f'第{i+1}个句子:')
            print(questions[i])
            print(max_prob, predict_word, '\n')

    print(accuracy)
    with open('prediction.txt', 'w', encoding='utf-8') as f:
        for prediction in predictions:
            f.write(prediction + '\n')
