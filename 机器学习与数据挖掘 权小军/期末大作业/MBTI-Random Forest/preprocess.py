import pandas as pd
import numpy as np
import re
import csv

# plotting
import seaborn as sns
import matplotlib.pyplot as plt



# 将用户特征标签数据处理成由01组成的向量
# 如[0, 1, 0, 1]代表ISFP
def translate_personality(personality):
    # transform mbti to binary vector
    b_Pers = {'I': '0', 'E': '1', 'N': '0', 'S': '1', 'F': '0', 'T': '1', 'J': '0', 'P': '1'}
    return [b_Pers[l] for l in personality]

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
nltk.download('wordnet')


def pre_process_data(data):
    # Lemmatize
    stemmer = PorterStemmer()
    lemmatiser = WordNetLemmatizer()

    # 获取英文停用词列表
    cachedStopWords = stopwords.words("english")

    # 清除文章内和type相关的数据
    capital_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
                        'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
    # 大写和小写数据都进行处理
    lower_type_list = [x.lower() for x in capital_type_list]

    list_personality = []
    list_posts = []
    len_data = len(data)
    i = 0

    for row in data.iterrows():
        i += 1
        if (i % 500 == 0 or i == 1 or i == len_data):
            print("%s of %s rows" % (i, len_data))

        # 移除链接和无用注释等信息
        posts = row[1].posts
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)
        temp = re.sub("[^a-zA-Z]", " ", temp)
        temp = re.sub(' +', ' ', temp).lower()

        # 去除停用词
        temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])

        # 去除type相关特征数据
        for index in range(len(capital_type_list)):
            temp = temp.replace(lower_type_list[index], "")

        # 将特征标签提取出来
        type_labelized = translate_personality(row[1].type)
        list_personality.append(type_labelized)
        list_posts.append(temp)


    with open("./data/preprocessing.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["type", "posts"])
        for i in range(len(list_posts)):
            writer.writerow([" ".join(list_personality[i]), list_posts[i]])

    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)

    return list_posts, list_personality


def main():
    # read data
    data = pd.read_csv('./data/mbti_1.csv')

    # 数据预处理
    pre_process_data(data)


if __name__ == "__main__":
    main()