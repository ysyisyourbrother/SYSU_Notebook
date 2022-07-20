from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.manifold import TSNE

import pandas as pd
import numpy as np

# 将预处理的数据读取出来变成nparray
def read_preprocessing():
    data = pd.read_csv("./data/preprocessing.csv")
    list_posts = data['posts']
    list_personality = [list(map(int, i.split())) for i in data['type']]
    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality


# 用TFIDF的方法将用户post数据变成向量
def tfidf():
    # 从预处理数据中读取 并组成nparray
    list_posts, list_personality = read_preprocessing()

    # Posts to a matrix of token counts
    cntizer = CountVectorizer(analyzer="word",
                                 max_features=2000,  # 限制词典维度上限
                                 max_df=0.8,  # 在0.7以上的文章出现过的词语可以去除
                                 min_df=0.05) # 在0.05以下的文章出现过的词语可以去除

    # 计算词语频率
    print("CountVectorizer...")
    word_count = cntizer.fit_transform(list_posts)
    print(word_count)


    print("Tf-idf...")
    # 根据词语频率得到tfidf矩阵
    tfizer = TfidfTransformer()
    tfidf =  tfizer.fit_transform(word_count).toarray()
    print(tfidf, len(tfidf),len(tfidf[0]))

    return tfidf, list_personality


if __name__ == "__main__":
    tfidf()