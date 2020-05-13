import math
import pandas as pd
import heapq
import time
import numpy as np
import matplotlib.pyplot as plt
import csv

import sys
sys.path.append('../TFIDF')
import TFIDF as T

def read_file(file_name,usecols=None):
    '''
    读取csv文件数据特定的列
    :return:返回读取的csv文件
    '''
    df = pd.read_csv('../data/lab1_data/classification_dataset/'+file_name,usecols=usecols)
    return df

def count_words(df):
    '''
    输入所有文本组成的列表
    按照出现先后收集所有的词语并组成列表返回
    :param data:
    :return: 返回词语按先后顺序形成的列表
    '''
    word_set=set()
    word_list=[]
    for i in range(len(df)):
        record = df.iloc[i][0].split()
        for word in record:
            if word not in word_set: # 利用集合来查询速度更快，再按照先后顺序加入到列表中
                word_list.append(word)
                word_set.add(word)
    return word_list


def onehot(df,word_list):
    '''
    构建每个语段的onehot向量
    :return:
    '''
    onehot_met=np.zeros(shape=(len(df),len(word_list)))
    for index in range(len(df)):
        record = df.iloc[index][0].split()
        for i,word in enumerate(word_list):
            if word in record:
                onehot_met[index][i]=1
    return onehot_met

def TFIDF(df,word_list):
    '''
    调用前面写的tfidf算法来训练矩阵
    :return: 返回TFIDF矩阵
    '''
    df = read_file("classification_simple_test.csv",[0])
    data = [a[0].split() for a in np.array(df).tolist()]
    IDF_list = T.IDF(data, word_list)  # 计算idf值
    TFIDF_met=np.empty(shape=(len(data),len(word_list)))
    for i,record in enumerate(data):
        TF_dic = T.TF(record)
        for index, word in enumerate(word_list):
            TFIDF_met[i][index]= TF_dic.get(word, 0) * IDF_list[index]
    return TFIDF_met

def train():
    '''
    读取训练集数据并训练出onehot矩阵
    :return: 返回训练出来的onehot矩阵和词语的列表
    '''
    df = read_file("train_set2.csv")
    word_list = count_words(df)
    return TFIDF(df, word_list),word_list,df

def cal_distance(vec1,vec2,cos=False,n=2):
    '''
    计算两个向量之间的距离，默认为欧式距离，可以调整改变阶数, cos为True是计算余弦相似度
    :return:返回一个距离值
    '''
    if not cos:
        return np.sum((vec1-vec2)**2)**0.5
    else:
        num = np.inner(vec1,vec2)   # 若为行向量则 A * B.T
        denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if denom==0:
            denom=0.001
        cos = num / denom  # 余弦值
        return 1-cos

def KNN_predict(k,word_list,train_met,test_met,train_df):
    '''
    读取验证集上的数据并用KNN模型计算预测结果
    :return: 返回模型预测出来的测试集上的结果
    '''
    predict_tag=[] # 预测出来的情感标签值
    for i,record in enumerate(test_met):
        dis_vec=[]# 记录当前预料和所有训练后的语料的距离，并选取前k个最近的语料
        for i,train_record in enumerate(train_met):
            dis = cal_distance(record,train_record,cos=False,n=2)
            dis_vec.append((dis,i))
        heapq.heapify(dis_vec)
        k_close = heapq.nsmallest(k,dis_vec)# 利用最小堆的方法选取前k个最小元素
        tag_list=[]
        for r in k_close:
            tag_list.append(train_df.iloc[r[1]][1])
        if len(tag_list)>0:
            # 防止不存在标签
            predict_tag.append(max(tag_list,key=tag_list.count)) # 取众数计算
        else:
            predict_tag.append("null") # 若不存在标签则加入null
    return predict_tag

def test_set_predict(k,res_filename,word_list,met,train_df):
    '''
    计算在测试集上的表现，并且输出结果到csv表上
    :return:
    '''
    predict_tag= KNN_predict(k, word_list, met[:-1],met[-1:], train_df)
    print(predict_tag)
    csvFile = open(res_filename, "w")
    writer = csv.writer(csvFile)
    # 先写入columns_name
    writer.writerow(["textid", "label"])  # 写入列的名称
    # 写入多行用writerows
    for i in range(len(predict_tag)):
        res = [i+1,predict_tag[i]]
        writer.writerow(res)
    csvFile.close()

def cal_accuracy(predict_tag,validation_df):
    '''
    根据模型预测的标签结果和在验证集上真实结果比较并计算精确度
    :return: 返回精确度
    '''
    total=len(validation_df)
    count=0
    for i in range(total):
        if predict_tag[i]==validation_df.iloc[i][1]:
            count+=1
    return count/total

def main():
    met,word_list,train_df = train()
    # test_df = read_file('test_set2.csv',[0]) # 只读字符串部分
    test_set_predict(1,"17341190_KNN_classification.csv",word_list,met,train_df)
    # x=[]
    # # y=[]
    # validation_df = read_file('validation_set.csv')
    # for k in range(5,16):
    #     predict_tag,validation_df = KNN_predict(k,word_list,validation_df,met,train_df)
    #     res=cal_accuracy(predict_tag,validation_df)
    #     # x.append(k)
    #     # y.append(res)
    #     print("KNN模型在k=%d分类问题的验证集上的准确度为：%s"%(k,res))
    # plt.plot(x,y)
    # plt.show()
if __name__=="__main__":
    main()