import math

def read_file():
    '''
    读取目标文件内容
    :return: 返回目标文件形成的列表
    '''
    data = []
    with open('../data/lab1_data/semeval.txt', 'r') as f:
        for record in f:
            data.append(record.split()[8:])
    return data

def TF(data):
    '''
    计算每篇文本词语的TF值
    :return:返回文本词语频率的字典
    '''
    total = len(data) # 统计总的词语数量
    dict={}
    for word in data:
        dict[word]=(dict.get(word,0)*total+1)/total
    return dict

def IDF(data,word_list):
    '''
    输入所有文本词语组成的列表，遍历目标词语列表并计算每个词语的IDF
    :param data:
    :return:返回目标词语列表的每一个词语对应的IDF
    '''
    IDF_list=[]
    total = len(data)
    for word in word_list:
        count=0
        for record in data:
            if word in record:
                count+=1
        IDF_list.append(math.log((total+1)/(count+1)))
    return IDF_list


def count_words(data):
    '''
    输入所有文本组成的列表
    按照出现先后收集所有的词语并组成列表返回
    :param data:
    :return: 返回词语按先后顺序形成的列表
    '''
    word_set=set()
    word_list=[]
    for record in data:
        for word in record:
            if word not in word_set: # 利用集合来查询速度更快，再按照先后顺序加入到列表中
                word_list.append(word)
                word_set.add(word)
    return word_list


def write_file(data):
    '''
    将输出的TFIDF矩阵写入文件内
    :param data:
    :return:
    '''
    with open("TFIDF_matrix.txt",'a') as f:
        for i in data:
            for j in i:
                f.write(str(j)+" ")
            f.write("\n")


def main():
    '''
    利用目标文件计算TFIDF矩阵
    :return: 返回TFIDF矩阵结果
    '''
    TFIDF_met=[]
    data = read_file() # 读取数据
    word_list=count_words(data) # 计算词语列表
    IDF_list=IDF(data,word_list) # 计算idf值
    print(IDF_list)
    for record in data:
        res=[]
        TF_dic=TF(record)
        for index,word in enumerate(word_list):
            res.append(TF_dic.get(word,0)*IDF_list[index])
        TFIDF_met.append(res)
    write_file(TFIDF_met)
    return TFIDF_met



if __name__=="__main__":
    main()

