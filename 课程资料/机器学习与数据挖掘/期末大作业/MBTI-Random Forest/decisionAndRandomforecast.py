#---------------
#
# 实现随机森林和决策树模型
#
#---------------

import numpy as np
import pandas as pd
import heapq
import matplotlib.pyplot as plt

from random import randrange

#---------------
#
# 实现随机森林模型
#
#---------------
# 构造数据子集
def get_subsample(dataSet, ratio):
    subdataSet = []
    lenSubdata = round(len(dataSet) * ratio)  # 返回浮点数
    while len(subdataSet) < lenSubdata:
        index = randrange(len(dataSet) - 1)
        subdataSet.append(dataSet[index])
    return subdataSet


# 分割数据集
def data_spilt(dataSet, index, value):
    left = []
    right = []
    for row in dataSet:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# 计算gini系数选取特征
def gini(left, right, class_values):
    loss = 0.0
    for class_value in class_values:
        left_size = len(left)
        if left_size != 0:
            prop = [row[-1] for row in left].count(class_value) / float(left_size)
            loss += (prop * (1.0 - prop))
        right_size = len(right)
        if right_size != 0:
            prop = [row[-1] for row in right].count(class_value) / float(right_size)
            loss += (prop * (1.0 - prop))
    print(loss)
    return loss


# 决定输出标签
def decide_label(data):
    output = [row[-1] for row in data]
    print(output)
    return max(set(output), key=output.count)


# 递归构建决策树
def sub_spilt(root, n_features, max_depth, min_size, depth):
    left = root['left']
    right = root['right']
    del (root['left'])
    del (root['right'])
    if not left or not right:
        root['left'] = root['right'] = decide_label(left + right)
        return
    if depth > max_depth:
        root['left'] = decide_label(left)
        root['right'] = decide_label(right)
        return
    if len(left) < min_size:
        root['left'] = decide_label(left)
    else:
        root['left'] = get_best_spilt(left, n_features)
        sub_spilt(root['left'], n_features, max_depth, min_size, depth + 1)
    if len(right) < min_size:
        root['right'] = decide_label(right)
    else:
        root['right'] = get_best_spilt(right, n_features)
        sub_spilt(root['right'], n_features, max_depth, min_size, depth + 1)

# 构造参数设置
n_folds = 5
max_depth = 3
min_size = 1
ratio = 1.0
n_features = 15
n_trees = 10

# 构造决策树
def build_tree(dataSet, n_features, max_depth, min_size):
    root = get_best_spilt(dataSet, n_features)
    sub_spilt(root, n_features, max_depth, min_size, 1)
    return root


# 预测测试集结果
def predict(tree, row):
    if row[tree['index']] < tree['value']:
        if isinstance(tree['left'], dict):
            return predict(tree['left'], row)
        else:
            return tree['left']
    else:
        if isinstance(tree['right'], dict):
            return predict(tree['right'], row)
        else:
            return tree['right']


# 选取任意的n个特征并选取最优的特征进行分裂
def get_best_spilt(dataSet, n_features):
    features = []
    class_values = list(set(row[-1] for row in dataSet))
    b_index, b_value, b_loss, b_left, b_right = 999, 999, 999, None, None
    while len(features) < n_features:
        index = randrange(len(dataSet[0]) - 1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataSet:
            left, right = data_spilt(dataSet, index, row[index])
            loss = gini(left, right, class_values)
            if loss < b_loss:
                b_index, b_value, b_loss, b_left, b_right = index, row[index], loss, left, right
    return {'index': b_index, 'value': b_value, 'left': b_left, 'right': b_right}




#---------------
#
# 实现决策树模型
#
#---------------
class DecisionNode(object):
    """
    决策树的节点
    """
    def __init__(self, label=None,feature=None,branch=None):
        self.feature = feature        # 当前节点对应的属性标签 众数
        self.label = label      # 保存的是针对当前分支的结果，有值则表示该点是叶子节点
        self.branch = branch            # 分支的字典


def readDataSet():
    """
    导入数据
    """
    # 对数据进行处理
    dataSet = pd.read_csv('../data/lab2_dataset/train.csv')
    # print(dataSet.values)
    labelSet = list(dataSet.columns.values)# 读取属性名
    dataSet = list(dataSet.values) # 读取数据变成列表
    return dataSet, labelSet

def count_feature_dic(dataSet,labelSet):
    '''
    计算特征的字典,格式为特征在labelSet的下标作为key sub特征作为values
    '''
    feature_num = len(labelSet)-1  # 属性的个数 不算标签
    feature_dic={}
    for i in range(0,feature_num):
        featureSet = set()
        for record in dataSet:
            featureSet.add(record[i]) # 获取当前特征所有可能的取值
        feature_dic[i] = featureSet

    return feature_dic

def cal_label_InforEntropy(subdataset):
    '''
    计算当前数据集的信息熵值
    :param subdataset: 最后一列为标签值，其他为属性值
    :return: 返回信息熵的结果
    '''
    total = len(subdataset)
    labelCounts = {} # 统计各个label的数量
    for record in subdataset:
        currentLabel = record[-1]
        labelCounts[currentLabel]=labelCounts.get(currentLabel,0)+1

    InforEntropy = 0.0
    for item in labelCounts.items():
        prob = float(item[1]) / total
        if prob!=0:
            # 如果是0的时候就log0=0
            InforEntropy -= prob * np.log2(prob)
    return InforEntropy


def cal_feature_InforEntropy(subdataset,index):
    '''
        计算当前数据集某一个属性的信息熵值
        用于C4.5的决策树中
        :param subdataset: 最后一列为标签值，其他为属性值
        :return: 返回信息熵的结果
        '''
    total = len(subdataset)
    featureCounts = {}  # 统计各个subfeature的数量
    for record in subdataset:
        currentLabel = record[index]
        featureCounts[currentLabel] = featureCounts.get(currentLabel, 0) + 1

    InforEntropy = 0.0
    for item in featureCounts.items():
        prob = item[1] / total
        if prob != 0:
            # 如果是0的时候就0*log0=0
            InforEntropy -= prob * np.log2(prob)
    return InforEntropy


def splitDataSet(dataSet, index, subfeature):
    '''
    根据选定的属性划分数据集
    :return:
    '''
    DataSet = []
    for record in dataSet:
        # 将相同数据特征的提取出来
        if record[index] == subfeature:
            DataSet.append(record)
    return DataSet


def countsubfeature(dataSet,feature):
    '''
    Gini指数的时候遍历数据集合统计需要的数据
    '''
    subfeature_count_dic={}
    for record in dataSet:
        # 这个子属性的统计加一
        subfeature_count_dic[record[feature]]=subfeature_count_dic.get(record[feature],[0,{}])
        subfeature_count_dic[record[feature]][0]+=1
        # 这个子属性对应的标签加一
        subfeature_count_dic[record[feature]][1][record[-1]]=subfeature_count_dic[record[feature]][1].get(record[-1],0)
        subfeature_count_dic[record[feature]][1][record[-1]]+=1
    return subfeature_count_dic

def chooseBestFeature(dataSet,feature_dic,available_feature,method="ID3"):
    '''
    根据前面的辅助函数选出最优的特征并
    作为当前子树的根结点
    '''
    if method=="ID3":
        total=len(dataSet)# 总的数据条数
        rootEntropy = cal_label_InforEntropy(dataSet) # 引入信息前的信息熵
        InforGain_list=[]
        for feature in available_feature:
            curInforEntropy = 0.0 # 当前特征计算出来的信息熵的值
            for subfeature in feature_dic[feature]:
                subDataSet = splitDataSet(dataSet, feature, subfeature)
                # 特征为i的数据集占总数的比例
                prob = len(subDataSet) / total
                curInforEntropy += prob * cal_label_InforEntropy(subDataSet)
            InforGain_list.append((rootEntropy - curInforEntropy,feature))

        heapq.heapify(InforGain_list)
        # print(InforGain_list)
        return heapq.nlargest(1,InforGain_list)[0][1]

    elif method == "C4.5":
        total = len(dataSet)  # 总的数据条数
        rootEntropy = cal_label_InforEntropy(dataSet)  # 引入信息前的信息熵
        InforGain_list = []
        for feature in available_feature:
            curInforEntropy = 0.0  # 当前特征计算出来的信息熵的值
            for subfeature in feature_dic[feature]:
                subDataSet = splitDataSet(dataSet, feature, subfeature)
                # 特征为i的数据集占总数的比例
                prob = len(subDataSet) / total
                curInforEntropy += prob * cal_label_InforEntropy(subDataSet)

            # 计算feature的信息熵：
            splitInfo = cal_feature_InforEntropy(dataSet,feature)
            if splitInfo==0:
                continue
            InforGain_list.append(((rootEntropy - curInforEntropy)/splitInfo, feature))

        heapq.heapify(InforGain_list)
        # print(InforGain_list)
        return heapq.nlargest(1, InforGain_list)[0][1]

    elif method =="Gini":
        total = len(dataSet)
        InforGini_list=[]
        for feature in available_feature:
            curInfoGini=0
            subfeature_count_dic = countsubfeature(dataSet, feature) # 统计每个子特征出现次数 和对应标签的出现次数
            for item in subfeature_count_dic.items():
                feature_prob=item[1][0]/total # 属性的概率值
                label_prob=0 # 统计标签的概率平方和
                for label in item[1][1].items():
                    label_prob+=np.power(label[1]/item[1][0],2)
                curInfoGini+=feature_prob*(1-label_prob)

        InforGini_list.append((curInfoGini , feature))

        heapq.heapify(InforGini_list)
        # print(InforGain_list)
        return heapq.nsmallest(1, InforGini_list)[0][1] # 选取最小的作为Gini指标选出的结果



def createTree(dataSet,parent_label,available_feature,feature_dic):
    """
    构造决策树
    """
    # 收集最后一列的标签值
    classList = [record[-1] for record in dataSet]
    # 如果数据集为空集，则节点的属性取父节点的值
    if len(dataSet) == 0:
        return DecisionNode(label=parent_label,feature=None, branch=None)
    # 当类别与属性完全相同时停止
    if classList.count(classList[0]) == len(classList):
        return DecisionNode(label=classList[0],feature=None, branch=None)
    # 当没有特征值时，直接返回数量最多的属性作为节点的值
    if (len(available_feature) == 0):
        return DecisionNode(label=max(classList,key=classList.count), feature=None, branch=None)

    # 选出最好的特征 对应数据集的列index
    bestFeature = chooseBestFeature(dataSet,feature_dic,available_feature,method="C4.5")
    available_feature.remove(bestFeature) # 删去用掉的最好特征
    branch={}
    parent_label=max(classList,key=classList.count) # 求出当前节点类别数量最多的一者传到子节点
    for subfeature in feature_dic[bestFeature]:
        split_dataSet=splitDataSet(dataSet,bestFeature,subfeature) # 根据子特征的值提取数据
        branch[subfeature]=createTree(split_dataSet,parent_label,available_feature[:],feature_dic)

    return DecisionNode(feature=bestFeature,label=parent_label,branch=branch)

def inorder(root,retract):
    print(retract+"属性值为：",root.feature,"标签值为：",root.label)
    if root.branch != None:
        for i in root.branch.values():
            inorder(i,retract+"    ")


def k_fold(dataSet,k,i):
    '''
    对数据集进行训练集和验证集的划分
    :param dataSet: 数据集
    :param k: 划分成k个部分
    :param i: 选取第i个部分为验证集
    :return: 返回训练和验证集的数据
    '''
    total = len(dataSet)
    step_len=total//k # 求出每一份的长度
    val_begin=i*step_len
    val_end=val_begin+step_len
    return dataSet[:val_begin]+dataSet[val_end:],dataSet[val_begin:val_end]



def validation(validation_dataset,root):
    count = 0
    for i,record in enumerate(validation_dataset):
        cur=root
        while cur.branch != None:
            cur = cur.branch[record[cur.feature]]
        if cur.label==record[-1]:
            count+=1
    return count/len(validation_dataset)

if __name__=="__main__":
    dataSet, labelSet = readDataSet()
    # x=[]
    # y=[]
    for k in range(3,9):
        # kfold 的k值取不同
        res=0
        for i in range(k):
            # 使用k-fold，对不同的验证集取均值作为一个k的结果
            train_set,validation_set = k_fold(dataSet,k,i)
            feature_dic = count_feature_dic(dataSet,labelSet)
            available_feature=list(range(0,len(labelSet)-1))
            root = createTree(train_set,-1,available_feature,feature_dic)
            res += validation(validation_set,root)
        # y.append(res/k)
        print("对数据集进行%s折后的正确率为%s"%(k,res/k))
        # x.append(k)

    # plt.figure()
    # plt.plot(x,y)
    # plt.show()