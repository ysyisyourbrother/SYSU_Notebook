import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def read_file(filename):
    df = pd.read_csv("./data/"+filename,header=None)
    data = df.values
    return data


def cal_wrong(w,data):
    '''
    口袋算法统计错误数量
    '''
    count = 0
    for point in data:
        # 遍历每个数据点不断更新w
        x = np.append(np.array([1]), point[:-1])  # 在开头加一个1表示阈值
        res = x.dot(w)
        if not ((point[-1] == 1 and np.sign(res) == 1) or (point[-1] == 0 and np.sign(res) == -1) or res == 0):
            count += 1
    return count

def pocket_alogrithm(data,validation_set):
    '''
        pocket_alogrithm算法，当算法迭代次数达到上限，或者没有错误的点的时候停止
    '''
    X = []
    Y = []
    iteration=100
    Bestw_wrong = len(data)+1
    bestW = np.ones(len(data[0]))
    itera=0
    while itera<iteration:
        point=data[np.random.choice(len(data))]  # 随机选取一个点更新
        x = np.append(np.array([1]), point[:-1])  # 在开头加一个1表示阈值
        res = x.dot(bestW)
        if not ((point[-1] == 1 and np.sign(res) == 1) or (point[-1] == 0 and np.sign(res) == -1) or res == 0):
            itera += 1
            # print(itera, end=' ')
            # 如果出现错误的点，更新线
            w = bestW + (-np.sign(res)) * x
            # 和当前最优的线比较谁最好
            w_wrong = cal_wrong(w, data)
            if w_wrong < Bestw_wrong:
                bestW = w
                Bestw_wrong = w_wrong
        acc = validation(validation_set, bestW)
        X.append(itera)
        Y.append(acc)
        print("PLA在迭代次数为%d时，验证集上的正确率为：%f" % (itera, acc))
    plt.plot(X, Y)
    plt.show()
    return bestW


def PLA_algorithm(data,validation_set):
    '''
    PLA算法，当算法迭代次数达到上限，或者没有错误的点的时候停止
    '''
    X=[]
    Y=[]
    iteration=100
    w = np.ones(len(data[0]))  # 定义权重向量
    count=0
    for iter in range(iteration):
        # PLA算法迭代iteration或者没有错误点的时候停止
        count+=1
        isComplete=1
        for point in data:
            # 遍历每个数据点不断更新w
            x=np.append(np.array([1]),point[:-1]) #在开头加一个1表示阈值
            res = x.dot(w)
            if (point[-1]==1 and np.sign(res)==1) or (point[-1]==0 and np.sign(res)==-1) or res == 0:
                continue
            else:
                # 出现错误的点，更新权重向量w并立刻返回
                w=w+(-np.sign(res))*x
                # isComplete=False
                break
        # if isComplete:
        #     # 如果遇到一次是没有错误点的就可以直接退出了
        #     break
        # acc = validation(validation_set,w)
        # X.append(iter)
        # Y.append(acc)
        # print("PLA在迭代次数为%d时，验证集上的正确率为：%f" % (iter,acc))
    # plt.plot(X,Y)
    # plt.show()
    return w


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
    return np.vstack((dataSet[:val_begin],dataSet[val_end:])),dataSet[val_begin:val_end]


def validation(validation_set,w):
    count=0
    total=validation_set.shape[0]
    for i,point in enumerate(validation_set):
        # 遍历每个数据点不断更新w
        x=np.append(np.array([1]),point[:-1]) #在开头加一个1表示阈值
        res = x.dot(w)
        # if (point[-1]==1 and np.sign(res)==1) or (point[-1]==0 and np.sign(res)==-1) or res == 0:
        #     count+=1
        #     print("第%d个点的预测标签为：%d"%(i,1))
        # else:
        #     print("第%d个点的预测标签为：%d" % (i, 0))
        if res>=0:
            print("第%d个点的预测标签为：%d"%(i,1))
        else:
            print("第%d个点的预测标签为：%d" % (i, 0))
    return count/total


def main():
    train_set = read_file("check_train.csv")
    validation_set=read_file("check_test.csv")
    # train_set,validation_set = k_fold(data,10,3)
    w = PLA_algorithm(train_set,validation_set)

    validation(validation_set,w)





if __name__=="__main__":
    main()






























