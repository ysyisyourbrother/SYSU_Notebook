import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_file(filename):
    df = pd.read_csv("./data/"+filename,header=None)
    data = df.values
    return data

def h(w,x):
    '''
    逻辑回归的预测函数h(x)
    '''
    return 1/(1+np.exp(-(x.dot(w))))


def logistic(data,validation_set):
    '''
    基于批梯度下降法的逻辑回归算法
    '''
    X=[]
    Y=[]
    w=np.zeros(len(data[0])) # 定义权重向量
    max_loop = 0 # 最大循环次数
    alpha=0.001 # 设置学习率
    diff = 1e-4 # 当梯度小于这个的时候停止
    while max_loop<100:
        max_loop+=1
        # print(max_loop)
        sum=0
        for point in data:
            x=np.append(np.array([1]),point[:-1]) #在开头加一个1表示阈值
            sum-=(point[-1]-h(w,x))*x
        w_new=w-alpha*sum
        now_diff = np.linalg.norm(w_new - w)
        if(now_diff <= diff):
            break
        w = w_new
        # acc = validation(validation_set, w)
        # X.append(max_loop)
        # Y.append(acc)
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
        # if (h(w,x)>=0.5 and point[-1]==1)or(h(w,x)<0.5 and point[-1]==0):
        #     count+=1
        #     print("第%d个点的预测标签为：%d" % (i, 1))
        # else:
        #     print("第%d个点的预测标签为：%d" % (i, 0))
        if h(w,x)>=0.5:
            print("第%d个点的预测标签为：%d" % (i, 1))
        else:
            print("第%d个点的预测标签为：%d" % (i, 0))
    # print("逻辑回归在验证集上的正确率为：%f"%(count/total))
    return count/total

def main():
    train_set = read_file("check_train.csv")
    validation_set=read_file("check_test.csv")
    # train_set,validation_set = k_fold(data,5,0)
    
    w = logistic(train_set,validation_set)

    validation(validation_set,w)



if __name__=="__main__":
    main()

























