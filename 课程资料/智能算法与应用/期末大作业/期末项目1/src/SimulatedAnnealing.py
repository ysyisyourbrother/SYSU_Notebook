import os
import sys
import numpy as np
import matplotlib.pyplot as plt 
import imageio
import shutil
from random import shuffle
import time
import math

def read_data(filepath):
    """
        读取城市坐标和最优解的数据
    """
    city_file = open(filepath[0], 'r')
    shortest_path_file = open(filepath[1], 'r')
    
    cities, shortest_path = [], []
    for coord in city_file.readlines():
        coord = coord.strip().split()
        cities.append([float(coord[1]),float(coord[2])])
    for index in shortest_path_file.readlines():
        index = index.strip().split()
        shortest_path.append(int(index[0])-1)
    city_file.close()
    shortest_path_file.close()
    return np.array(cities), shortest_path, len(cities)
# 城市坐标文件和最优路径文件地址
filepath = ["../tc/kroC100.tsp", "../tc/kroC100.opt.tour"]
cities, shortest_path, N = read_data(filepath)

def cal_dist_mat(N):
    """
        计算距离矩阵
    """
    dist_mat = np.zeros((N,N))
    for i in range(N):
        for j in range(i,N):
            dist_mat[i][j] = dist_mat[j][i] = np.linalg.norm(cities[i]-cities[j])
    return dist_mat
dist_mat = cal_dist_mat(N)

def cal_path_len(path):
    """
        输入路径
        输出路径对应的距离
    """
    path_len = 0
    for i in range(N-1):
        path_len += dist_mat[path[i]][path[i+1]] 
    path_len += dist_mat[path[N-1]][path[0]]
    return path_len


def random_reverse_subpath(path):
    """
        随机将原路径中的部分路径反向
    """
    tmp_path = path.copy()
    while True:
        point1 = np.random.randint(0, N-1)
        point2 = np.random.randint(0, N-1)
        if point1 != point2 and point1 < point2:
            break
    tmp_path[point1:point2+1] = tmp_path[point1:point2+1][::-1]
    return tmp_path

def get_best_path(paths):
    """
        从多条路径中选出路程最短的路径
    """
    best_path = paths[0]
    best_path_len = cal_path_len(best_path)
    for path in paths:
        length = cal_path_len(path)
        if length < best_path_len:
            best_path_len = length
            best_path = path
    return best_path_len, best_path

def compare(path1, path2):
    """
        比较两条路径是否完全相同
    """
    for i in range(N):
        if path1[i] != path2[i]:
            return False
    return True

# 将结果图保存下来
pic_num = 0
def draw(path):
    """
        生成当前路径图并保存
    """
    global pic_num
    pic_path = './result'
    
    plt.cla()
    xs = [cities[i][0] for i in range(N)]
    ys = [cities[i][1] for i in range(N)]
    plt.scatter(xs, ys, color='b')
    xs = np.array(xs)
    ys = np.array(ys)
    for i in range(N-1):
        plt.plot(xs[[path[i], path[i+1]]], ys[[path[i], path[i+1]]], color='y')
    plt.plot(xs[[path[N-1], path[0]]], ys[[path[N-1], path[0]]], color='y')
    for i, p in enumerate(cities):
        plt.text(p[0], p[1], i)
    plt.savefig('%s/%d.png' % (pic_path, pic_num))
    pic_num += 1 


if __name__ == '__main__':
    # 随机选取一个相对较短的路径作为初始路径
    cur_path = np.arange(N)
    shuffle(cur_path)
    cur_path_len = cal_path_len(cur_path)

    all_path_len = [cur_path_len]   # 记录每轮降温的温度变化，用于绘图
    
    T = 5000 # 默认初始温度为度
    T_end = 0.01   # 终止条件为0.01度
    q = 0.98    # 退火系数
    L = 1000    # 每隔温度迭代1000次 
    strategy = random_reverse_subpath   # 默认使用第二种临近解方法

    while(T>T_end):
        print("current T：",T)
        for i in range(L):
            tmp_path = strategy(cur_path)
            tmp_len = cal_path_len(tmp_path)
            dE = tmp_len - cur_path_len
            if dE < 0:  # 如果新解更优，直接获取它
                cur_path = tmp_path.copy()
                cur_path_len = tmp_len
            else:   # 如果新解比较差，使用MetroPolis准则
                if np.random.rand() <= np.exp(-dE/T):   # 使用这个解
                    cur_path = tmp_path.copy()
                    cur_path_len = tmp_len
        T = T*q # 退火
        all_path_len.append(cur_path_len)  # 记录每轮降温的温度变化，用于绘图
    opt_dist = cal_path_len(cur_path)
    actual_opt_dist = cal_path_len(shortest_path)
    print("===============================================")
    print("计算得到的局部最优路径长度为：", opt_dist)
    print("实际最优路径长度：", actual_opt_dist)
    print('局部最优解超出全局最优解的比例为： %f%%' % ((opt_dist/actual_opt_dist-1)*100))
    print("===============================================")
    
    # 绘制最优情况下的路径图
    draw(cur_path)

    # 路径长度变化图
    plt.figure()
    plt.plot(np.arange(len(all_path_len)), all_path_len)
    plt.show()


