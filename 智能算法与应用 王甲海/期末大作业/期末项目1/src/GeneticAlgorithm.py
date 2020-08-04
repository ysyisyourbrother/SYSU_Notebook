import os
import sys
import numpy as np
import matplotlib.pyplot as plt 
import imageio
import shutil
from random import shuffle
import random
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


def random_reverse_subpath(path):
    """
        随机将原路径中的部分路径反向
    """
    while True:
        point1 = np.random.randint(0, N-1)
        point2 = np.random.randint(0, N-1)
        if point1 != point2 and point1 < point2:
            break
    path[point1:point2+1] = path[point1:point2+1][::-1]
    return path

# 种群数
pop_count = 150
# 进化次数
iteration = 5000
# 交配概率
crossover_rate = 0.8
# 变异率
mutation_rate = 0.2
# 适者生存的比例
sustain_rate = 0.3
# 弱者被接受的概率
random_select_rate = 0.3

def selection(population):
    """
        自然选择, 适应性强的生存，适应性弱的有一定几率生存，保证种群数量不变
    """
    population = sorted(population, key = cal_path_len)

    # 适应能力强的比例为
    retain_length = int(len(population) * sustain_rate)
    sustain = population[:retain_length]    # 适应能力强，一定会存活下来

    # 适应能力不够强，但有一定几率可以存活
    for chromosome in population[retain_length:]:
        if random.random() < random_select_rate:
            sustain.append(chromosome)

    # 用适应性强的个体补充数量，保证种群数量不变
    if len(sustain) < pop_count:
        sustain = sustain + population[:pop_count-len(sustain)]
    return sustain


def mutation(population):
    """
        发生变异，用临近解选择策略重新生成路径
    """
    for i,path in enumerate(population):
        if np.random.rand() < mutation_rate:
            tmp_path = random_reverse_subpath(path) # 发生变异，用临近策略重新生成
            population[i] = tmp_path
    return population
        
            

def cross(population):
    """
        遗传算法交配策略
        相邻的两条路径有一定几率发生交配
    """
    index = 0
    while index < pop_count-1:
        if random.random() < crossover_rate:
            index += 1
            continue
        else:
            while True:
                left=random.randint(0,N)
                right=random.randint(0,N)
                if left != right and left < right:
                    break
            pop1, pop2 = population[index], population[index+1]

            # 交叉片段，因此gene1和gene2不完全相同，要把各自缺少的城市补上
            gene1 = pop1[left:right]
            gene2 = pop2[left:right]
            p1_other, p2_other = [], []
            for i in range(len(pop1)):
                if pop1[i] not in gene2:
                    p1_other.append(pop1[i])
                if pop2[i] not in gene1:
                    p2_other.append(pop2[i])
            population[index] = p2_other[:left] + gene1 + p2_other[left:]
            population[index+1] = p1_other[:left] + gene2 + p1_other[left:]
            index += 2
    return population


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
    # 首先需要初始化一定规模的种群
    population = []
    all_path_len = []   # 记录每轮迭代后最优的路径，用来绘图
    for i in range(pop_count):
        x = [i for i in range(N)]
        shuffle(x)
        population.append(x)

    for i in range(iteration):
        # 开始迭代，进行交换和突变
        population = selection(population)  # 适者生存
        # 交换基因
        population = cross(population)
        # 变异：使用临近搜索函数
        population = mutation(population)

        best_path_len, best_path = get_best_path(population)
        all_path_len.append(best_path_len)
        print("iteration: ", i)

    # 训练后找出最优解
    opt_dist, opt_path = get_best_path(population)
    actual_opt_dist = cal_path_len(shortest_path)
    print("===============================================")
    print("计算得到的局部最优路径长度为：", opt_dist)
    print("实际最优路径长度：", actual_opt_dist)
    print('局部最优解超出全局最优解的比例为： %f%%' % ((opt_dist/actual_opt_dist-1)*100))
    print("===============================================")
    draw(opt_path)
    plt.figure()
    plt.plot(np.arange(len(all_path_len)), all_path_len)
    plt.show()
