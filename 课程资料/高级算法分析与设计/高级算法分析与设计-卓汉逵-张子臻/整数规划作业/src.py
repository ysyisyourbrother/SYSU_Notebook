from gurobipy import *
import numpy as np

def readfile(filename):
    with open(filename, 'r') as f:
        rows = f.readline()
        num = int(rows)
        site = np.zeros((num, 2), float)
        for i in range(0, num):
            _, x, y = map(float, f.readline().split())
            site[i, 0] = x
            site[i, 1] = y
    # print(num)
    # print(site)
    return num, site

# 输入文件名
prefix = "./"
filenames = ["burma14.txt", "bays29.txt", "eil51.txt", "eil76.txt", "eil101.txt"]

#读取城市坐标
def MTZ(filename):
    num, coord = readfile(prefix + filename)

    #构造城市距离矩阵      
    dis = np.zeros((num, num))
    for i in range(0, num):
        for j in range(0, num):
                dis[i, j] = np.sqrt(np.square((coord[i][0] - coord[j][0])) + np.square((coord[i][1] - coord[j][1])))


    # 构建模型
    name = filename.split('.')[0]
    model = Model(name)

    #添加约束变量
    x = model.addMVar((num, num), lb=0, ub=1, vtype=GRB.BINARY)  
    u = model.addMVar((num), lb=0, vtype = GRB.INTEGER) 

    model.setObjective(quicksum(dis[i][j] * x[i][j] for i in range(num) for j in range(num)), GRB.MINIMIZE)
    
    model.addConstrs((quicksum(x[i][j] for j in range(num)) == 1) for i in range(num))

    model.addConstrs((quicksum(x[i][j] for i in range(num)) == 1) for j in range(num))

    model.addConstrs(((u[i]-u[j] + num * x[i][j]) <= (num - 1)) for j in range(1, num) for i in range(1, num) if i!=j)

    model.optimize()
    print("Problem： ", filename)
    print("MinLength： ", model.objVal)

if __name__ == "__main__":
    for file in filenames:
        MTZ(file)
