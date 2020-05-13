import heapq
import matplotlib.pyplot as plt
import time

moves = [[0,-1],[1,0],[-1,0],[0,1],]# 四种可能的移动方式
class Node:
    def __init__(self,location,cost=0,ancestor=None):
        self.location=location
        self.cost=cost # 到达这个点的最小花费
        self.ancestor = ancestor
    def show_data(self):
        print(self.location,self.ancestor,self.cost)

    def __lt__(self, other):
        # 重新定义小于符号
        return self.cost < other.cost

def readfile():
    with open('MazeData.txt', 'r', encoding='utf-8') as f:
        maze=[]
        for row,line in enumerate(f.readlines()):
            line = line.strip()
            s_col = line.find('S')# 找开始的位置
            e_col = line.find('E')# 找结束的位置
            if s_col is not -1:# 找到开始和结尾的位置
                S = Node([row, s_col])
            if e_col is not -1:
                E = Node([row, e_col])
            maze.append(line)
    return maze,S,E

def search():
    ### 进行一致代价搜索 0可以走1不可以走 代价都为1
    count = 0
    length=1
    maze,S,E = readfile()
    row,col = len(maze),len(maze[0])
    frontier = []  # 加入小顶堆  heapq.heappush(heap, num)
    explored = [[0] * col for _ in range(row)] # 0表示没有访问，1表示已经访问过了
    heapq.heappush(frontier,S)
    X,Y=[],[]
    while True:
        if len(frontier)==0:
            return None
        curNode = heapq.heappop(frontier) # 获取当前要搜索的节点
        if curNode.location == E.location:# 如果找到了目标的节点
            X.append(curNode.location[1])
            Y.append(-curNode.location[0])
            plt.xlim(0, 35)
            plt.ylim(-17, 0)
            plt.scatter(X, Y)
            plt.show()
            print("时间复杂度为：",count)
            print("空间复杂度为:", length)
            return curNode
        explored[curNode.location[0]][curNode.location[1]] = 1 # 将visit标记为 1
        count+=1

        X.append(curNode.location[1])
        Y.append(-curNode.location[0])

        for move in moves:
            new_location = [x + y for x, y in zip(curNode.location, move)]
            if explored[new_location[0]][new_location[1]]==0 and maze[new_location[0]][new_location[1]]=='0' or maze[new_location[0]][new_location[1]]=='E': # 如果这个点没被探索过
                flag=1
                for node in frontier:# 检查节点是否在边界队列中
                    if node.location==new_location and node.cost>curNode.cost+1:# 如果在边界就更新cost
                        node.cost = curNode.cost+1
                        node.ancestor = curNode
                        heapq.heapify(frontier)
                        flag=0
                        break
                if flag: # 如果不在边界队列中 就要加入到边界队列中去
                    new_node=Node(new_location,curNode.cost+1,curNode)
                    heapq.heappush(frontier, new_node)
                    if len(frontier)>length:
                        length=len(frontier)

def draw_map():
    maze, S, E = readfile()
    X,Y=[],[]

    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j]!='1':
                X.append(j)
                Y.append(-i)

    plt.xlim(0, 35)
    plt.ylim(-17, 0)
    plt.scatter(X, Y)
    plt.show()
def draw(end_Node):
    maze, S, E = readfile()
    X,Y=[],[]
    cur_node = end_Node
    while cur_node is not None:
        X.append(cur_node.location[1])
        Y.append(-cur_node.location[0])
        cur_node=cur_node.ancestor
    plt.xlim(0, 35)
    plt.ylim(-17, 0)
    plt.annotate('S',xy=(S.location[1],-S.location[0]),xytext=(S.location[1],-S.location[0]+0.1))
    plt.annotate('E',xy=(S.location[1],-E.location[0]),xytext=(E.location[1]-0.5,-E.location[0]+0.1))
    plt.plot(X, Y)
    plt.show()

if __name__=="__main__":
    res = search()
    if res is None:
        print("没有路径可以到达终点")
    else:
        draw(res)

    draw_map()