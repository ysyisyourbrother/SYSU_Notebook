import os
from graphics import *
from math import *
import numpy as np


AI_pos = set()  # 记录电脑下的棋的落子位置
Man_pos = set()  # 记录玩家下的棋的落子位置
All_pos = set()  # 记录整个当前的棋局
ChessBoard = set()  # 记录整个棋盘可以落子的位置

GRID_WIDTH = 40
col = 11
row = 11
MAX_DEPTH = 3

who_first = 1  # 1是人先走  0是机先走
AI_FIRST_DEFAULT = (col // 2, row // 2)  # 如果AI先手，默认移动的位置

# 棋型的评估分数
shape_score = [(50, (0, 1, 1, 0, 0)),
               (50, (0, 0, 1, 1, 0)),
               (200, (1, 1, 0, 1, 0)),
               (500, (0, 0, 1, 1, 1)),
               (500, (1, 1, 1, 0, 0)),
               (5000, (0, 1, 1, 1, 0)),
               (5000, (0, 1, 0, 1, 1, 0)),
               (5000, (0, 1, 1, 0, 1, 0)),
               (5000, (1, 1, 1, 0, 1)),
               (5000, (1, 1, 0, 1, 1)),
               (5000, (1, 0, 1, 1, 1)),
               (5000, (1, 1, 1, 1, 0)),
               (5000, (0, 1, 1, 1, 1)),
               (50000, (0, 1, 1, 1, 1, 0)),
               (99999999, (1, 1, 1, 1, 1))]


# 评估函数
def evaluation():
    '''
    计算当前整个棋局的得分   人是极大节点  AI是极小节点
    '''
    # 算自己的得分
    score_all_arr_man = []  # 得分形状的位置 用于计算如果有相交 得分翻倍
    Man_score = 0
    for pt in Man_pos:
        m = pt[0]
        n = pt[1]
        Man_score += cal_score(m, n, 0, 1, AI_pos, Man_pos, score_all_arr_man) # 每个点每个方向只取一个最高得分
        Man_score += cal_score(m, n, 1, 0, AI_pos, Man_pos, score_all_arr_man)
        Man_score += cal_score(m, n, 1, 1, AI_pos, Man_pos, score_all_arr_man)
        Man_score += cal_score(m, n, -1, 1, AI_pos, Man_pos, score_all_arr_man)

    #  算ai的得分， 并减去
    score_all_arr_ai = []
    AI_score = 0
    for pt in AI_pos:
        m = pt[0]
        n = pt[1]
        AI_score += cal_score(m, n, 0, 1, Man_pos, AI_pos, score_all_arr_ai)
        AI_score += cal_score(m, n, 1, 0, Man_pos, AI_pos, score_all_arr_ai)
        AI_score += cal_score(m, n, 1, 1, Man_pos, AI_pos, score_all_arr_ai)
        AI_score += cal_score(m, n, -1, 1, Man_pos, AI_pos, score_all_arr_ai)

    return Man_score - AI_score*0.1


def cal_score(m, n, x_direct, y_direct, enemy_list, my_list, score_all_arr):
    '''
    计算当前步数的得分
    '''
    add_score = 0  # 加分项
    # 在一个方向上， 只取最大的得分项
    max_score_shape = (0, None)

    # 如果此方向上，该点已经有得分形状，不重复计算
    for item in score_all_arr:
        for pt in item[1]:
            if m == pt[0] and n == pt[1] and x_direct == item[2][0] and y_direct == item[2][1]:
                return 0

    # 在落子点 左右方向上循环查找得分形状
    for offset in range(-5, 1):
        # offset = -2
        pos = []
        for i in range(0, 6):
            if (m + (i + offset) * x_direct, n + (i + offset) * y_direct) in enemy_list:
                pos.append(2)  # 敌人的标记2
            elif (m + (i + offset) * x_direct, n + (i + offset) * y_direct) in my_list:
                pos.append(1)  # 自己的标记1
            else:
                pos.append(0)  # 空的标记0
        tmp_shap5 = (pos[0], pos[1], pos[2], pos[3], pos[4])
        tmp_shap6 = (pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])

        for (score, shape) in shape_score:
            if tmp_shap5 == shape or tmp_shap6 == shape:
                if score > max_score_shape[0]:
                    max_score_shape = (score, ((m + (0 + offset) * x_direct, n + (0 + offset) * y_direct),
                                               (m + (1 + offset) * x_direct, n + (1 + offset) * y_direct),
                                               (m + (2 + offset) * x_direct, n + (2 + offset) * y_direct),
                                               (m + (3 + offset) * x_direct, n + (3 + offset) * y_direct),
                                               (m + (4 + offset) * x_direct, n + (4 + offset) * y_direct)),
                                       (x_direct, y_direct))

    # 计算两个形状相交， 如两个3活 相交， 得分增加 一个子的除外
    if max_score_shape[1] is not None:
        for item in score_all_arr: # 查看别的方向上的得分形状
            for pt1 in item[1]:
                for pt2 in max_score_shape[1]:  # 如果存在两个得分形状有点重合
                    if pt1 == pt2 and max_score_shape[0] > 10 and item[0] > 10:
                        add_score += item[0] + max_score_shape[0]  # 将重合的形状得分翻倍

        score_all_arr.append(max_score_shape)

    return add_score + max_score_shape[0]


def has_neighbour(point):
    '''
    判断点四周是否有棋子，如果都没有就不考虑这个点，加快搜索速度
    '''
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i == 0 and j == 0:
                continue
            if (point[0] + i, point[1] + j) in All_pos:
                return True
    return False


def game_win(point, is_ai):
    '''
    判断上一步下完后是否有一方取胜
    '''
    dir_list = ([(-1, 0), (1, 0)],[ (0, -1), (0, 1)],[ (-1, 1), (1, -1)], [(1, 1), (-1, -1)])  # 记录八个方向
    if is_ai == True:
        mylist = AI_pos
    else:
        mylist = Man_pos
    for dir_pair in dir_list:
        seq = 1
        for dir in dir_pair:
            cur = point
            while ((cur[0] + dir[0], cur[1] + dir[1]) in mylist) and row > cur[0] and cur[0]>= 0 and col > cur[1] and cur[1]>= 0:
                seq += 1
                if seq >= 5:
                    return True
                cur = (cur[0] + dir[0], cur[1] + dir[1])
    return False


def get_available_pos():
    return ChessBoard - All_pos


def MiniMax(last_step, depth, max_alpha, min_beta):
    # 极大节点 人走
    if game_win(last_step, True) or depth == 0:
        return evaluation(), (-1, -1)
    cur_max_alpha = float('-inf')  # 设定alpha为无穷小
    cur_best_pos = (-1, -1)
    available_pos = get_available_pos()  # 获取可以走的位置tuple
    for pos in available_pos:
        if not has_neighbour(pos):  # 不考虑周围没有棋子的位置，加快搜索速度
            continue
        Man_pos.add(pos)
        All_pos.add(pos)
        cur_alpha, _ = MiniMin(pos, depth - 1, max_alpha, min_beta)

        if cur_max_alpha < cur_alpha:
            cur_max_alpha = cur_alpha
            cur_best_pos = pos

        Man_pos.remove(pos)
        All_pos.remove(pos)

        if cur_max_alpha > min_beta:  # alpha 剪枝
            return cur_max_alpha, cur_best_pos

        # 如果要继续搜索，查看是否需要更新当前最大的alpha值
        if cur_max_alpha >= max_alpha:
            max_alpha = cur_max_alpha

    return cur_max_alpha, cur_best_pos


def MiniMin(last_step, depth, max_alpha, min_beta):
    # 极小节点 AI走
    if game_win(last_step, False) or depth == 0:
        return evaluation(), (-1, -1)
    cur_min_beta = float('inf')  # 设定beta为无穷大
    cur_best_pos = (-1, -1)
    available_pos = get_available_pos()  # 获取可以走的位置tuple
    for pos in available_pos:
        if not has_neighbour(pos):  # 不考虑周围没有棋子的位置，加快搜索速度
            continue

        AI_pos.add(pos)
        All_pos.add(pos)

        cur_beta, _ = MiniMax(pos, depth - 1, max_alpha, min_beta)
        if cur_min_beta > cur_beta:  # 将子节点中最小的返回值赋值给当前节点的beta值
            cur_min_beta = cur_beta
            cur_best_pos = pos
        AI_pos.remove(pos)
        All_pos.remove(pos)

        if cur_min_beta < max_alpha:  # beta 剪枝
            return cur_min_beta, cur_best_pos

        # 如果要继续搜索，查看是否需要更新当前最小的beta值
        if cur_min_beta < min_beta:
            min_beta = cur_min_beta

    return cur_min_beta, cur_best_pos


# def Display():
#     # 画图
#     board = [[' ' for n in range(col)] for m in range(row)]
#     for i in range(col):
#         board[0][i] = i
#     for j in range(row):
#         board[j][0] = j
#     #  print(board)
#     if who_first:
#         for pos in AI_pos:
#             board[pos[0]][pos[1]] = 'x'
#         for pos in Man_pos:
#             board[pos[0]][pos[1]] = 'o'
#     else:
#         for pos in AI_pos:
#             board[pos[0]][pos[1]] = 'o'
#         for pos in Man_pos:
#             board[pos[0]][pos[1]] = 'x'
#
#     for n in range(col):
#         print('----', end='')
#     print('')
#     for m in range(row):
#         for n in range(col):
#             print("| {} ".format(board[m][n]), end='')
#         print('|')
#         for n in range(col):
#             print('----', end='')
#         print('')


# def main():
#     # 初始化棋盘
#     for i in range(col + 1):
#         for j in range(row + 1):
#             ChessBoard.add((i, j))
#
#     if who_first == 0:  # 如果用户先走的话
#         a, b = map(int, input('输入a,b空格隔开:').split())
#         Man_pos.add((a, b))
#         All_pos.add((a, b))
#         Display()
#     else:  # 如果AI先走  默认走中间
#         AI_pos.add(AI_FIRST_DEFAULT)
#         All_pos.add(AI_FIRST_DEFAULT)
#         print("电脑下了：", AI_FIRST_DEFAULT)
#         a, b = map(int, input('输入a,b空格隔开:').split())
#         Man_pos.add((a, b))
#         All_pos.add((a, b))
#         Display()
#
#     while True:
#         _, pos = MiniMin((a, b), MAX_DEPTH, -float("inf"), float("inf"))
#         print("电脑下了：", pos)
#         AI_pos.add(pos)
#         All_pos.add(pos)
#         Display()
#         if game_win(pos, True):
#             # 如果机器人获胜
#             print("robot win!!")
#             return
#         a, b = map(int, input('输入a,b空格隔开:').split())
#         Man_pos.add((a, b))
#         All_pos.add((a, b))
#         Display()
#         if game_win((a, b), False):
#             print("man win!!")
#             return


def gobangwin():
    win = GraphWin("gobang game by ysy", GRID_WIDTH * col, GRID_WIDTH * row)
    win.setBackground("grey")
    i1 = 0

    while i1 <= GRID_WIDTH * col:
        l = Line(Point(i1, 0), Point(i1, GRID_WIDTH * col))
        l.draw(win)
        i1 = i1 + GRID_WIDTH
    i2 = 0

    while i2 <= GRID_WIDTH * row:
        l = Line(Point(0, i2), Point(GRID_WIDTH * row, i2))
        l.draw(win)
        i2 = i2 + GRID_WIDTH
    return win

def main():
    win = gobangwin()
    for pos in Man_pos:
        piece = Circle(Point(GRID_WIDTH * pos[1], GRID_WIDTH * pos[0]), 16)
        piece.setFill('black')
        piece.draw(win)
    for pos in AI_pos:
        piece = Circle(Point(GRID_WIDTH * pos[1], GRID_WIDTH * pos[0]), 16)
        piece.setFill('white')
        piece.draw(win)


    for i in range(col+1):
        for j in range(row+1):
            ChessBoard.add((i, j))

    change = 0
    g = 0
    m = 0
    n = 0
    last_step=(-1,-1)
    while g == 0:

        if change % 2 == who_first:
            _,pos = MiniMin(last_step, MAX_DEPTH, -float("inf"), float("inf"))

            if pos in All_pos:
                message = Text(Point(200, 200), "不可用的位置" + str(pos[0]) + "," + str(pos[1]))
                message.draw(win)
                g = 1

            AI_pos.add(pos)
            All_pos.add(pos)
            print("第%d回合电脑落子得分为：" % (change//2+1), evaluation())

            piece = Circle(Point(GRID_WIDTH * pos[1], GRID_WIDTH * pos[0]), 16)
            piece.setFill('white')
            piece.draw(win)

            if game_win(pos,True):
                message = Text(Point(100, 100), "white win.")
                message.draw(win)
                g = 1
            change = change + 1

        else:
            p2 = win.getMouse()
            if not ((round((p2.getY()) / GRID_WIDTH), round((p2.getX()) / GRID_WIDTH)) in All_pos):

                a2 = round((p2.getX()) / GRID_WIDTH)
                b2 = round((p2.getY()) / GRID_WIDTH)
                Man_pos.add((b2, a2))
                All_pos.add((b2, a2))
                print("第%d回合用户落子得分为："%(change//2+1),evaluation())
                last_step=(b2,a2)

                piece = Circle(Point(GRID_WIDTH * a2, GRID_WIDTH * b2), 16)
                piece.setFill('black')
                piece.draw(win)
                if game_win(last_step,False):
                    message = Text(Point(100, 100), "black win.")
                    message.draw(win)
                    g = 1

                change = change + 1

    message = Text(Point(100, 120), "Click anywhere to quit.")
    message.draw(win)
    win.getMouse()
    win.close()



if __name__ == "__main__":
    #     for i in range(col+1):
    #         for j in range(row+1):
    #             ChessBoard.add((i, j))
    #
    #     AI_pos.add((4,4))
    #     AI_pos.add((4,6))
    #     AI_pos.add((3,5))
    #     AI_pos.add((2,4))
    #     AI_pos.add((5,6))
    #     AI_pos.add((9,1))
    #     AI_pos.add((2,6))
    #     AI_pos.add((8,5))
    #     AI_pos.add((1,7))
    #     AI_pos.add((3,6))
    #
    #     Man_pos.add((5,5))
    #     Man_pos.add((6,4))
    #     Man_pos.add((4,5))
    #     Man_pos.add((7,3))
    #     Man_pos.add((5,7))
    #     Man_pos.add((8,2))
    #     Man_pos.add((7,5))
    #     Man_pos.add((6,5))
    #     Man_pos.add((6,6))
    #     Man_pos.add((5,3))
    #     Man_pos.add((1,6))
    #
    #     All_pos=AI_pos | Man_pos
    #     # ChessBoard=All_pos | {(0,8),(10,2)}
    #     Display()
    #     print(MiniMin((1, 6), MAX_DEPTH, float("-inf"), float("inf")))


    who_first=int(input("请选择玩家先手1 还是电脑先手0  请输入："))

    if who_first:# 此时是人先手 人是x
        print("人先手")
        AI_pos.add((5, 4))
        AI_pos.add((6, 5))
        Man_pos.add((5, 5))
        Man_pos.add((5, 6))
    else:
        print("电脑先手")
        Man_pos.add((5, 4))
        Man_pos.add((6, 5))
        AI_pos.add((5, 5))
        AI_pos.add((5, 6))
    All_pos.add((5,4))
    All_pos.add((6,5))
    All_pos.add((5,5))
    All_pos.add((5,6))


    main()
