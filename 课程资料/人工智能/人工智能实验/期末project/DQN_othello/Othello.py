import numpy as np

BOARD_SIZE = 8


class Game(object):
    def __init__(self):
        self.board_size = BOARD_SIZE
        self.black_chess = set()
        self.white_chess = set()
        self.board = [[0 for j in range(self.board_size)] for i in range(self.board_size)]

        self.black_chess.add((BOARD_SIZE // 2 - 1, BOARD_SIZE // 2))
        self.black_chess.add((BOARD_SIZE // 2, BOARD_SIZE // 2 - 1))
        self.white_chess.add((BOARD_SIZE // 2 - 1, BOARD_SIZE // 2 - 1))
        self.white_chess.add((BOARD_SIZE // 2, BOARD_SIZE // 2))

        # 1表示黑棋，-1表示白棋
        for item in self.black_chess:
            self.board[item[0]][item[1]] = 1
        for item in self.white_chess:
            self.board[item[0]][item[1]] = -1

    def Gameover(self):
        """判断游戏是否结束，即整个棋盘下满了棋

        Returns:
            int -- 没结束返回0；黑棋赢返回1；白棋赢返回-1
        """
        if len(self.black_chess) + len(self.white_chess) == pow(self.board_size, 2) \
                or len(self.black_chess) * len(self.white_chess) == 0:
            return 1 if len(self.black_chess) > len(self.white_chess) else -1
        else:
            return 0

    def Get_Valid_Pos(self, my_chess, oppo_chess):
        """找出所有合法的落子点，即该位置落子后必须存在对方的棋需要翻转

        Arguments:
            my_chess {set} -- 己方当前的所有棋
            oppo_chess {set} -- 对方当前的所有棋

        Returns:
            valid_pos {set} -- 合法位置的集合,位置表示为元组(x,y)
        """
        valid_pos = set()
        for (x, y) in my_chess:
            temp = (x, y)
            while (x - 1, y) in oppo_chess:
                x -= 1
            if (x, y) != temp and x - 1 >= 0 and (x - 1, y) not in my_chess:
                valid_pos.add((x - 1, y))

            (x, y) = temp
            while (x + 1, y) in oppo_chess:
                x += 1
            if (x, y) != temp and x + 1 < self.board_size and (x + 1, y) not in my_chess:
                valid_pos.add((x + 1, y))

            (x, y) = temp
            while (x, y - 1) in oppo_chess:
                y -= 1
            if (x, y) != temp and y - 1 >= 0 and (x, y - 1) not in my_chess:
                valid_pos.add((x, y - 1))

            (x, y) = temp
            while (x, y + 1) in oppo_chess:
                y += 1
            if (x, y) != temp and y + 1 < self.board_size and (x, y + 1) not in my_chess:
                valid_pos.add((x, y + 1))

            (x, y) = temp
            while (x - 1, y - 1) in oppo_chess:
                x -= 1
                y -= 1
            if (x, y) != temp and x - 1 >= 0 and y - 1 >= 0 and (x - 1, y - 1) not in my_chess:
                valid_pos.add((x - 1, y - 1))

            (x, y) = temp
            while (x + 1, y + 1) in oppo_chess:
                x += 1
                y += 1
            if (x, y) != temp and x + 1 < self.board_size and y + 1 < self.board_size and (
            x + 1, y + 1) not in my_chess:
                valid_pos.add((x + 1, y + 1))

            (x, y) = temp
            while (x + 1, y - 1) in oppo_chess:
                x += 1
                y -= 1
            if (x, y) != temp and x + 1 < self.board_size and y - 1 >= 0 and (x + 1, y - 1) not in my_chess:
                valid_pos.add((x + 1, y - 1))

            (x, y) = temp
            while (x - 1, y + 1) in oppo_chess:
                x -= 1
                y += 1
            if (x, y) != temp and x - 1 >= 0 and y + 1 < self.board_size and (x - 1, y + 1) not in my_chess:
                valid_pos.add((x - 1, y + 1))
        return valid_pos

    def Reverse(self, last_pos, my_chess, oppo_chess, my_color):
        """每一次有一步新的落子后，把新的落子一方夹着的棋翻转；
            即把对方一部分棋归入己方
        Arguments:
            last_pos {tuple} -- 我方刚刚落子的位置
            my_chess {set} -- 当前我方的棋面，包含刚落子的位置
            oppo_chess {set} -- 当前对方的棋面
            my_color {int} -- 1表示黑色，-1表示白色
        """
        # print(last_pos)
        (x_, y_) = last_pos
        for (x, y) in my_chess:
            if (x, y) != (x_, y_):
                temp = []
                if x == x_:
                    temp = [(x, yy) for yy in range(min(y, y_) + 1, max(y, y_))]
                elif y == y_:
                    temp = [(xx, y) for xx in range(min(x, x_) + 1, max(x, x_))]
                elif x + y == x_ + y_:
                    const = x + y
                    temp = [(xx, const - xx) for xx in range(min(x, x_) + 1, max(x, x_))]
                elif x - y == x_ - y_:
                    const = x - y
                    temp = [(xx, xx - const) for xx in range(min(x, x_) + 1, max(x, x_))]

                if all(map(lambda para: para in oppo_chess, temp)):
                    temp = set(temp)
                    my_chess = my_chess.union(temp)
                    oppo_chess = oppo_chess.difference(temp)
        if my_color == 1:
            self.black_chess = my_chess
            self.white_chess = oppo_chess
        else:
            self.black_chess = oppo_chess
            self.white_chess = my_chess
        # 更新board！！！！！
        for (x, y) in self.black_chess:
            self.board[x][y] = 1
        for (x, y) in self.white_chess:
            self.board[x][y] = -1

    def Get_State(self):
        """返回当前棋盘的状态

        Returns:
            64维向量
        """
        return np.array(self.board, dtype=np.int).flatten()

    def Add(self, my_color, pos):
        """加入一个新的棋子

        Arguments:
            my_color {int} -- 1表示黑棋，-1表示白棋
            pos {int} -- 位置
        """
        if pos != 64:
            (x, y) = (pos // self.board_size, pos % self.board_size)
            self.board[x][y] = my_color
            if my_color == 1:
                self.black_chess.add((x, y))
                self.board[x][y] = 1
            elif my_color == -1:
                self.white_chess.add((x, y))
                self.board[x][y] = -1

            if my_color == 1:
                self.Reverse((x, y), self.black_chess, self.white_chess, my_color)
            elif my_color == -1:
                self.Reverse((x, y), self.white_chess, self.black_chess, my_color)

    def Display(self):
        for n in range(self.board_size):
            print('----', end='')
        print('')
        for m in range(self.board_size):
            for n in range(self.board_size):
                if self.board[m][n] == 1:
                    print("| x ", end='')
                elif self.board[m][n] == -1:
                    print("| o ", end='')
                else:
                    print("|   ", end='')
            print('|', m)
            for n in range(self.board_size):
                print('----', end='')
            print('')
        for n in range(self.board_size):
            print('  {} '.format(n), end='')
        print('\n\n')