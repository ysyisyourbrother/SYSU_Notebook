import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Othello import *

"""
黑白棋，DQN, 
各种超参数
"""
BOARD_SIZE = 8
N_STATE = pow(BOARD_SIZE, 2)
N_ACTION = pow(BOARD_SIZE, 2) + 1

LR = 0.001
EPISODE = 10000
BATCH_SIZE = 32
GAMMA = 0.9
ALPHA = 0.8
TRANSITIONS_CAPACITY = 200
UPDATE_DELAY = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NET(nn.Module):
    """定义网络结构

    Returns:
        x [tensor] -- (batch, N_ACTION)，每一行表示各个action的分数
    """

    def __init__(self):
        super(NET, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(N_STATE, 128),
            nn.LeakyReLU()
        )
        # self.linear1.weight.data.normal_(0, 0.1)

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 4, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(4, 8, 3, 1, 1),
            nn.LeakyReLU()
        )

        self.linear2 = nn.Sequential(
            nn.Linear(8 * 128, N_ACTION)
        )

    def forward(self, x):
        x = self.linear1(x)
        x = x.view(x.shape[0], 1, -1)
        x = self.conv1(x)
        x = self.conv2(x)
        # x = x.flatten()
        x = x.view(x.shape[0], -1)
        x = self.linear2(x)
        return x


class DQN(object):
    def __init__(self, color):
        """
        color: 1表示先手；-1表示后手

        transitions : 存储状态的空间，格式为(state, action, reward, state_), state_为后继状态
        transitions_index : 记录当前使用存储空间的索引
        learn_iter : 当到达UPDATE_ITERS时，就更新预测网络 Q_ ，把Q的参数复制给它
        """
        self.transitions = np.zeros((TRANSITIONS_CAPACITY, 2 * N_STATE + 2))
        self.transitions_index = 0
        self.learn_iter = 0

        self.Q, self.Q_ = NET(), NET()
        # if color == 1:
        #     self.Q.load_state_dict(torch.load('model_offensive.pth'))
        # elif color == -1:
        #     self.Q.load_state_dict(torch.load('model_defensive.pth'))

        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=LR)
        self.criteria = nn.MSELoss()

    def Choose_Action_EpsilonGreedy(self, x, game_state, color, Epsilon=0.1):
        """ε-greedy算法选择下一个action。以ε概率随机选择一个action，否则就选择Q值最大的action

        Arguments:
            x [tensor] -- NET网络的输入值，即当前状态，在Q-Learning中，选择下一个动作应该是查表得到的，
                            在DQN中没有这个表，所以要先经过Q网络得到一个状态的Q值，然后选择这向量里概率最大的action
            game_state [class] -- 当前的游戏状态
            color int -- 1表示黑棋，-1表示白棋

        Returns:
            action [int] -- 0~64中的一个数，表示下棋的位置；64表示跳过
        """

        if color == 1:
            avaliable_pos = game_state.Get_Valid_Pos(game_state.black_chess, game_state.white_chess)
        elif color == -1:
            avaliable_pos = game_state.Get_Valid_Pos(game_state.white_chess, game_state.black_chess)

        avaliable_pos = list(map(lambda a: game_state.board_size * a[0] + a[1], avaliable_pos))  # 列表,表明合法位置
        if len(avaliable_pos) == 0:
            return 64  # 表示这一步只能跳过

        if np.random.uniform() < Epsilon:  # random choose an action
            action = np.random.choice(avaliable_pos, 1)[0]
        else:  # choose the max Q-value action
            x = torch.tensor(x, dtype=torch.float)
            x = x.view(1, -1)
            actions_values = self.Q(x)[0]  # 65维tensor，各个action在各个位置的值

            ava_actions = torch.tensor(actions_values[avaliable_pos])

            _, action_ind = torch.max(ava_actions, 0)
            action = avaliable_pos[action_ind]
        return action

    def Store_transition(self, s, a, r, s_):
        """把一组转移属性存储到transitions中

        Arguments:
            s {[type]} -- 当前状态
            a {[type]} -- 选择的动作
            r {[type]} -- reward值
            s_ {[type]} -- 后继状态
        """
        transition = np.hstack((s, a, r, s_))
        self.transitions[self.transitions_index % TRANSITIONS_CAPACITY] = transition
        self.transitions_index += 1

    def Learn(self, oppo_Q_):
        for step in range(10):
            if self.learn_iter % UPDATE_DELAY == 0:  # update parameters of Q_ 每隔一段时间将Q的参数直接给到Q_
                self.Q_.load_state_dict(self.Q.state_dict())
            self.learn_iter += 1

            sample_index = np.random.choice(TRANSITIONS_CAPACITY,
                                            BATCH_SIZE)  # randomly choose BATCH_SIZE samples to learn 从经验池中随机选取进行训练
            batch_tran = self.transitions[sample_index, :]
            batch_s = batch_tran[:, :N_STATE]
            batch_a = batch_tran[:, N_STATE: N_STATE + 1]
            batch_r = batch_tran[:, N_STATE + 1: N_STATE + 2]
            batch_s_ = batch_tran[:, N_STATE + 2:]

            batch_s = torch.tensor(batch_s, dtype=torch.float)
            batch_s_ = torch.tensor(batch_s_, dtype=torch.float)
            batch_a = torch.tensor(batch_a, dtype=int)
            batch_r = torch.tensor(batch_r, dtype=torch.float)

            batch_y = self.Q(batch_s).gather(1,
                                             batch_a)  # gather figure out which action actually is chosen 相当于从第一维取第batch_a位置的值
            batch_y_ = oppo_Q_(
                batch_s_).detach()  # detach return a new Variable which do not have gradient detach就是禁止梯度更新，这些图变量包含了梯度，在计算loss的时候会更新，因为Q_不用更新，因此禁止梯度。
            batch_y_ = batch_r - GAMMA * torch.max(batch_y_, 1)[0].view(-1,
                                                                        1)  # max(1) return (value,index) for each row

            loss = self.criteria(batch_y, batch_y_)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    offensive = DQN(1)
    defensive = DQN(-1)

    for episode in range(EPISODE * 50):
        game_state = Game()
        round_ = 0
        while True:
            # 先手
            # print(round_)
            round_ += 1
            # game_state.Display()    # 输出棋盘
            s = game_state.Get_State()
            a = offensive.Choose_Action_EpsilonGreedy(s, game_state, 1)
            game_state.Add(1, a)
            r = game_state.Gameover() * 100.0
            s_ = game_state.Get_State()

            offensive.Store_transition(s, a, r, s_)   # 先后手的经验池分开存
            # defensive.Store_transition(s, a, -r, s_)

            if r != 0 or round_ > 100:  # 当这局游戏结束或双方下够了100次。经验池已经有很多样本，此时可以开始训练
                offensive.Learn(defensive.Q_) # 用对手的Q_网络来计算下一个状态
                # print("END~~~~~")
                # game_state.Display()
                # print("==================================")
                print('Episode:{} | Reward:{}'.format(episode, r))
                break

            # 后手
            # game_state.Display()
            s = game_state.Get_State()
            a = defensive.Choose_Action_EpsilonGreedy(s, game_state, -1)
            game_state.Add(-1, a)
            r = game_state.Gameover() * 100.0
            s_ = game_state.Get_State()

            # offensive.Store_transition(s, a, r, s_)
            defensive.Store_transition(s, a, -r, s_) # 先后手的经验池分开存

            if r != 0:
                defensive.Learn(offensive.Q_) # 用对手的Q_网络来计算下一个状态
                # print("END~~~~~")
                # game_state.Display()
                # print("==================================")
                print('Episode:{} | Reward:{}'.format(episode, r))
                break

        if (episode + 1) % 100 == 0:
            torch.save(offensive.Q.state_dict(), 'model_offensive.pth')
            torch.save(defensive.Q.state_dict(), 'model_defensive.pth')