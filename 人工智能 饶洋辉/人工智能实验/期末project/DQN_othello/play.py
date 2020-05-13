from Othello import *
from DQN import *
import numpy as np
import pygame
import time

N = 8
background = pygame.image.load(r'python\AI_Lecture\project2\board.jpg')
WIDTH = 400
BOX = WIDTH / (8 + 2)
screen = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption('你是赢不了我的')


def show_board(board, last_pos=0, valid=[], surf=screen):
    surf.blit(background, (0, 0))

    BOUNDS = [((BOX, BOX), (BOX, WIDTH - BOX)),
              ((WIDTH - BOX, BOX), (BOX, BOX)),
              ((WIDTH - BOX, WIDTH - BOX), (WIDTH - BOX, BOX)),
              ((WIDTH - BOX, WIDTH - BOX), (BOX, WIDTH - BOX))]
    for line in BOUNDS:
        pygame.draw.line(surf, (0, 0, 0), line[0], line[1], 1)
    for i in range(N - 1):
        pygame.draw.line(surf, (0, 0, 0),
                         (BOX * (2 + i), BOX),
                         (BOX * (2 + i), WIDTH - BOX))
        pygame.draw.line(surf, (0, 0, 0),
                         (BOX, BOX * (2 + i)),
                         (WIDTH - BOX, BOX * (2 + i)))

    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 1:
                t = (int((j + 1.5) * BOX), int((i + 1.5) * BOX))
                pygame.draw.circle(surf, (0, 0, 0), t, int(BOX / 3))
            if board[i][j] == -1:
                t = (int((j + 1.5) * BOX), int((i + 1.5) * BOX))
                pygame.draw.circle(surf, (255, 255, 255), t, int(BOX / 3))

    for (x, y) in valid:
        t = (int((y + 1.5) * BOX), int((x + 1.5) * BOX))
        pygame.draw.circle(surf, (0, 255, 0), t, int(BOX / 3))

    if isinstance(last_pos, tuple):
        (x, y) = last_pos
        t = (int((y + 1.5) * BOX), int((x + 1.5) * BOX))
        pygame.draw.circle(surf, (255, 0, 0), t, int(BOX / 8))

    pygame.display.flip()


if __name__ == "__main__":
    me_first = 1  # 1表示ai先手；-1表示人先手

    ai = DQN(me_first)

    game = Game()

    running = True

    if me_first:
        step = 1
        ai_color = 1
        human_color = -1
    else:
        step = 0
        ai_color = -1
        human_color = 1

    grid = 0
    while running:
        # show_board(game.board)

        # 人走
        if human_color == 1:
            valid_pos = game.Get_Valid_Pos(game.black_chess, game.white_chess)
        else:
            valid_pos = game.Get_Valid_Pos(game.white_chess, game.black_chess)

        show_board(game.board, grid, valid_pos)

        while step % 2 == 0:
            if len(valid_pos) == 0:
                time.sleep(1)
                step += 1
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    step += 1
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    grid = (int(event.pos[1] / (BOX + .0) - 1), int(event.pos[0] / (BOX + .0) - 1))
                    # print(grid)
                    if grid[0] >= 0 and grid[0] < 8 and grid[1] >= 0 and grid[1] < 8:
                        # print('hahahahaha')
                        if grid in valid_pos:
                            # print('yes~~~~:', a, grid)
                            a = N * grid[0] + grid[1]
                            game.Add(human_color, a)
                            show_board(game.board, grid)
                            time.sleep(1)
                            step += 1
                            break

        # 电脑走
        begin_time = time.time()

        s = game.Get_State()
        a = ai.Choose_Action_EpsilonGreedy(s, game, ai_color, 0)
        game.Add(ai_color, a)
        grid = (a // N, a % N)
        show_board(game.board, grid)
        step += 1

        end_time = time.time()
        cal_time = end_time - begin_time
        print('calculating time: %.4f' % cal_time)