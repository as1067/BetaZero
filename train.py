import numpy as np
from Connect4Game import Connect4Game
from engine import BetaZero

p1 = BetaZero()
p2 = BetaZero()

game = Connect4Game(6,7)
board = game.getInitBoard()
player = 1

p1_boards = []
p1_actions = []

p2_boards = []
p2_actions = []
winners = []
p1w = 0
p2w = 0
try:
    for i in range(100):
        for j in range(100):
            p1b = []
            p1a = []
            p2b = []
            p2a = []
            while game.getGameEnded(board,player)==0:
                actions = game.getValidMoves(board, player)
                if player == 1:
                    p1.set_board(board,actions)
                    move = p1.get_move()
                    p1b.append(np.asarray(board))
                    a = [0]*game.getActionSize()
                    a[move] = 1
                    p1a.append(np.asarray(a,dtype="float32"))
                else:
                    p2.set_board(board,actions)
                    p2b.append(np.asarray(board))
                    move = p2.get_move()
                    a = [0]*game.getActionSize()
                    a[move] = 1
                    p2a.append(np.asarray(a,dtype="float32"))
                # print(move)
                next_state = game.getNextState(board,player,move)
                board = next_state[0]
                player = next_state[1]
            # print("Player: "+str(player))
            # print("Winner: "+str(game.getGameEnded(board,player)))
            # print(board)
            winner = player*game.getGameEnded(board,player)
            winners.append(winner)
            if winner == 1:
                p1_boards.extend(p1b)
                p1_actions.extend(p1a)
                p1w+=1
            elif winner == -1:
                p2_boards.extend(p2b)
                p2_actions.extend(p2a)
                p2w+=1
            board = game.getInitBoard()
        print(i)
        p1.model.fit(x=[p1_boards],y=[p1_actions],epochs=10)
        print(i)
        p2.model.fit(x=[p2_boards],y=[p2_actions],epochs=10)
        print(i)
        p1.set_game_num(p1w)
        p2.set_game_num(p2w)
except(KeyboardInterrupt):
    p1.model.save("checkpoint1_p1")
    p2.model.save("checkpoint1_p2")
p1.model.save("checkpoint1_p1")
p2.model.save("checkpoint1_p2")