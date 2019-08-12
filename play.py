import numpy as np
from Connect4Game import Connect4Game
from engine import BetaZero
import keras

p1 = BetaZero()
p1.model = keras.models.load_model("checkpoint1_p1")
p2 = BetaZero()
p2.model = keras.models.load_model("checkpoint1_p2")
game = Connect4Game(6,7)
board = game.getInitBoard()
player = 1
while game.getGameEnded(board,player) == 0:
    actions = game.getValidMoves(board,player)
    if player == 1:
        print(board)
        p1.set_board(board,actions)
        move = p1.choose_nn_move()
    else:
        print(board)
        p2.set_board(board,actions)
        move = p2.choose_nn_move()
    next_state = game.getNextState(board, player, move)
    board = next_state[0]
    player = next_state[1]
print(board)