import numpy as np
from Connect4Game import Connect4Game
from engine import BetaZero
import keras

p1 = BetaZero()
p1.model = keras.models.load_model("checkpoint_p1")
game = Connect4Game(6,7)
board = game.getInitBoard()
player = 1
while game.getGameEnded(board,player) == 0:
    actions = game.getValidMoves(board,player)
    if player == 1:
        print(board)
        print(actions)
        move = int(input())
    else:
        p1.set_board(board,actions)
        move = p1.choose_nn_move()
    next_state = game.getNextState(board, player, move)
    board = next_state[0]
    player = next_state[1]