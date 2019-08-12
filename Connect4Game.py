from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from Connect4Logic import Board
import numpy as np


class Connect4Game(Game):
    def __init__(self, r, c):
        self.r = r
        self.c = c
        if r>c:
            self.n = r
        else:
            self.n = c

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.r, self.c)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.r, self.c)

    def getActionSize(self):
        # return number of actions
        return self.c

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        #print(action)
        if action == self.c:
            return (board, -player)
        b = Board(self.r, self.c)
        b.pieces = np.copy(board)
        move = (action)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.r, self.c)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(player)
        #print(legalMoves)
        for c  in legalMoves:
            #print(self.n*c+r)
            valids[c]=1
        #print(valids)
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.r, self.c)
        b.pieces = np.copy(board)
        winstate = b.get_win_state()
        if winstate[0]:
            if winstate[1] is None:
                # draw has very little value.
                return -.5
            elif winstate[1] == player:
                return +1
            elif winstate[1] == -player:
                return -1
            else:
                raise ValueError('Unexpected winstate found: ', winstate)
        else:
            # 0 used to represent unfinished game.
            return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board


    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()

    # def getScore(self, board, player):
    #     b = Board(self.r, self.c)
    #     b.pieces = np.copy(board)
    #     return b.countDiff(player)
    
    def getSymmetries(self, board, pi):
        return [(board, pi), (board[:, ::-1], pi[::-1])]

    def getPlayerI(self, curPlayer):
        return curPlayer+1

def display(board):
    #print(board.shape)
    n = board.shape[0]
    m = board.shape[1]
    print("   ",end="")
    for y in range(board.shape[1]):
        print (y,"|",end="")
    print("")
    print(" -----------------------")
    for y in range(n):
        print(y, "|",end="")    # print the row #
        for x in range(board.shape[1]):
            #print(x)
            piece = board[y][x]    # get the piece to print
            if piece == -1: print("X  ",end="")
            elif piece == 1: print("O  ",end="")
            else:
                if x==board.shape[1]:
                    print("-",end="")
                else:
                    print("-  ",end="")
        print("|")

    print("   -----------------------")

# def display(board):
#     print(" -----------------------")
#     print(' '.join(map(str, range(len(board[0])))))
#     print(board)
#     print(" -----------------------")
