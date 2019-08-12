'''
Author: Eric P. Nichols
Date: Feb 8, 2008.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''
import numpy as np
class Board():

    # list of all 8 directions on the board, as (x,y) offsets
    __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self, r, c):
        "Set up initial board configuration."

        self.r = r
        self.c = c
        # Create the empty board array.
        self.pieces = np.zeros((r,c))
        #print(self.pieces)

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def countDiff(self, color):
        """Counts the # pieces more of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        count = 0
        for y in range(self.r):
            for x in range(self.c):
                if self[x][y]==color:
                    count += 1
                if self[x][y]==-color:
                    count -= 1
        return count
    
    def findFirstUnoccupied(self,x):
        #increment loop and return row value
        #print(self[5][6])
        for y in range(self.r-1,-1,-1):
            if self[y][x] == 0.0:
                return y
        return -1

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        moves = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for x in range(0,self.c,1):
            #print(x)
            y = self.findFirstUnoccupied(x)
            if y != -1:
                # print("returned")
                # print(x)
                # print(y)
                moves.add(x)
        # if len(moves) == 0:
        #     moves.add(self.c)
        return moves

    def has_legal_moves(self, color):
        if len(self.get_legal_moves(color)) != 0:
            return True
        else:
            return False


    def execute_move(self, move, color):
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """
        
        # Add the piece to the empty square.
        #print(move)
        move_x = move
        move_y = self.findFirstUnoccupied(move_x)
        if move_x!=self.c:
            self[move_y][move_x] = color
        
        #print(self.getNumInARow(color))
        
    # def isValid(self, x, y):
    #     for d in self.__directions:
    #         dx = d[0]
    #         dy = d[1]
    #         if (dx+x) < 0:
    #             return False
    #         elif dx+x >= self.c:
    #             return False
    #         if dy+y < 0:
    #             return False
    #         elif dy+y >= self.r:
    #             return False
    #     return True
    
    def isValid(self, r, c):
        if c < 0:
            return False
        elif c >= self.c:
            return False
        if r < 0:
            return False
        elif r >= self.r:
            return False
        return True

        
    
        
    # def getNumInARow(self, color,move):
    #     start = False
    #     greatest = 1
    #     temp = 0
    #     validD = None
    #     r = move[0]
    #     c = move[1]
    #     print("Last move was ",r,c)
    #     temp = 0
    #     for y in range(self.r):
    #         start = False
    #         if not start and self.isValid(y+1,c):
    #             i = y
    #             print("Move: ",i,c)
    #             if self[i][c] == color:
    #                 start = True
    #                 while self[i][c] == color and self.isValid(i+1,c):
    #                     temp += 1
    #                     i+=1
    #                     print("Move: ",i,c)
    #                 print("temp up down is ",temp)
    #     if temp>greatest:
    #         greatest = temp
    #     temp = 0
    #     for x in range(self.c):
    #         start = False
    #         if not start and self.isValid(r,x):
    #             i = x
    #             if self[r][i] == color:
    #                 start = True
    #                 while self[r][i] == color and self.isValid(r,i+1):
    #                     temp += 1
    #                     i+=1
    #                 print("temp side is ",temp)
    #     if temp>greatest:
    #         greatest = temp
    #     k = r
    #     l = c
    #     while self.isValid(k-1,l-1):
    #         k-=1
    #         l-=1
    #     temp = 0
    #     y = k
    #     for x in range(l,self.c):
    #         start = False
    #         #print(x,y)
    #         if not start and y<self.r:
    #             i = x
    #             j = y
    #             if self[j][i] == color:
    #                 start = True
    #                 while self[j][i] == color and self.isValid(j+1,i+1):
    #                     temp += 1
    #                     i+=1
    #                     j+=1
    #                 print("temp down right is ",temp)
    #         y+=1
    #     if temp>greatest:
    #         greatest = temp
    #     temp = 0
    #     k = r
    #     l = c
    #     while self.isValid(k+1,l+1):
    #         k+=1
    #         l+=1
    #     x = c
    #     y = r
    #     for x in range(self.c):
    #         start = False
    #         if not start and y<self.r:
    #             i = r
    #             j = c
    #             if self[i][j] == color:
    #                 start = True
    #                 print(i,j,self[i][j], color, self[i][j] == color)
    #                 while self[i][j] == color and self.isValid(i-1,j-1):
    #                     temp+=1
    #                     i-=1
    #                     j-=1
                        
    #     if temp>greatest:
    #         greatest = temp
            
    #     print("Num in a row is ",greatest)
    #     return greatest
    
    def get_win_state(self):
        for player in [-1, 1]:
            # Check rows & columns for win
            if self.get4InaRow(player):
                return (True, player)
        tie = True
        for player in [-1,1]:
            if len(self.get_legal_moves(player))!=0:
                tie = False
        if tie:
            return (True,None)
        else:
            return (False,None)

    def get4InaRow(self,color):
        
        #horizantal
        for i in range(self.r):
            for j in range(self.c-3):
                if self[i][j] == color and self[i][j+1] == color and self[i][j+2] == color and self[i][j+3] == color:
                    return color
        #vertical
        for i in range(self.r-3):
            for j in range(self.c):
                if self[i][j] == color and self[i+1][j] == color and self[i+2][j] == color and self[i+3][j] == color:
                    return color
        #up right diaganol
        for i in range(3,self.r):
            for j in range(self.c-3):
                #print(i,j)
                if self[i][j] == color and self[i-1][j+1] == color and self[i-2][j+2] == color and self[i-3][j+3] == color:
                    return color
        #down right diaganol
        for i in range(3,self.r):
            for j in range(3,self.c):
                if self[i][j] == color and self[i-1][j-1] == color and self[i-2][j-2] == color and self[i-3][j-3] == color:
                    return color
        return 0
                
                
                
    def getNumSameD(self, x, y, dx, dy, color):
        count = 2
        while self.isValid(dy+y, dx+x) and self[dy+y][dx+x] == color:
            count +=1
            x = dx+x
            y = dy+y
        return count
        
    def getLastMove(self, color):
        #print(color)
        if color == 1:
            return self.lastMoveW
        else:
            return self.lastMoveB
            
    def getR(self):
        return self.r
    def getC(self):
        return self.c
        

    @staticmethod
    def _increment_move(move, direction, n):
        # print(move)
        """ Generator expression for incrementing moves """
        move = list(map(sum, zip(move, direction)))
        #move = (move[0]+direction[0], move[1]+direction[1])
        while all(map(lambda x: 0 <= x < n, move)): 
        #while 0<=move[0] and move[0]<n and 0<=move[1] and move[1]<n:
            yield move
            move=list(map(sum,zip(move,direction)))
            #move = (move[0]+direction[0],move[1]+direction[1])

