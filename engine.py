from keras.models import Model
from keras.layers import Input,Reshape,Activation,BatchNormalization,Conv2D,Flatten,Dropout,Dense
from keras.optimizers import Adam
from random import choice
import numpy as np
from math import log
from Connect4Game import Connect4Game as game
class BetaZero:

    def __init__(self):
        self.g = game(6,7)
        self.game_num = 0
        self.model = self.get_model()
        self.board = np.zeros((self.r,self.c))
        self.actions = []*self.g.getActionSize()

    def get_beta(self):
        return 1-(log(1+self.game_num)/10.0)

    def get_move(self):
        rand = self.get_beta()
        r = np.random.choice([True,False],None,p=[rand,1-rand])
        if r:
            return self.choose_rand_move()
        else:
            return self.choose_nn_move()

    def choose_rand_move(self):
        # print(self.actions)
        valids = []
        for i in range(len(self.actions)):
            if self.actions[i] == 1:
                valids.append(i)
        # print(valids)
        # print(choice(valids))
        return choice(valids)

    def choose_nn_move(self):
        board = np.expand_dims(self.board,axis=0)
        pi = self.model.predict(board)
        return np.argmax(pi)

    def set_board(self,board,actions):
        self.board = board
        self.actions = actions

    def set_game_num(self,n):
        self.game_num = n

    def get_model(self):
        dims = self.g.getBoardSize()
        self.r = dims[0]
        self.c = dims[1]
        action_size = self.g.getActionSize()
        num_c = 512
        dropout = .3
        lr = .001
        # Neural Net
        input_boards = Input(shape=(self.r, self.c))    # s: batch_size x board_r x board_c

        x_image = Reshape((self.r, self.c, 1))(input_boards)                # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('elu')(BatchNormalization(axis=3)(Conv2D(num_c, 3, padding='same', use_bias=False)(x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('elu')(BatchNormalization(axis=3)(Conv2D(num_c, 3, padding='same', use_bias=False)(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('elu')(BatchNormalization(axis=3)(Conv2D(num_c, 3, padding='valid', use_bias=False)(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('elu')(BatchNormalization(axis=3)(Conv2D(num_c, 3, padding='valid', use_bias=False)(h_conv3)))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(dropout)(Activation('elu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(dropout)(Activation('elu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))          # batch_size x 1024
        pi = Dense(action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size

        model = Model(inputs=input_boards, outputs=pi)
        model.compile(loss='mean_squared_logarithmic_error', optimizer=Adam(lr))
        return model