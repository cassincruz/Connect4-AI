#!/usr/bin/env python3
#%% Importing packages
import numpy as np
from scipy.signal import convolve2d
from matplotlib.pyplot import imshow
#%%
class Connect4 :
    def __init__(self,draw=False) :
        self.board = np.zeros((6,7))
        self.current_color = 1 # 1=red, -1=blue, 0=empty
        self.available_moves = range(7)
        self.winner = 0
        self.settings = {'Draw':draw}
    
    def move(self,col) : 
        '''
        Currently illegal moves simply result in the turn being skipped, however its possible to abuse this strategically and should be fixed later
        '''
        if col in self.available_moves : # Check to ensure that column isnt full
            self.board[max(np.where(self.board[:,col]==0)[0]),col] = self.current_color
            if self.settings['Draw']==True: plt.imshow(self.board)
            self.check()
            self.current_color = -self.current_color
        else :
            pass
    
    def check(self) :
        patterns = [np.ones((1,4)), np.ones((4,1)), np.diag(np.ones(4)), np.fliplr(np.diag(np.ones(4)))]
        if True in [convolve2d(self.board, pattern, 'same').max()>=4 for pattern in patterns] : 
            #print('Player 1 wins!')
            self.winner = 1
        if True in [convolve2d(self.board, pattern, 'same').min()<=-4 for pattern in patterns] : 
            #print('Player 2 wins!')
            self.winner = -1
        
        self.available_moves = [n for n in range(7) if sum(abs(self.board[:,n])) < 6]
    
    def draw(self) : imshow(game.board)
# %%
