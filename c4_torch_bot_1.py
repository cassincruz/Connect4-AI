#%% Importing torch modules
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
# %% Importing Connect4 module
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from connect4 import Connect4

#%%
class bot1(nn.Module) :
    def __init__(self, color) : #color can be -1 or 1
        super().__init__()
        self.color = color
        self.evaluate = nn.Sequential(
            nn.Linear(6*7, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32,7), 
            nn.Sigmoid()
        )

    def eval(self, board) :
        return self.evaluate(torch.Tensor(self.color * board.reshape(6*7)))

    def makemove(self, board) : # select move probabilistically
        alpha = 2
        available_moves = np.where(abs(board).sum(axis=0)<6)[0]
        #self.eval = self.evaluate(torch.Tensor(self.color * board.reshape(6*7)))
        return int((self.eval(board)[available_moves]**alpha).multinomial(1)[0])

# %% Playing against itself
n_games = 10000
bot = bot1(1)

lr = 3e-4
optimizer = optim.Adam(bot.parameters(), lr=lr)
#%% 
for game_idx in range(n_games) :
    boards = []
    moves = []
    
    game = Connect4()
    n_turn = 0

    while (game.winner == 0) & (n_turn <= 42) :
        bot.color = game.current_color
        move_choice = bot.makemove(game.board)
        game.move(move_choice)

        boards.append(game.board)
        moves.append(move_choice)
        n_turn += 1

    if game.winner != 0 :
        # Training net on game outcome
        lam = 0.9 # how fast "true" board value drops off. Lower values -> faster dropoff
        beta = 0.8 # 
        
        N_steps = 10
        for step in range(N_steps) :
            Loss = 0
            for turn in range(n_turn) :
                L_win = (bot.eval(boards[turn])[moves[turn]]-lam**(n_turn-turn))**2
                L_lose = (bot.eval(-boards[turn])[moves[turn]]+lam**(n_turn-turn))**2
                Loss += beta**(n_turn-turn)*(L_win + L_lose)
                
            bot.zero_grad()
            Loss.backward()
            optimizer.step()

        print(Loss)
#%% Batching games 
#%% 
batch_size = 20
for game_idx in range(n_games) :
    boards = []
    moves = []
    
    game = Connect4()
    n_turn = 0

    while (game.winner == 0) & (n_turn <= 42) :
        bot.color = game.current_color
        move_choice = bot.makemove(game.board)
        game.move(move_choice)

        boards.append(game.board)
        moves.append(move_choice)
        n_turn += 1

    if game.winner != 0 :
        # Training net on game outcome
        lam = 0.9 # how fast "true" board value drops off. Lower values -> faster dropoff
        beta = 0.8 # 
        
        N_steps = 10
        for step in range(N_steps) :
            Loss = 0
            for turn in range(n_turn) :
                L_win = (bot.eval(boards[turn])[moves[turn]]-lam**(n_turn-turn))**2
                L_lose = (bot.eval(-boards[turn])[moves[turn]]+lam**(n_turn-turn))**2
                Loss += beta**(n_turn-turn)*(L_win + L_lose)
                
            bot.zero_grad()
            Loss.backward()
            optimizer.step()

        print(Loss)

# %% Monte Carlo Tree Search
import numpy as np
import random
import matplotlib.pyplot as plt 

class MCTS : 
    def __init__(self, color, N=100) : #color can be -1 or 1
        self.color = color
        self.N = N

    def evaluate(self, board) :
        self.move_scores = []
        self.available_moves = np.where(abs(board).sum(axis=0)<6)[0]
        
        for n in self.available_moves :
            val = 0
            for i in range(self.N) :
                g = Connect4()
                g.board = board.copy()
                g.current_color = self.color

                n_turn = 0
                g.move(n)
                while (g.winner == 0) & (n_turn <=42) :
                    move = random.choice(self.available_moves)
                    g.move(move)
                    n_turn += 1
                val += g.winner * self.color
            self.move_scores.append(val)
        
        choice = np.argmax(self.move_scores)
        return self.available_moves[choice]
    
# %%
