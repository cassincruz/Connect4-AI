#%% Importing packages
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import pandas as pd
import torch
from torch.utils.data import Dataset

#%% Defining core Connect4Game class
class Connect4Game :

    def __init__(self, *players, draw=False, print_results=False) :
        
        self.settings = {'Draw':draw, 'Print':print_results, 'Max move':43}
        
        self.reset(*players)
        
        if self.settings['Draw'] : #TODO
            #plt.ion()
            self.fig, self.ax = plt.subplots()
            self.img = self.ax.imshow(self.board)
            plt.show()
            
    def reset(self, *players) :
    # Resets the game board and move record to begin new game between (optionally) given players.
    # If one player given, player plays against itself. If no players given, a random game is played.
        if len(players) == 0 :
            if hasattr(self, 'players') : pass
            else: self.players = [Agent(), Agent()]
        
        elif len(players) == 1 :
            self.players = (players[0], players[0])
        
        elif len(players) == 2 : 
            self.players = (players[0], players[1])

        else : 
            if self.settings['Print'] : print("Cannot have more than two players.") 
            
        self.board = np.zeros((6,7)).astype(int)
        self.available_moves = np.arange(7)
        
        self.board_record = np.array([self.board])
        self.move_record = np.array([]).astype(int)
        self.move_number = 1
        self.current_color = 1 # 1=P1, -1=P2, 0=empty
        
        self.winner = 0 # 1=P1 Wins, -1=P2 Wins, 0 = Draw
    
    def move(self, col) : 
    # Returns the board if the current player plays move at given col. Does not update game variables.
        if col in self.available_moves : 
            board = self.board.copy()
            board[max(np.where(board[:,col]==0)[0]),col] = self.current_color
            return board
        
        else: 
            if self.settings['Print']: print('Illegal move!')
            return self.board
    
    
    # Current color plays move at position 'move'. Updates board, move record, etc.
    def play_move(self, col) : 
        self.board = self.move(col)

        self.board_record = np.vstack((self.board_record, [self.board]))
        self.move_record = np.append(self.move_record, col)
        self.available_moves = [n for n in range(7) if sum(abs(self.board[:,n])) < 6]
        
        # Check for 4 in a row using convolution
        patterns = [np.ones((1,4)), np.ones((4,1)), np.diag(np.ones(4)), np.fliplr(np.diag(np.ones(4)))]

        if True in [convolve2d(self.board, pattern * self.current_color, 'same').max()>=4 for pattern in patterns] : 
            
            self.winner = self.current_color
            
            if self.settings['Print']: 
                print(f'Player {self.winner} wins in {self.move_number} moves.')
            
        # Updating game attributes
        else :
            self.move_number += 1
            self.current_color *= -1

    # Play a sequence of moves.
    def play_moves(self, moves) :
        for move in moves : 
            self.play_move(int(move))
    
    # Plays a game between two players. Returns move record and the color code of the winner (1 for P1, -1 for P2, 0 for Draw).
    def play_game(self) : 
        
        while (self.winner==0) & (len(self.available_moves)>0) & (self.move_number < self.settings['Max move']) :
            
            if self.settings['Draw']: self.draw()
            
            player = self.players[self.move_number % 2]
            move_choice = player.submit_move(self)
            self.play_move(move_choice)
            
        if len(self.available_moves)==0 : 
            if self.settings['Print'] : print("It's a draw!")
        
        return self.winner
    
    #TODO
    def draw(self) : 
        #self.ax.imshow(self.board)
        self.img.set_data(self.board)
        self.fig.canvas.draw()
        plt.show()

# %%
class Agent :
# Takes a game board and returns a move between 0-6.
# The Engine is a function which is used to evaluate board quality. If none is passed to __init__, then the Agent will move randomly.
# Option to choose move probabilistically or always choose move with maximum calculated value
# Prob should take list of values and return list of values corresponding to probability of selecting nth option. E.g. [-0.2,0.9,0.1] -> [0.1,0.7,0.2] => 70% chance of choosing second available game move. 
# If prob=True, probability of selection will be proportional to value (e.g. -1 -> 0%, 1 -> 100%, with options normalized).
# If a function is passed to prob with codomain [0,1] (e.g. softmax), then this function will be applied to the calculated Values to determine probabilities
    def __init__(self, engine=None, prob=True): 
        
        if engine == None :
            self.engine = lambda board : 0.0 # Default to constant value, to produce random moves
        else : 
            self.engine = engine
        
        normalize = lambda v : v/v.sum()
        if prob == True :
            self.prob = lambda vals : normalize((vals+1)/2)
        elif prob == False : 
            self.prob = lambda vals : normalize(vals == vals.max())
        else :
            self.prob = prob

    def evaluate(self, game): # From the current board position, calculates the value of each of the available moves
        return np.array([self.engine(game.move(c))*game.current_color for c in game.available_moves])

    def submit_move(self, game) :
        values = self.evaluate(game)
        
        return np.random.choice(game.available_moves, p=self.prob(values))

#%%
class C4Data(Dataset) : 
    
    def __init__(self, games, winners, Lambda=0.9, T=lambda x:torch.Tensor(x)) : 
        # games should be an array of move records. Winners should be an array of game winners.
        self.T = T
        self.read_data(games, winners, Lambda)
        
    def __len__(self) :
        # Returns the total number of samples
        return self.Data.shape[0]
    
    def __getitem__(self, index) :
        # Generates one sample of data
        board = ID2Board(self.Data.ID.iloc[index])
        
        X = self.T(board) 
        y = self.Data.Value.iloc[index]
        return X, y
    
    def read_data(self, games, winners, Lambda) :
        # games should be a list of move records. 
        # winners should be a list of winners (1, -1, or 0 for draw).
        # Lambda should be a positive number between 0 and 1.
        IDs, Values = [], []

        C4 = Connect4Game()
        for i in range(len(winners)) :
            moves = games[i]
            result = winners[i]

            C4.reset()
            C4.play_moves(moves)

            # Adding additional boards for LR and inversion symmetries (x4)
            boards = C4.board_record.copy()
            boards = np.vstack((boards, [np.fliplr(board) for board in boards]))
            boards = np.vstack((boards, [-board for board in boards]))
            
            game_IDs = [Board2ID(board) for board in boards]
            IDs.extend(game_IDs)

            game_values = result * Lambda**np.arange(len(moves), -1, -1)
            Values.extend(game_values.tolist()*2 + (-game_values).tolist()*2) 
        
        data = pd.DataFrame(data={'ID':IDs, 'Value':Values})
        self.Data = data.groupby('ID').Value.mean().reset_index()
#%% ID functions
Board2ID = lambda board: ''.join(board.reshape(-1).astype(int).astype(str).tolist()).replace('-1', '2')
ID2Board = lambda ID: np.array([int(n) if n !='2' else -1 for n in ID]).reshape((6,7))
# %%
