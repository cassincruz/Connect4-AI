#%% Importing packages
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
#%%
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
            self.players = [Agent(), Agent()]
        
        elif len(players) == 1 :
            self.players = (players[0], players[0])
        
        elif len(players) == 2 : 
            self.players = (players[0], players[1])

        else : 
            if self.settings['Print'] : print("Cannot have more than two players.") 
            
        self.board = np.zeros((6,7))
        self.available_moves = np.arange(7)
        
        self.board_record = [self.board]
        self.move_record = np.array([])
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
        self.board_record.append(self.board)
        
        # Check for 4 in a row using convolution
        patterns = [np.ones((1,4)), np.ones((4,1)), np.diag(np.ones(4)), np.fliplr(np.diag(np.ones(4)))]

        if True in [convolve2d(self.board, pattern * self.current_color, 'same').max()>=4 for pattern in patterns] : 
            
            self.winner = self.current_color
            
            if self.settings['Print']: 
                print(f'Player {self.winner} wins in {self.move_number} moves.')
            
        # Updating game attributes
        else :
            self.available_moves = [n for n in range(7) if sum(abs(self.board[:,n])) < 6]
            self.move_number += 1
            self.move_record = np.append(self.move_record, col)
            self.current_color *= -1

    
    #TODO
    # Play a sequence of moves and return array of boards. 
    def play_moves(self, moves) :
        pass

    
    # Plays a game between two players. Returns move record and the color code of the winner (1 for P1, -1 for P2, 0 for Draw).
    def play_game(self) : 
        
        while (self.winner==0) & (len(self.available_moves)>0) & (self.move_number < self.settings['Max move']) :
            
            if self.settings['Draw']: self.draw()
            
            player = self.players[self.move_number % 2]
            move_choice = player.submit_move(self)
            self.play_move(move_choice)
            
        if len(self.available_moves)==0 : 
            if self.settings['Print'] : print("It's a draw!")
            return 0
        
        return self.winner
    
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
            self.engine = lambda board : 1.0 # Default to constant value, to produce random moves
        else : 
            self.engine = engine
        
        if prob == True :
            normalize = lambda v : v/v.sum()
            self.prob = lambda vals : normalize((vals+1)/2)
        elif prob == False : 
            self.prob = lambda vals : vals == vals.max()
        else :
            self.prob = prob

    def evaluate(self, game): # From the current board position, calculates the value of each of the available moves
        return np.array([self.engine(game.move(c)) for c in game.available_moves])

    def submit_move(self, game) :
        values = self.evaluate(game)
        
        return np.random.choice(game.available_moves, p=self.prob(values))
# %%
