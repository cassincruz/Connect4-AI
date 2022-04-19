import numpy as np
from connect4 import Connect4
# %%
class Bot :
    def __init__(self, weights=None) :
        if weights == None : self.weights = [np.random.random((20, 6*7)), np.random.random((7,20))]
        else: self.weights = weights
    
    def makemove(self, board) :
        layer = board.reshape(6*7)
        for weight in self.weights :
            layer = np.maximum(0, weight @ layer)

        layer += np.where(abs(game.board).sum(axis=0)==6,-np.inf,0) # For columns which are full replace with -inf to ensure they arent picked
        return layer.argmax()
        return layer.argmax()
# %% Training
#p1 = Bot() # Red
#p2 = Bot() # Blue

for epoch in range(1000) :
    game = Connect4()
    turns = 0
    while (game.winner == 0) & (turns < 6*7) :
        turns +=1
        game.move(p1.makemove(game.board))
        if game.winner != 0 : break
        game.move(p2.makemove(game.board))
    
    p1 = [p1, p1, p2][game.winner]
    #p2 = Bot(weights=[w+(np.random.random(w.shape)-0.5)/turns for w in p1.weights])
    p2 = basicbot()
    print(epoch)
# %% human play
game = Connect4()
while game.winner == 0 :
    game.move(int(input()))
    game.move(p2.makemove(game.board))
    game.draw()
    plt.show()
# %%
class basicbot :
    def __init__(self) :
        self.N = np.random.randint(0,7)
    def makemove(self, board) : return self.N

# %% Temporal Difference Learning
class TD_Bot :
    def __init__(self, policy=None, color=1) :
        if policy==None: self.policy = {}
        else: self.policy = policy

        self.color = color
        self.lastboard = None
    
    def value(self, board) :
        return self.policy[board.tostring()] if board.tostring() in self.policy.keys() else 0

    def makemove(self, board) :
        self.lastboard = board

        possible_moves = np.where(abs(board).sum(axis=0)<6)[0]
        move_scores = []
        
        #Exploratory moves: Some percent of the time, choose randomly
        if np.random.random()<0.05 : 
            print('Exploratory choice!')
            return np.random.choice(possible_moves)
        
        for move in possible_moves :
            newboard = board.copy()
            newboard[max(np.where(newboard[:,move]==0)[0]),move] = self.color
            move_scores.append(self.value(newboard))
        move_scores = np.array(move_scores)

        # Choose randomly from the choices that have the highest Value 
        return possible_moves[np.random.choice(np.where(move_scores==move_scores.max())[0])]        
    
    def update(self, board, reward=None):
        if reward : self.policy[board.tostring()] = reward
        
        learning_rate = 0.9
        self.policy[self.lastboard.tostring()] = self.policy.setdefault(self.lastboard.tostring(), 0) + learning_rate * (self.policy.setdefault(board.tostring(), 0) - self.policy.setdefault(self.lastboard.tostring(), 0))
        self.policy = {k:v for k, v in self.policy.items() if v!=0}

        self.lastboard = board
# %%
bot1 = TD_Bot(color=1)
bot2 = TD_Bot(color=-1)
training_game = Connect4()
# %%
training_game.move(bot1.makemove(training_game.board))
bot1.update(training_game.board)
# %%
