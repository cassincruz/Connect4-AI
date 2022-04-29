
#%%
from connect4 import *

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm, trange

#%% Simple Perceptron Neural Network
# Creating NNs
SimpleNN = nn.Sequential(nn.Linear(6*7, 120), 
              nn.ReLU(), 
              nn.Linear(120, 84), 
              nn.ReLU(), 
              nn.Linear(84,10), 
              nn.ReLU(), 
              nn.Linear(10,1),
              nn.Sigmoid())

T = lambda board : torch.tensor(board.reshape(6*7)).float()

engine = lambda x : float(SimpleNN(T(x)))
Comp = Agent(engine=engine)


#%%
def selfplay(comp, n_games) :
    Game = Connect4Game(comp)
    games = []
    winners = []

    for _ in trange(n_games, desc="Playing games.") :
        Game.reset()
        Game.play_game()

        games.append(Game.move_record)
        winners.append(Game.winner)

    return C4Data(games, winners, T=T)

#%%
def train(data, n_epochs) :
    training_generator = DataLoader(data, **loader_params)

    losses = np.array([])
    for epoch in trange(n_epochs, desc="Training on game records.") :
        running_loss = 0.0
        
        for batch, labels in training_generator : 
            batch, labels  = batch.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = SimpleNN(batch)
            loss = criterion(outputs, labels.reshape((-1,1)).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        losses = np.append(losses, running_loss)
    
    return losses

#%% Setting training parameters 
max_games = 1000

device = 'cpu'
criterion = nn.MSELoss()
optimizer = optim.Adam(SimpleNN.parameters(), lr=0.001)
max_epochs = 20

loader_params = {'batch_size': 16,
          'shuffle': True,
          #'num_workers': 2,
          'pin_memory': True}

#%%
data = selfplay(Comp, 1000)
training_generator = DataLoader(data, **loader_params)

#%%
for epoch in range(max_epochs) :
    
    running_loss = 0.0
    for batch, labels in training_generator :

        batch, labels = batch.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = SimpleNN(batch)
        loss = criterion(outputs, labels.reshape((-1,1)).float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(running_loss)
    
# %%
for i in range(1000) :
    engine=lambda x : float(SimpleNN(T(x)))
    print(RandTest(engine, 100))
    Comp = Agent(engine)
    data = selfplay(Comp, 500)
    losses = train(data, 10)


# %% Testing against random agent
def RandTest(engine, N) :
    results = np.array([])
    comp = Agent(engine, prob = False)
    game = Connect4Game(comp, Agent())
    for _ in range(N) : 
        game.reset()
        game.play_game()
        results = np.append(results, game.winner)

    return (results==1).sum()/N
# %%
