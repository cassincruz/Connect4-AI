#%%
from connect4 import *
# %%
class TD :
    def __init__(self, V_i={}, lp=0.9) :
        self.V = V_i
        self.lp = lp
    
    def Value(self, board) :
        ID = Board2ID(board)
        if ID in self.V.keys() : return self.V[ID]
        else: 
            self.V[ID] = 0
            return self.V[ID]

#%%
default = 0
lp = 0.9
vals = {}
def Value(board) :
    ID = Board2ID(board)
    if ID in vals.keys() : return vals[ID]
    else : return default

def Update(board, nextboard) :
    ID = Board2ID(board)
    vals[ID] = Value(board) + lp*(Value(nextboard)-Value(board))
# %%
