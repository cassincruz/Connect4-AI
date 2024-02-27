#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#%% Deep fully connected network
class DeepQNetwork(nn.Module) : 
    def __init__(self, lr: float, input_dims: int, fc1_dims: int, fc2_dims: int, n_actions: int) : 
        """
        Neural network used by an agent 
        """
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) # Star allows us to extend to 2D inputs
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state) : 
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        # Want to return raw estimates, not activated
        return actions

#%% Deep convolutional network
class SimpleCNN(nn.Module) : 
    def __init__(self, lr=1e-5) : 
        """
        Neural network used by an agent 
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 7, 4)
        self.conv2 = nn.Conv2d(7, 7, 2)

        self.fc1 = nn.Linear(7 * 2 * 3, 256) 
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 7)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state) : 
        x = torch.reshape(state, (-1, 1, 6, 7))
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        
        x = torch.reshape(x, (state.shape[0], -1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        actions = self.fc3(x)

        # Want to return raw estimates, not activated
        return actions

#%%
class CNN_B(nn.Module) : 
    def __init__(self, lr=1e-5) : 

        super(CNN_B, self).__init__()
        self.conv = nn.Conv2d(1, 128, 4)
        self.fc1 = nn.Linear(128 * 3 * 4, 64)
        self.fc2 = nn.Linear(64, 7)

        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state) : 
        x = torch.reshape(state, (-1, 1, 6, 7))
        x = F.leaky_relu(self.conv(x))
        x = torch.reshape(x, (state.shape[0], -1))
        x = F.relu(self.fc1(x))
        
        actions = self.fc2(x)

        return actions

#%%
class CNN_C(nn.Module) : 
    def __init__(self, lr=1e-5) : 

        super(CNN_C, self).__init__()
        self.conv = nn.Conv2d(1, 128, 4)
        self.dropout1 = nn.Dropout2d(0.2)
        self.fc1 = nn.Linear(128 * 3 * 4, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 128)
        self.dropout3 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 7)

        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=0.2)
        self.loss = nn.MSELoss()
        self.device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state) : 
        x = torch.reshape(state, (-1, 1, 6, 7))
        x = F.leaky_relu(self.conv(x))
        x = self.dropout1(x)
        x = torch.reshape(x, (state.shape[0], -1))
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)

        actions = self.fc3(x)

        return actions
# %%
cnn = CNN_C()
# %%
