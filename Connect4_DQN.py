"""
DeepQ Network for Connect 4
- Tutorial: https://www.youtube.com/watch?v=wc-FxNENg9U
- Also borrowing code from here: https://www.albertobas.com/blog/training-dqn-agent-to-play-connect-four
- Also useful for conv network: https://towardsdatascience.com/deep-reinforcement-learning-and-monte-carlo-tree-search-with-connect-4-ba22a4713e7a
- Also helpful, esp for NN architectures: https://codebox.net/pages/connect4
- For general theory on architecture design, see Calin - Deep Learning Architectures, A Mathematical Approach
"""

#%%
from nn import *
from Connect4 import Connect4Env

import torch

import gymnasium as gym 

import numpy as np 
import copy
from tqdm import trange

import matplotlib.pyplot as plt 
    
#%%
class Agent() : 
    def __init__(self, env, network, gamma=0.9, epsilon=0.8, batch_size=1024, 
                 mem_size=100000, eps_min=0.1, eps_dec=5e-6):
        
        self.env = env
        self.gamma = gamma # Reward decay parameter
        self.epsilon = epsilon # Explore/exploit parameter
        self.eps_min = eps_min
        self.eps_dec = eps_dec 
        self.action_space = [i for i in range(7)]
        self.mem_size = mem_size
        self.batch_size = batch_size 
        self.mem_cntr = 0 # memory counter 

        self.Q_eval = network
        
        self.state_memory = np.zeros((self.mem_size, 42), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, 42), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    @property
    def max_mem(self) : 
        return min(self.mem_cntr, self.mem_size)

    def store_transition(self, observation, info, action, reward, observation_, done) : 
        index = self.mem_cntr % self.mem_size # Need to find the position of the first open memory
        self.state_memory[index] = observation['board'].flatten() * info['active color']
        self.new_state_memory[index] = observation_['board'].flatten() * info['active color']
        self.reward_memory[index] = reward * info['active color']
        self.action_memory[index] = action 
        self.terminal_memory[index] = done 

        self.mem_cntr += 1

        return index

    def _evaluate(self, observation=None, info=None, state=None) :
        if state is None :  
            state = observation['board'] * info['active color']

        state = torch.reshape(torch.Tensor(state).flatten(), (1, -1)).type(torch.float).to(self.Q_eval.device)
        action_values = self.Q_eval.forward(state)

        return action_values
    
    def _getbatch(self, batch, randomize=True) : 
        batch_index = np.arange(len(batch), dtype=np.int32)
        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        if randomize : 
            # Randomly flip boards along their symmetric axis
            randomize_idx = batch_index[np.random.random(len(batch)) > 0.5]
            state_batch[randomize_idx] = state_batch[randomize_idx].reshape(-1, 6, 7).flip(2).reshape(-1, 6*7)
            action_batch[randomize_idx] = 6 - action_batch[randomize_idx]
            new_state_batch[randomize_idx] = new_state_batch[randomize_idx].reshape(-1, 6, 7).flip(2).reshape(-1, 6*7)

        return batch_index, state_batch, action_batch, new_state_batch, reward_batch, terminal_batch

    def choose_action(self, observation, info, epsilon=None) : 
        epsilon = self.epsilon if epsilon is None else epsilon
        self.action_space = info['available actions']

        if np.random.random() > self.epsilon : 
            action_values = self._evaluate(observation, info)

            # Masking to only choose from available actions
            mask = torch.Tensor([0 if action in self.action_space else -np.inf for action in range(7)]).to(self.Q_eval.device)
            action = torch.argmax(action_values + mask).item()
        
        else : 
            action = np.random.choice(self.action_space)

        return action 
    
    def play_game(self, store_transitions=True, epsilon=None, initial_actions=[], rand_move=False) : 
        # Initialize environment
        observation, info = self.env.reset() 
        done = False

        # Apply initial actions if any
        for action in initial_actions : 
            observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

        # Initialize rolling memory
        actions = []
        observations = [copy.deepcopy(observation)]
        infos = [copy.deepcopy(info)]

        # Player 1 goes first to commence learning loop, or random action if rand_move
        action = self.choose_action(observation, info, epsilon) if not rand_move else np.random.choice(info['available actions'])
        actions.append(action)

        observation_, reward, terminated, truncated, info_ = self.env.step(action)
        observations.append(copy.deepcopy(observation_))
        infos.append(copy.deepcopy(info_))
        observation, info = observation_, info_

        # Record batch indices
        indices = []

        # Begin learning loop
        while not done : 
            action  = self.choose_action(observation, info, epsilon)
            actions.append(action)

            observation_, reward, terminated, truncated, info_ = self.env.step(action)
            observations.append(copy.deepcopy(observation_))
            infos.append(copy.deepcopy(info_))

            done = terminated or truncated

            if store_transitions : 
                indices.append(self.store_transition(observations.pop(0), infos.pop(0), actions.pop(0), 
                                      reward, observation_, done))
            observation, info = observation_, info_

        # Store transition for other player (win and loss both get recorded)
        if store_transitions : 
            indices.append(self.store_transition(observations.pop(0), infos.pop(0), actions.pop(0), 
                                  reward, observation_, done))
            
        # Return game length
        return indices
    
    
    def learn(self, batch=[], n_steps=1, dec_eps=True, return_deltas=False) : 
        """
        n_steps: Number of gradient descent steps to occur during training.
        batch: Either list of indices used to get transitions from memory, OR a positive integer in which case last N transitions will be used in learning.
        dec_eps: Whether or not to decrease epsilon after learning.
        return_loss: Whether or not to return final loss 
        """
        
        if self.mem_cntr < self.batch_size : 
            # We wont start learning until the agent has filled up its memory
            return 
        
        if type(batch) == int : 
            batch = np.arange(self.max_mem - batch, self.max_mem)
        
        elif len(batch) == 0 : 
            batch = np.random.choice(self.max_mem, self.batch_size, replace=False) # False, since we don't want to select the same memory more than once

        batch_index, state_batch, action_batch, new_state_batch, reward_batch, terminal_batch = self._getbatch(batch)

        if return_deltas : 
            q_eval_0 = self.Q_eval.forward(state_batch)[batch_index, action_batch]

        for _ in range(n_steps) : 
            self.Q_eval.optimizer.zero_grad()

            q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
            q_next = self.Q_eval.forward(new_state_batch)

            q_next[terminal_batch] = 0.0 

            q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0] 

            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()

        if dec_eps : 
            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min 

        if return_deltas : 
            q_eval_f = self.Q_eval.forward(state_batch)[batch_index, action_batch]

            return q_eval_f - q_eval_0
        
    def train(self, n_games, learn_steps=10, n_replays=0, show_pbar=True) : 
        
        for _ in (pbar := trange(n_games) if show_pbar else range(n_games)) : 
            if show_pbar : 
                pbar.set_description(f"Training (epsilon={self.epsilon:.3f})")

            # Play a game and learn from it
            self.learn(batch := self.play_game(), n_steps=learn_steps, dec_eps=False)

            # Replaying games if n_replays > 0 
            for _ in range(n_replays) : 
                try : 
                    replay_move = np.random.randint(0, len(batch))
                    initial_actions = self.action_memory[batch[:replay_move]]
                    batch_ = self.play_game(initial_actions=initial_actions, rand_move=True)
                    self.learn(batch[:replay_move] +  batch_, n_steps = min(learn_steps // n_replays, 1), dec_eps=False)
                
                except : 
                    return batch

            # Learn from all recorded moves
            self.learn(n_steps=learn_steps)

# %% Validation helper functions
def play_agent(agent, env, opp="random") : 
    epsilon = agent.epsilon
    agent.epsilon = 0

    actions = []

    done = False
    observation, info = env.reset()

    while not done : 
        if opp == "human" : 
            plt.imshow(observation['board'])
            plt.show()
        
        action = np.random.choice(info['available actions']) if opp != "human" else int(input("What move?"))
        observation, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        done = terminated or truncated 

        if not done : 
            action = agent.choose_action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            done = terminated or truncated 

    agent.epsilon = epsilon 

    return (reward, actions)

def rand_validate(agent, env, n_games) : 
    winners = []
    for _ in trange(n_games, desc='Validating') : 
        reward, _ = play_agent(agent, env)

        winners.append(reward)

    winrate = np.array([winner == -1 for winner in winners]).mean()
    return winrate

#%% Training loop
if __name__ == '__main__' : 
    env = gym.make('Connect4-v0')
    network = CNN_B()
    #network.load_state_dict(torch.load('./weights/98wr_cnn_b.pt', map_location=network.device))

    agent_params = dict(
        gamma=0.9, 
        epsilon=0.8, 
        batch_size=1024, 
        mem_size=100000, 
        eps_min=0.1, 
        eps_dec=0
    )

    agent = Agent(env, network, **agent_params)
    
    #writer = SummaryWriter()
    
    print(f"Start: {rand_validate(agent, env, 1000) * 100:.1f}% WR")
    for i in range(100) : 
        agent.train(500)
        print(f"Epoch {i+1}: {rand_validate(agent, env, 1000) * 100:.1f}% WR")

# %%
