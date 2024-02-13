#%%
import gymnasium as gym 
from gymnasium import spaces
from gymnasium.envs.registration import register
from setuptools import setup

import pygame 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.signal import convolve2d

import torch
from torch.nn.functional import conv2d

#%%
class Connect4Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(-1, 1, shape=(6, 7), dtype=int),
                "turn": spaces.Discrete(43, 1),
            }
        )

        # We have 7 available actions
        self.action_space = spaces.Discrete(7)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self._board = torch.zeros((6, 7), dtype=torch.int32)
        self._turn = 1

    def _get_obs(self):
        return {"board": self._board, "turn": self._turn}

    def _get_info(self):
        return {"active color": self._active_color, "available actions": self._available_actions}

    @property
    def _available_actions(self):
        return [n for n in range(7) if torch.sum(torch.abs(self._board[:, n])) < 6]

    @property
    def _active_color(self):
        return 2 * (self._turn % 2) - 1

    def reset(self, options=None, seed=None):
        self._board = torch.zeros((6, 7), dtype=torch.int32)
        self._turn = 1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        if action in self._available_actions:
            empty_rows = torch.where(self._board[:, action] == 0)[0]
            if len(empty_rows) > 0:
                self._board[empty_rows[-1], action] = self._active_color
                reward, terminated, truncated = self._check_board()
                self._turn += 1

            observation = self._get_obs()
            info = self._get_info()

            if self.render_mode == "human":
                self._render_frame()

            return observation, reward, terminated, truncated, info

    def _check_board(self):
        patterns = [torch.ones((1, 4), dtype=torch.float32),
                    torch.ones((4, 1), dtype=torch.float32),
                    torch.eye(4, dtype=torch.float32),
                    torch.flip(torch.eye(4, dtype=torch.float32), [1])]

        board = self._board.unsqueeze(0).unsqueeze(0).float()
        for pattern in patterns:
            pattern = pattern.unsqueeze(0).unsqueeze(0)
            if torch.any(conv2d(board, pattern * float(self._active_color), padding='same').max() >= 4):
                return float(self._active_color), True, False

        if self._turn >= 42:
            return 0.0, False, True

        return 0.0, False, False

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # White background

        cell_size = self.window_size // 7
        radius = cell_size // 2 - 5

        for row in range(6):
            for col in range(7):
                pygame.draw.rect(
                    canvas, 
                    (0, 0, 0), 
                    (col * cell_size, row * cell_size, cell_size, cell_size), 
                    3
                )
                color = (255, 255, 255)
                if self._board[row, col] == 1:
                    color = (255, 0, 0)
                elif self._board[row, col] == -1:
                    color = (0, 0, 255)
                pygame.draw.circle(
                    canvas,
                    color,
                    (int(col * cell_size + cell_size // 2), int(row * cell_size + cell_size // 2)),
                    radius
                )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

#%%
register(
    id='Connect4-v0',
    entry_point=Connect4Env,
    max_episode_steps=42,
)

# %%
