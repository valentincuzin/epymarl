from pathlib import Path

import gymnasium as gym
from gymnasium.spaces import Tuple
import torch

from .ddpg import DDPG
from .mappo import MAPPO


class FrozenTag(gym.Wrapper):
    """Tag with pretrained prey agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt_action_space = self.action_space[-1]
        self.pt_observation_space = self.observation_space[-1]
        self.action_space = Tuple(self.action_space[:-1])
        self.observation_space = Tuple(self.observation_space[:-1])
        self.n_agents = 3
        self.unwrapped.n_agents = 3

    def reset(self, seed=None, options=None):
        obss, info = super().reset(seed=seed, options=options)
        return obss[:-1], info

    def step(self, action):
        random_action = 0
        action = tuple(action) + (random_action,)
        obs, rew, done, truncated, info = super().step(action)
        obs = obs[:-1]
        rew = rew[:-1]
        return obs, rew, done, truncated, info


class RandomTag(gym.Wrapper):
    """Tag with pretrained prey agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{})
        self.num_good = kwargs["num_good"]
        self.num_adversaries = kwargs["num_adversaries"]
        self.pt_action_space = self.action_space[-self.num_good:]
        self.pt_observation_space = self.observation_space[-self.num_good:]
        self.action_space = Tuple(self.action_space[:-self.num_good])
        self.observation_space = Tuple(self.observation_space[:-self.num_good])
        self.n_agents = self.num_adversaries
        self.unwrapped.n_agents = self.num_adversaries

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs[:-self.num_good], info

    def step(self, action):
        if isinstance(self.pt_action_space, tuple):
            random_action = tuple([pt_action_space.sample().item() for pt_action_space in self.pt_action_space])
        else:
            random_action = self.pt_action_space.sample()
        action = tuple(action) + random_action
        obs, rew, done, truncated, info = super().step(action)
        obs = obs[:-self.num_good]
        rew = rew[:-self.num_good]
        return obs, rew, done, truncated, info


class PretrainedTag(gym.Wrapper):
    """Tag with pretrained prey agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{})
        self.num_good = kwargs["num_good"]
        self.num_adversaries = kwargs["num_adversaries"]
        self.pt_action_space = self.action_space[-self.num_good:]
        self.pt_observation_space = self.observation_space[-self.num_good:]
        self.action_space = Tuple(self.action_space[:-self.num_good])
        self.observation_space = Tuple(self.observation_space[:-self.num_good])
        self.n_agents = self.num_adversaries
        self.unwrapped.n_agents = self.num_adversaries

        self.preys = MAPPO(16, 5, 64)
        # current file dir
        param_path = Path(__file__).parent / "agent.th"
        save_dict = torch.load(param_path, map_location="cpu")
        self.preys.load_params(save_dict)
        self.preys.policy.eval()
        self.last_prey_obs = None

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.last_prey_obs = obs[-self.num_good:]
        return obs[:-self.num_good], info

    def step(self, action):
        prey_action = self.preys.step(self.last_prey_obs)
        action = tuple(action) + prey_action
        obs, rew, done, truncated, info = super().step(action)
        self.last_prey_obs = obs[-self.num_good:]
        obs = obs[:-self.num_good]
        rew = rew[:-self.num_good]
        return obs, rew, done, truncated, info
