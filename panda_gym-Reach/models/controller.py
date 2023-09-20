""" Define controller """
import torch
import torch.nn as nn

import gym
import panda_gym
import numpy as np
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3 .common.noise import NormalActionNoise
from sb3_contrib.common.wrappers import TimeFeatureWrapper

class Controller(nn.Module):
    """ Controller """
    def __init__(self, latents, recurrents, actions):
        super().__init__()
        self.fc = nn.Linear(latents + recurrents, actions)

        rb_kwargs = {'online_sampling' : True,
             'goal_selection_strategy' : 'future',
             'n_sampled_goal' : 4}

        policy_kwargs = {'net_arch' : [512, 512, 512], 
                        'n_critics' : 2}

        env = gym.make("PandaReach-v2")

        n_actions = env.action_space.shape[0]
        noise = NormalActionNoise(mean = np.zeros(n_actions), sigma = 0.1 * np.ones(n_actions))

        env = gym.make("PandaReach-v2")
        self.env = TimeFeatureWrapper(env)

        self.controller = DDPG.load("logs/ctrl/model", env = env)

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=1)
        return self.fc(cat_in)
