import gym
import numpy as np

import matplotlib.pyplot as plt

from utils.misc import sample_continuous_policy

import multiprocessing
import cv2

from models.vae import VAE

from nvae.utils import add_sn
from nvae.vae_celeba import NVAE

import torch

gym.envs.box2d.car_racing.STATE_W, gym.envs.box2d.car_racing.STATE_H = 128, 128

def generate_data():
    NUM_EPISODES = 10
    NUM_STEPS = 1000

    SKIP_STEPS = 3

    nvae = NVAE(z_dim=256, img_dim=(128, 128))
    nvae.apply(add_sn)

    device = "cpu"

    nvae.load_state_dict(torch.load("logs" + "/nvae/" + "ae_ckpt_28_0.902669.pth", map_location=device), strict=False)
    nvae.eval()

    count = 0

    for episode in range(NUM_EPISODES):
        env = gym.make("CarRacing-v2", render_mode="rgb_array")

        obs, r, done = [], [], []
        actions = []

        env.reset()

        for step in range(25):
            action = np.array([0,0,0])
            
            observation, reward, terminated, truncated, info = env.step(action)

        a_rollouts = sample_continuous_policy(env.action_space, NUM_STEPS, 1. / 50)

        for step in range(NUM_STEPS):
            action = a_rollouts[step]
            
            observation, reward, terminated, truncated, info = env.step(action)

            obs = observation[0:112,:,:]

            # resize to 64x64 
            obs = cv2.resize(obs, (128, 128), interpolation=cv2.INTER_LINEAR)
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)

            # save data
            cv2.imwrite("fid/real/" + str(count) + ".png", obs)

            z = torch.randn(1, 256, 4, 4)

            obs, _ = nvae.decoder(z)
            # permute to (H, W, C)
            obs = obs.permute(0, 2, 3, 1)
            # convert to numpy
            obs = obs.detach().numpy()[0]

            # change valies to 0-255
            obs = (obs * 255).astype(np.uint8)
            print(f"Counter: {count}    ", end="\r")
            # save data
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            cv2.imwrite("fid/fake/" + str(count) + ".png", obs)

            # if terminated:
            #     done.append(1)
            #     break
            
            done.append(0)

            count += 1

        env.close()    

generate_data()