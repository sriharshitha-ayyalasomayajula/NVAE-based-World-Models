import gymnasium as gym
import panda_gym
import numpy as np

import matplotlib.pyplot as plt

from utils.misc import sample_continuous_policy

import multiprocessing

def generate_data():
    NUM_EPISODES = 40
    NUM_STEPS = 3000

    SKIP_STEPS = 1

    for episode in range(NUM_EPISODES):
        env = gym.make('PandaPickAndPlace-v3')

        obs, r, done = np.zeros((NUM_STEPS, 25), dtype=np.float32), np.zeros((NUM_STEPS, 1), dtype=np.float32), np.zeros((NUM_STEPS, 1))
        actions = np.zeros((NUM_STEPS, 4), dtype=np.float32)

        env.reset()

        for step in range(NUM_STEPS):
            action = env.action_space.sample()
            
            obse, reward, terminated, info, _ = env.step(action)

            arr1 = obse["observation"]
            arr2 = obse["desired_goal"]
            arr3 = obse["achieved_goal"]

            # concatenate arrays
            obse = np.concatenate((arr1, arr2, arr3))

            obs[step] = obse
            actions[step] = action
            r[step] = reward

            if terminated:
                done[step] = 1
                break
            
            done[step] = 0

            print("Episode: ", episode, " Step: ", step)

        env.close()

        # create random filename
        filename = "dataset/panda_data/" + str(np.random.randint(100000000000)) + ".npz"
        print("yes")
        # obs = np.array(obs)
        # print("yes")
        # actions = np.array(actions)
        # print("yes")
        # r = np.array(r)
        # print("yes")
        # done = np.array(done)
        # print("yes")

        # save data
        np.savez(filename, obs=obs, action=actions, r=r, done=done)

        print("Episode: ", episode, " saved to: ", filename)

generate_data()