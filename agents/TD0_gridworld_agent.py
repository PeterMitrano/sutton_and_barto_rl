"""
This code shows a greedy actor in gridworld.
First it solves for the state-value function of a uniformly random policy with TD(0)
Then, it acts greedily with respect to those state value functions.
"""
import argparse
from time import sleep

import gym
import numpy as np
from gym.envs.classic_control import GridWorld

from gridworld_model import A, Env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--render', action='store_true', help='if set, show the environment gui')
    args = parser.parse_args()

    np.set_printoptions(precision=1, suppress=True)
    env = gym.make("GridWorld-v0")

    s = env.reset()
    s = (s[0], s[1])

    if args.render:
        env.render()
        sleep(0.1)

    V = np.zeros((5, 5))
    visits = np.zeros((5, 5))
    gamma = 0.90

    for i in range(10000):
        a = env.action_space.sample()
        s_prime, reward, done, info = env.step(a)
        s_prime = (s_prime[0], s_prime[1])

        old_v = V[s]
        visits[s] += 1
        alpha = 1 / visits[s]
        V[s] = (1 - alpha) * V[s] + alpha * (reward + gamma * V[s_prime])
        s = s_prime

    print(V)

    for i in range(10):
        obs = env.reset()

        for i in range(10):
            best_V = -1e12
            best_a = env.action_space.sample()
            for a in A(obs):
                s1 = Env(obs, a)
                if V[s1] > best_V:
                    best_V = V[s1]
                    best_a = a

            move = GridWorld.string_to_action(best_a)
            obs, reward, done, info = env.step(move)

            if args.render:
                env.render()
                sleep(0.1)
