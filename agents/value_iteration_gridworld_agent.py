"""
This code shows a greedy actor in gridworld.
At each iteration, it solves for the state-value function of the current greedy policy
"""
import argparse
from time import sleep

import gym
import numpy as np

from agents.grid_world_model import value_iteration, print_policy, A, Env, R

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--render', action='store_true', help='if set, show the environment gui')
    args = parser.parse_args()
    env = gym.make("GridWorld-v0")

    done = False
    V, iters = value_iteration()
    print("Value iteration completed in {} iterations".format(iters))
    print(V)
    np.set_printoptions(precision=1, suppress=True)
    pi = np.zeros((5, 5, 4))

    for r in range(pi.shape[0]):
        for c in range(pi.shape[1]):
            s = (r, c)
            maxQ = -1e12
            maxA = None
            for a in A(s):
                s1 = Env(s, a)
                reward = R(s, a, s1)
                q = reward + 0.9 * V[s1]
                if q > maxQ:
                    maxA = a
                    maxQ = q
            pi[r, c, env.string_to_action(maxA)] = 1

    print_policy(pi)

    obs = env.reset()
    if args.render:
        env.render()
        sleep(0.1)

    for j in range(10):
        a = np.argmax(pi[(obs[0], obs[1])])
        obs, reward, _, info = env.step(a)
        if args.render:
            env.render()
            sleep(0.1)
