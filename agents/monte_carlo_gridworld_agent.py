import argparse
import sys
from time import sleep

import gym
import numpy as np


def sample_return(sequence, gamma):
    ret = 0
    s = sequence[0][0]
    for t, step in enumerate(sequence):
        reward = step[1]
        ret += np.power(gamma, t) * reward

    return s, ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--render', action='store_true', help='if set, show the environment gui')
    args = parser.parse_args()
    env = gym.make("GridWorld-v0")

    np.set_printoptions(precision=1, suppress=True)

    obs = env.reset()
    if args.render:
        env.render()
        sleep(0.1)

    rewards = []
    j = 0
    N = 100000
    for j in range(N):
        move = np.random.randint(0, 4)
        s = (obs[0], obs[1])
        obs, reward, done, info = env.step(move)
        rewards.append((s, reward))

    V = np.zeros((5, 5))
    gamma = 0.9

    sample_returns = {}
    max_seq_len = 50
    xs = []
    ys = []
    for i, step in enumerate(rewards):
        seq = rewards[i:i + max_seq_len]
        s, sample_ret = sample_return(seq, gamma)
        if s == (0, 1):
            xs.append(len(seq))
            ys.append(sample_ret)

        if s not in sample_returns:
            sample_returns[s] = []

        sample_returns[s].append(sample_ret)

    for s, sample_rets in sample_returns.items():
        ret = np.mean(sample_rets)
        V[s] = ret

    print("Final V")
    print(V)


if __name__ == "__main__":
    sys.exit(main())
