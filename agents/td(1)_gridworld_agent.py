import argparse
import sys
from time import sleep

import gym
import numpy as np


def a_to_char(a):
    if a == 0:
        return 'N'
    elif a == 1:
        return 'S'
    elif a == 2:
        return 'E'
    elif a == 3:
        return 'W'


def pi(Q, s, epislon):
    if np.random.rand() < epislon:
        return np.random.randint(0, 4)
    else:
        return np.argmax(Q[s])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--render', action='store_true', help='if set, show the environment gui')
    args = parser.parse_args()
    env = gym.make("GridWorld-v0")

    np.set_printoptions(precision=1, suppress=True)

    obs = env.reset()
    N = 10000
    Q = np.zeros((5, 5, 4))
    e = np.zeros((5, 5, 4))
    s = (obs[0], obs[1])
    gamma = 0.9
    alpha = 0.1
    lmda = 0.9
    epsilon = 1
    for j in range(N):
        a = pi(Q, s, epsilon)
        epsilon *= 0.999
        obs, reward, done, info = env.step(a)
        td_error = reward + gamma * np.max(Q[obs[0], obs[1]]) - Q[s[0], s[1], a]
        e[s[0], s[1], a] += 1
        for r in range(5):
            for c in range(5):
                for a_ in range(4):
                    Q[r, c, a_] += alpha * td_error * e[r, c, a_]
                    e[r, c, a_] *= lmda * gamma

        s = (obs[0], obs[1])

    print("Final Policy")
    pfunc = np.vectorize(a_to_char)
    print(pfunc(np.argmax(Q, axis=2)))

    # for j in range(10):
    #     env.reset()
    #     for j in range(10):
    #         a = pi(Q, s, 0)
    #         env.step(a)
    #         env.render()
    #         sleep(0.1)


if __name__ == "__main__":
    sys.exit(main())
