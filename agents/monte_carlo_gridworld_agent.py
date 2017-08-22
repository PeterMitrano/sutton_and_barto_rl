import argparse
from time import sleep

import gym
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--render', action='store_true', help='if set, show the environment gui')
    args = parser.parse_args()
    env = gym.make("GridWorld-v0")

    V = np.zeros((5, 5))
    first_visit_counts = np.zeros((5, 5))

    obs = env.reset()
    if args.render:
        env.render()
        sleep(0.1)

    for i in range(1000):
        done = False
        first_visit_returns = {}
        for j in range(100):
            move = np.random.randint(0, 4)
            s = (obs[0], obs[1])
            obs, reward, done, info = env.step(move)
            if s not in first_visit_returns:
                first_visit_returns[s] = reward
                first_visit_counts[s] += 1
            if args.render:
                env.render()
                sleep(0.1)

        # update the state value estimate to be the average for first-visit results
        for s, reward in first_visit_returns.items():
            alpha = 1 / first_visit_counts[s]
            V[s] = (1 - alpha) * V[s] + alpha * reward

    print(V)
