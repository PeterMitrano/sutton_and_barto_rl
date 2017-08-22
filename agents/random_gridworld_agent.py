from time import sleep

import gym
import numpy as np

if __name__ == "__main__":
    env = gym.make("GridWorld-v0")

    obs = env.reset()
    env.render()
    j = 0
    sleep(0.1)

    while True:
        move = np.random.randint(0, 4)
        obs, reward, done, info = env.step(move)
        env.render()
        sleep(0.1)
        j += 1
