from time import sleep

import gym
import numpy as np
from gym.envs.classic_control import GridWorld

if __name__ == "__main__":
    env = gym.make("GridWorld-v0")

    obs = env.reset()
    print(obs)
    env.render()
    j = 0
    sleep(0.1)

    while True:
        move = np.random.randint(0, 4)
        obs, reward, done, info = env.step(move)
        env.render()
        print(j, obs, GridWorld.action_to_string(move), obs, reward)
        sleep(0.1)

        j += 1
