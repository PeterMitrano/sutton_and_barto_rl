from time import sleep

import gym
import numpy as np
from gym.envs.classic_control import GridWorld
from gym.envs.classic_control.gridworld import XY

if __name__ == "__main__":
    env = gym.make("GridWorld-v0")

    obs = env.reset()
    print(obs)
    env.render()
    j = 0
    sleep(0.1)

    while True:
        move = np.random.randint(0, 4)
        s0 = XY(obs.x, obs.y)
        obs, reward, done, info = env.step(move)
        env.render()
        print(j, s0, GridWorld.actionToString(move), obs, reward)
        sleep(0.1)

        j += 1
