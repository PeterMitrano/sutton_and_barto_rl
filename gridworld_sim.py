import numpy as np
import gym
from gym.envs.classic_control import GridWorld
from time import sleep

if __name__ == "__main__":
    env = gym.make("GridWorld-v0")

    for i in range(1):
        obs = env.reset()
        j = 0
        while True:
            move = np.random.randint(0, 4)
            env.render()
            s0 = obs
            obs, reward, done, info = env.step(move)
            print(s0, GridWorld.actionToString(move), obs, reward)
            sleep(10)

            j += 1
            if j > 10:
                break
