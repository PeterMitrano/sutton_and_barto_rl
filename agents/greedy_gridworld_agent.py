"""
This code shows a greedy actor in gridworld.
First, it solves for the state-value function of a uniformly random policy.
Then, it acts greedily with respect to those state value functions.
The main problem with this, is that there is a local minimum around the +5 B loop.
More long-term reward would be gained if the actor always favored the +10 A loop.
However, because the state-value function is being computed for a uniform random policy,
we don't get the optimal policy. See `iterative_greedy_gridworld_agent.py` for the next step.

"""
from time import sleep

import gym
import numpy as np
from gym.envs.classic_control import GridWorld

from agents.gridworld_model import evaluate_policy, A, Env

if __name__ == "__main__":
    np.set_printoptions(precision=4)
    env = gym.make("GridWorld-v0")

    obs = env.reset()
    env.render()
    j = 0
    sleep(0.1)

    V, _ = evaluate_policy()
    print(V, _)

    while True:
        best_V = -1e12
        best_a = env.action_space.sample()
        for a in A(obs):
            s1 = Env(obs, a)
            if V[s1] > best_V:
                best_V = V[s1]
                best_a = a

        move = GridWorld.string_to_action(best_a)
        obs, reward, done, info = env.step(move)
        env.render()
        sleep(0.1)
        j += 1
