import sys

import numpy as np

from blackjack_model import play_hand


def main():
    np.random.seed(1)

    # initialize random policy
    pi = np.random.rand(10, 21, 2) > 0.5

    def Pi(state):
        player_sum = state.player_showing + state.player_hidden
        return pi[state.dealer_showing - 1, player_sum - 1, state.usable_ace]

    rewards = []
    for j in range(10):
        # play an episode of blackjack
        s, reward = play_hand(Pi)
        print(reward)
        rewards.append((s, reward))

    # state dimensions are 10 x 21 x 2
    V = np.zeros((10, 21, 2))


if __name__ == "__main__":
    sys.exit(main())
