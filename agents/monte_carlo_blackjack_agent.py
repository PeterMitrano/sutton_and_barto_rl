import sys

from blackjack_model import *


def main():
    np.random.seed(1)

    # initialize random policy
    pi = np.round(np.random.rand(10, 21, 2))

    def Pi(state):
        player_sum = state.player_showing + state.player_hidden
        return pi[state.dealer_showing - 1, player_sum - 1, state.usable_ace]

    rewards = []
    for j in range(10000):
        s, reward = play_hand(Pi)
        rewards.append((s, reward))

    # state dimensions are 10 x 21 x 2
    V = np.zeros((10, 21, 2))


if __name__ == "__main__":
    sys.exit(main())
