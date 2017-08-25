import sys

import numpy as np

from blackjack_model import BlackJackState, play_hand, sample_card_value


def main():
    rewards = []
    for j in range(100000):
        # play an episode of blackjack
        s, reward = play_hand()
        rewards.append((s, reward))

    V = np.zeros((5, 5))


if __name__ == "__main__":
    sys.exit(main())
