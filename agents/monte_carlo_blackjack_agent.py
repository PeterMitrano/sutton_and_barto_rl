import sys

import matplotlib.pyplot as plt

from blackjack_model import *


def main():
    np.random.seed(1)

    # state dimensions are: 9 x 18 x 2
    # dealer showing card (1 - 10)
    # player sum (4 - 21)
    # usable ace (0 or 1)
    # action space is HIT or STICK: 2
    returns = {}
    Q = np.zeros((10, 18, 2, 2))

    # initialize policy to stick only on 20 or 21
    pi = np.ones((10, 18, 2)).astype(np.int8)
    test = np.zeros((10, 18, 2))
    pi[:, 16:17, :] = 0

    def Pi(state):
        return pi[state.dealer_showing - 1, state.player_sum - 4, state.usable_ace]

    for j in range(2000):
        states, actions, reward = play_hand(Pi)
        # skip the last state since we took no action there?
        experiences = []
        for s, a in zip(states, actions):
            experiences.append((s, a, reward))

        for s, a, r in experiences:
            state_action_idx = (s.dealer_showing - 1, s.player_sum - 4, s.usable_ace, a)
            test[state_action_idx[:3]] = 1

            if state_action_idx not in returns:
                returns[state_action_idx] = []

            returns[state_action_idx].append(r)

            Q[state_action_idx] = np.mean(returns[state_action_idx])

        # compute the greedy policy with respect to Q
        for dealer_showing in range(pi.shape[0]):
            for player_sum in range(pi.shape[1]):
                usable_ace_state_idx = (dealer_showing, player_sum, 0)
                # pi[usable_ace_state_idx] = np.argmax(Q[usable_ace_state_idx])

                no_usable_ace_state_idx = (dealer_showing, player_sum, 1)
                # pi[no_usable_ace_state_idx] = np.argmax(Q[no_usable_ace_state_idx])

    # compute the greedy policy with respect to Q
    usable_ace_hit_x = []
    usable_ace_hit_y = []
    usable_ace_stick_x = []
    usable_ace_stick_y = []
    no_usable_ace_hit_x = []
    no_usable_ace_hit_y = []
    no_usable_ace_stick_x = []
    no_usable_ace_stick_y = []
    for dealer_showing in range(pi.shape[0]):
        s = []
        for player_sum in range(pi.shape[1]):
            usable_ace_state_idx = (dealer_showing, player_sum, 0)
            pi[usable_ace_state_idx] = np.argmax(Q[usable_ace_state_idx])

            if pi[usable_ace_state_idx] == HIT:
                usable_ace_hit_x.append(dealer_showing)
                usable_ace_hit_y.append(player_sum)
            elif pi[usable_ace_state_idx] == STICK:
                usable_ace_stick_x.append(dealer_showing)
                usable_ace_stick_y.append(player_sum)

            no_usable_ace_state_idx = (dealer_showing, player_sum, 1)
            s.append("({:+0.2f} vs {:+0.2f})".format(*Q[no_usable_ace_state_idx]))
            pi[no_usable_ace_state_idx] = np.argmax(Q[no_usable_ace_state_idx])

            if pi[no_usable_ace_state_idx] == HIT:
                no_usable_ace_hit_x.append(dealer_showing)
                no_usable_ace_hit_y.append(player_sum)
            elif pi[no_usable_ace_state_idx] == STICK:
                no_usable_ace_stick_x.append(dealer_showing)
                no_usable_ace_stick_y.append(player_sum)

        print(", ".join(s))

    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].scatter(usable_ace_hit_x, usable_ace_hit_y, color='r')
    axarr[0, 0].scatter(usable_ace_stick_x, usable_ace_stick_y, color='b')
    axarr[0, 0].set_title('Usable Ace')
    axarr[0, 0].set_xticks(range(10))
    axarr[0, 0].set_xticklabels(['A'] + [str(i) for i in range(2, 11)])
    axarr[0, 0].set_yticks(range(18))
    axarr[0, 0].set_yticklabels(range(4, 22))
    axarr[1, 0].scatter(no_usable_ace_hit_x, no_usable_ace_hit_y, color='r')
    axarr[1, 0].scatter(no_usable_ace_stick_x, no_usable_ace_stick_y, color='b')
    axarr[1, 0].set_title('No Usable Ace')
    axarr[1, 0].set_xticks(range(10))
    axarr[1, 0].set_xticklabels(['A'] + [str(i) for i in range(2, 11)])
    axarr[1, 0].set_yticks(range(18))
    axarr[1, 0].set_yticklabels(range(4, 22))
    plt.show()


if __name__ == "__main__":
    sys.exit(main())
