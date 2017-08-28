import sys

import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D, axes3d

from blackjack_model import *


def main():
    np.random.seed(1)

    # state dimensions are: 10 x 11 x 2
    # dealer showing card (1 - 10)
    # player sum (11 - 21)
    # usable ace (0 or 1)
    # action space is HIT or STICK: 2
    returns = {}
    Q = np.zeros((10, 11, 2, 2))

    # initialize policy to stick only on 20 or 21
    pi = np.ones((10, 11, 2)).astype(np.int8)
    pi[:, 9:11, :] = 0

    def Pi(state):
        return pi[state.dealer_showing - 1, state.player_sum - 11, state.usable_ace]

    for j in range(500000):
        states, actions, reward = play_hand(Pi)
        # skip the last state since we took no action there?
        experiences = []
        for s, a in zip(states, actions):
            experiences.append((s, a, reward))

        for s, a, r in experiences:
            state_idx = (s.dealer_showing - 1, s.player_sum - 11, s.usable_ace)

            state_action_idx = (s.dealer_showing - 1, s.player_sum - 11, s.usable_ace, a)

            if state_action_idx not in returns:
                returns[state_action_idx] = []

            returns[state_action_idx].append(r)

            Q[state_action_idx] = np.mean(returns[state_action_idx])

            # if s.dealer_showing == 2 and s.player_sum == 4 and not s.usable_ace:
            #     print(a, r, Q[state_idx])

            # compute the greedy policy with respect to Q
            pi[state_idx] = np.argmax(Q[state_idx])

    # compute the greedy policy with respect to Q
    usable_ace_hit_x = []
    usable_ace_hit_y = []
    usable_ace_stick_x = []
    usable_ace_stick_y = []
    no_usable_ace_hit_x = []
    no_usable_ace_hit_y = []
    no_usable_ace_stick_x = []
    no_usable_ace_stick_y = []
    v_x = np.dstack([range(10)] * 11)[0]
    v_y = np.dstack([range(11)] * 10)[0].T
    usable_ace_v_z = np.zeros((10, 11))
    no_usable_ace_v_z = np.zeros((10, 11))
    for dealer_showing in range(pi.shape[0]):
        # s = []
        for player_sum in range(pi.shape[1]):
            usable_ace_state_idx = (dealer_showing, player_sum, 1)
            z_state_idx = (dealer_showing, player_sum)
            usable_ace_v_z[z_state_idx] = Q[dealer_showing, player_sum, 1, pi[usable_ace_state_idx]]

            if pi[usable_ace_state_idx] == HIT:
                usable_ace_hit_x.append(dealer_showing)
                usable_ace_hit_y.append(player_sum)
            elif pi[usable_ace_state_idx] == STICK:
                usable_ace_stick_x.append(dealer_showing)
                usable_ace_stick_y.append(player_sum)

            no_usable_ace_state_idx = (dealer_showing, player_sum, 0)
            no_usable_ace_v_z[z_state_idx] = Q[dealer_showing, player_sum, 0, pi[no_usable_ace_state_idx]]
            # s.append("({:+0.2f} vs {:+0.2f})".format(*Q[no_usable_ace_state_idx]))

            # if dealer_showing + 1 == 2 and player_sum + 4 == 4:
            #     print(Q[dealer_showing, player_sum, 0], pi[no_usable_ace_state_idx])

            if pi[no_usable_ace_state_idx] == HIT:
                no_usable_ace_hit_x.append(dealer_showing)
                no_usable_ace_hit_y.append(player_sum)
            elif pi[no_usable_ace_state_idx] == STICK:
                no_usable_ace_stick_x.append(dealer_showing)
                no_usable_ace_stick_y.append(player_sum)

                # print(", ".join(s))

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(usable_ace_hit_x, usable_ace_hit_y, color='r')
    ax.scatter(usable_ace_stick_x, usable_ace_stick_y, color='b')
    ax.set_title('Usable Ace')
    ax.set_xticks(range(10))
    ax.set_xticklabels(['A'] + [str(i) for i in range(2, 11)])
    ax.set_yticks(range(11))
    ax.set_yticklabels(range(11, 22))

    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(no_usable_ace_hit_x, no_usable_ace_hit_y, color='r')
    ax.scatter(no_usable_ace_stick_x, no_usable_ace_stick_y, color='b')
    ax.set_title('No Usable Ace')
    ax.set_xticks(range(10))
    ax.set_xticklabels(['A'] + [str(i) for i in range(2, 11)])
    ax.set_yticks(range(11))
    ax.set_yticklabels(range(11, 22))

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.plot_wireframe(v_x, v_y, usable_ace_v_z)
    ax.set_xticks(range(10))
    ax.set_xticklabels(['A'] + [str(i) for i in range(2, 11)])
    ax.set_yticks(range(11))
    ax.set_yticklabels(range(11, 22))
    ax.set_title("V* usable ace")

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.plot_wireframe(v_x, v_y, no_usable_ace_v_z)
    ax.set_xticks(range(10))
    ax.set_xticklabels(['A'] + [str(i) for i in range(2, 11)])
    ax.set_yticks(range(11))
    ax.set_yticklabels(range(11, 22))
    ax.set_title("V* no usable ace")
    plt.show()


if __name__ == "__main__":
    sys.exit(main())
