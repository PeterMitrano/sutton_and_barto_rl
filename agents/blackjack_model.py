from copy import deepcopy

import numpy as np


class BlackJackState:
    def __init__(self):
        self.player_cards = []
        self.usable_ace = False
        self.dealer_cards = []
        self.dealer_showing = None

    @property
    def player_sum(self):
        sum = 0
        for c in self.player_cards:
            if c == 1 and sum + 11 < 21:
                sum += 11
            else:
                sum += c

        return sum

    @property
    def dealer_sum(self):
        return np.sum(self.dealer_cards) + self.dealer_showing

    def __repr__(self):
        s = "P("
        for p in self.player_cards:
            s += str(p) + ", "
        s += "), D(" + str(self.dealer_showing) + ", "

        for d in self.dealer_cards:
            s += str(d) + ", "
        s += ")"

        if self.usable_ace:
            s = "*" + s

        return s


STICK = 0
HIT = 1

LOSE = -1
DRAW = 0
WIN = 1


def sample_card_value():
    """ gives a random value from 1 to 10, 1 indicating ace. Sampled according to a standard 52 card deck"""
    card = np.random.randint(1, 14)
    if card > 10:  # J, K, Q
        return 10
    else:
        return card


def play_hand(pi):
    dealer_showing = sample_card_value()
    dealer_hidden = sample_card_value()
    player_showing = sample_card_value()
    player_hidden = sample_card_value()
    first_action = np.random.randint(0, 2)
    return play_hand_(pi, dealer_showing, dealer_hidden, player_showing, player_hidden, first_action)


def play_hand_(pi, dealer_showing, dealer_hidden, player_showing, player_hidden, first_action):
    states = []
    actions = []

    s = BlackJackState()
    s.player_cards.append(player_showing)
    s.player_cards.append(player_hidden)
    s.dealer_showing = dealer_showing
    s.dealer_cards.append(dealer_hidden)
    s.usable_ace = 1 if (player_hidden == 1 and s.player_sum < 21) else 0
    states.append(deepcopy(s))

    # print(s)

    # implement user policy
    i = 0
    while True:
        if i == 0:
            action = first_action
        else:
            action = pi(s)

        actions.append(action)
        if action == HIT:
            c = sample_card_value()
            # print("HIT", c)
            s.player_cards.append(c)
            if s.player_sum > 21:
                # print("player bust")
                return states, actions, LOSE
            else:
                states.append(deepcopy(s))
        elif action == STICK:
            # now it's the dealer's turn
            break
        else:
            raise ValueError("must HIT or STICK")

        i += 1

    # implement dealer policy
    while True:
        if s.dealer_sum > 21:
            # print("dealer BUST")
            return states, actions, WIN
        elif s.dealer_sum < 17:
            c = sample_card_value()
            # print("DEALER HIT", c)
            s.dealer_cards.append(c)
        elif s.dealer_sum == s.player_sum:
            # print("draw")
            return states, actions, DRAW
        elif s.dealer_sum > s.player_sum:
            # print("dealer won")
            return states, actions, LOSE
        elif s.player_sum > s.dealer_sum:
            # print("player won")
            return states, actions, WIN
