from collections import namedtuple

import numpy as np

BlackJackState = namedtuple("BlackjackState", ["dealer_showing", "player_showing", "player_hidden", "usable_ace"])

HIT = 1
STICK = 0
WIN = 1
LOSE = -1
DRAW = 0


def sample_card_value():
    """ gives a random value from 1 to 10, 1 indicating ace. Sampled according to a standard 52 card deck"""
    card = np.random.randint(1, 14)
    if card > 10:  # J, K, Q
        return 10


def play_hand(pi):
    dealer_showing = sample_card_value()
    player_showing = sample_card_value()
    player_hidden = sample_card_value()
    player = player_hidden + player_showing
    usable_ace = (player_hidden == 1 and player < 21)
    s = BlackJackState(dealer_showing, player_showing, player_hidden, usable_ace)

    # implement user policy
    action = pi(s)
    if action == HIT:
        player += sample_card_value()
        if player > 21:
            return LOSE
    elif action == STICK:
        # now it's the dealer's turn
        pass
    else:
        raise ValueError("must HIT or STICK")

    # implement dealer policy
    dealer_hidden = sample_card_value()
    dealer = dealer_hidden + s.dealer_showing
    if dealer > 21:
        return s, WIN
    elif dealer == player:
        return s, DRAW
    elif dealer > player:
        return s, LOSE
    elif player > dealer:
        return s, WIN
