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
    else:
        return card


def play_hand(pi):
    dealer_showing = sample_card_value()
    dealer_hidden = sample_card_value()
    player_showing = sample_card_value()
    player_hidden = sample_card_value()
    print("{},{} {},{}".format(dealer_hidden, dealer_showing, player_hidden, player_showing))
    return play_hand_(pi, dealer_showing, dealer_hidden, player_showing, player_hidden)


def play_hand_(pi, dealer_showing, dealer_hidden, player_showing, player_hidden):
    player = player_hidden + player_showing
    usable_ace = 1 if (player_hidden == 1 and player < 21) else 0
    s = BlackJackState(dealer_showing, player_showing, player_hidden, usable_ace)

    # implement user policy
    while True:
        action = pi(s)
        if action == HIT:
            v = sample_card_value()
            print("HIT", v)
            player += v
            if player > 21:
                print("player bust")
                return s, LOSE
        elif action == STICK:
            # now it's the dealer's turn
            break
        else:
            raise ValueError("must HIT or STICK")

    # implement dealer policy
    dealer = dealer_hidden + s.dealer_showing

    while True:
        if dealer > 21:
            print("dealer BUST")
            return s, WIN
        elif dealer < 17:
            v = sample_card_value()
            print("DEALER HIT", v)
            dealer += v
        elif dealer == player:
            print("draw")
            return s, DRAW
        elif dealer > player:
            print("dealer won")
            return s, LOSE
        elif player > dealer:
            print("player won")
            return s, WIN
