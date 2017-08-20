import numpy as np


def S():
    states = []
    for i in range(5):
        for j in range(5):
            states.append((i, j))
    return states


def Pi(state, action):
    return 0.25


def R(state, action, next_state):
    r = state[0]
    c = state[1]
    if c == 1 and r == 0:
        return 10
    elif c == 3 and r == 0:
        return 5
    elif (action == 'N' and r == 0) or \
            (action == 'S' and r == 4) or \
            (action == 'E' and c == 4) or \
            (action == 'W' and c == 0):
        return -1
    else:
        return 0


def A(state):
    return ['N', 'S', 'E', 'W']


def Env(state, action):
    r = state[0]
    c = state[1]
    if c == 1 and r == 0:
        s = (4, 1)
    elif c == 3 and r == 0:
        s = (2, 3)
    elif (action == 'N' and r == 0) or \
            (action == 'S' and r == 4) or \
            (action == 'E' and c == 4) or \
            (action == 'W' and c == 0):
        s = (r, c)
    elif action == 'N':
        s = (r - 1, c)
    elif action == 'S':
        s = (r + 1, c)
    elif action == 'E':
        s = (r, c + 1)
    elif action == 'W':
        s = (r, c - 1)

    return s


def P(state, action, next_state):
    s = Env(state, action)
    if next_state[0] == s[0] and next_state[1] == s[1]:
        return 1
    else:
        return 0


def next_states(state):
    r = state[0]
    c = state[1]
    if c == 1 and r == 0:
        return [(4, 1)]
    elif c == 3 and r == 0:
        return [(2, 3)]
    return {(r - 1 if r > 0 else r, c),
            (r + 1 if r < 4 else r, c),
            (r, c - 1 if c > 0 else c),
            (r, c + 1 if c < 4 else c)}


def evaluate_policy(gamma=0.9, pi=Pi):
    V = np.zeros((5, 5))
    iters = 0
    converged = False
    while not converged:
        delta = 0
        newV = np.zeros((5, 5))
        for s in S():
            v = V[s]
            sum_over_actions = 0
            for a in A(s):  # this corresponds to the summation over actions
                sum_over_next_states = 0
                for next_state in next_states(s):  # this corresponds to the summation over next states
                    p = P(s, a, next_state)
                    r = R(s, a, next_state)
                    x = gamma * V[next_state]
                    sum_over_next_states += p * (r + x)
                sum_over_actions += pi(s, a) * sum_over_next_states
            newV[s] = sum_over_actions
            delta = np.fmax(delta, abs(newV[s] - v))

        if delta < 1e-5:
            converged = True

        iters += 1
        V = newV

    return V
