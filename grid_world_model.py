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
    y = state[1]
    x = state[0]
    if x == 1 and y == 0:
        return 10
    elif x == 3 and y == 0:
        return 5
    elif (action == 'N' and y == 0) or \
            (action == 'S' and y == 4) or \
            (action == 'E' and x == 4) or \
            (action == 'W' and x == 0):
        return -1
    else:
        return 0


def A(state):
    return ['N', 'S', 'E', 'W']


def Env(state, action):
    y = state[1]
    x = state[0]
    if x == 1 and y == 0:
        s = (1, 4)
    elif x == 3 and y == 0:
        s = (3, 2)
    elif (action == 'N' and y == 0) or \
            (action == 'S' and y == 4) or \
            (action == 'E' and x == 4) or \
            (action == 'W' and x == 0):
        s = (x, y)
    elif action == 'N':
        s = (x, y - 1)
    elif action == 'S':
        s = (x, y + 1)
    elif action == 'E':
        s = (x + 1, y)
    elif action == 'W':
        s = (x - 1, y)

    return s


def P(state, action, next_state):
    s = Env(state, action)
    if next_state[0] == s[0] and next_state[1] == s[1]:
        return 1
    else:
        return 0


def next_states(state):
    y = state[1]
    x = state[0]
    if x == 1 and y == 0:
        return [(1, 4)]
    elif x == 3 and y == 0:
        return [(3, 2)]
    return {(x, y - 1 if y > 0 else y),
            (x, y + 1 if y < 4 else y),
            (x - 1 if x > 0 else x, y),
            (x + 1 if x < 4 else x, y)}


def evaluate_policy(gamma=0.9, pi=Pi, r_=R, p_=R):
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
                    p = p_(s, a, next_state)
                    r = r_(s, a, next_state)
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
