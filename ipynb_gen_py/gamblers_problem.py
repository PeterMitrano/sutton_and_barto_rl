
# coding: utf-8

# # Value Iteration for the Gambler's Problem

# Exercise 4.8

# ### In this notebook, we cover the implementation of value iteration for the gambler's problem

# This environment is episodic, and stochastic. The state is the amount of money the gambler has, and the available actions are to bet a certain amount of money. With some fixed percentage, the gambler will win and receive the amount of money bet, and otherwise loses the amount of money bet. The gambler's capital is between 1 and 99 dollars. The episode wins when the gambler has 0 or 100 dollars. Reward is zero on all transitions except those in which the gambler reaches \$100. Since this is episodic, we use no discount $\gamma$, so you won't see it in the code.

# In[368]:

import numpy as np
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer
np.set_printoptions(precision=17, suppress=True)
plt.style.use('bmh')


# In[372]:

converged = False
p = 0.4
V = np.zeros(101)
iters = 0
Vs = []

while not converged:
    # sweep over states and update the value function
    delta = 0
    for s in range(1, 100):
        # update the value of state s
        max_bet = min(s, 100 - s)
        Q = np.zeros(max_bet)
        for a in range(1, max_bet + 1):
            # reward of winning and losing weighted by p and (1 - p)
            # the addition of these two conditions represents the sum of s' in Eq 4.10
            win_capital = s + a
            lose_capital = s - a
            lose_reward = 0 # works cuz episode ends when this happens so reward is limited
            win_reward = 1 if win_capital == 100 else 0
            q = p * (win_reward + V[win_capital]) + (1 - p) * (lose_reward + V[lose_capital])
            Q[a - 1] = q
        v = V[s]
        V[s] = np.max(Q)
        delta = max(delta, abs(V[s] - v))
    
    Vs.append(V.copy())
    iters += 1
    
    if delta == 0.0:
        converged = True
        
# Show graphs of value estimates
plt.xlabel("Capital")
plt.ylabel("Value Estimates")
axes = plt.gca()
axes.set_xlim([1, 99])
axes.set_ylim([0, 1])

plt.plot(Vs[0], label="sweep 1")
plt.plot(Vs[1], label="sweep 2")
plt.plot(Vs[2], label="sweep 3")
plt.plot(Vs[-1], label="sweep {}".format(iters))
plt.legend()
    
plt.show()


# ### Now the Final Policy

# In[373]:

pi = np.ndarray(100)
Qs = [0]

for s in range(1, 100):
    max_bet = min(s, 100 - s)
    Q = np.zeros(max_bet + 1)
    for a in range(1, max_bet + 1):
        # reward of winning and losing weighted by p and (1 - p)
        # the addition of these two conditions represents the sum of s' in Eq 4.10
        win_capital = s + a
        lose_capital = s - a
        lose_reward = 0 # works cuz episode ends when this happens so reward is limited
        win_reward = 1 if win_capital == 100 else 0
        q = p * (win_reward + V[win_capital]) + (1 - p) * (lose_reward + V[lose_capital])
        Q[a] = q
        
    best_a = np.argmax(Q)
    pi[s] = best_a
    Qs.append(Q)
    
plt.plot(pi[1:], marker='o', linestyle='--')
axes = plt.gca()
axes.set_xlim([0, 100])
axes.set_xticks([0, 12, 25, 38, 50, 62, 75, 88, 100])
axes.set_ylim([1, 50])
plt.xlabel("Capital")
plt.ylabel("Final policy (stake)")
plt.show()


# ### That graph looks funny
# 
# First, notice how the graph doesn't exactly match the graph in the book. That's a numerical instability issue that I haven't fixed yet. But first let's explain why the way it should look is correct.
# 
# I think the key is that you're usually going to lose. Whatever you bet, you're more likely to lose that amount. So essentially you should do the opposite of what you want to do.
# 
# For states below 50, you can never win, so you first need to get to 50 or higher. So when you're at $1, 2, ..., 12$, it makes sense to bet all you have. At 13 we bet 12. So why do we bet less after 13? Because you're trying to get back down to 13. So why is 13 is nice? Well if you bet 12 at 13, you get 25. Bet 25 at 25 and you get 50, then 100. Of course being at 80 is better than being at 50, at least according to our monotonically increasing state value function.

# In[375]:

for Q in Qs[1:]:
    Q = Q[1:]
    plt.plot(Q)

plt.xlabel("Capital (how much to bet)")
plt.ylabel("Q Value")
plt.title("\"Q Tree\" for Gambler's Problem p=0.4")
plt.show()


# ** WOAH, that was unexpected. What a pretty graph! Let's see if I can explain what it means**
# 
# ## Let's call it a Q Tree!
# 
# This is a visualization of the action values for the Gambler's problem, with p=0.4. Each colored line represents one state, and in this case there are 99 of them {1 - 99}. The Y axis is the Q value. In this environment, we can also think about it as the probability of winning, since the reward for winning is 1, and the reward for everything else is zero. At 99, the action value for your only available action, betting 1, is 0.96433. We can derive this from the formula:
# 
# \begin{equation}
# \begin{split}
# Q(99, 1) &= P^{1}_{99, 100} [ R_{win} + V(100)] + P^{1}_{99, 98} [ R_{lose} + V(98)] \\
# &=  0.4 * [ 1 + V(100)] + 0.6 * [ 0 + V(98)] \\ 
# \end{split}
# \end{equation}
# 
# So as you get closer to 100, you get closer to that _sweet sweet_ reward of 1, so the action values get higher. The swoopyness still kind of confuses me. The fact that it is triangular, with the middle being the longest, is because at 50 you have the most possible actions/capital to bet.

# In[ ]:



