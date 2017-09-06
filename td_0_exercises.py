
# coding: utf-8

# # Exercises 6.1 - 6.5
# 
# ## TD(0) Learning
# 
# **Exericise 6.1: for the driving example, exmplain why TD(0) is better than Monte Carlo when you change the starting state. Lets say you have lots of experience driving home but now you start in a different parking lot.**
# 
# With TD(0), the transition from the parking lot to the road (which was part of your old route) will cause the state-value of the new parking lot to include the state-value of the rest of your trip (from the road to your home). If this estimate is already very good, then you're state-value of the new parking lot will instantly become very good. For Monte-Carlo methods, your first state-value estiamte of the new parking lot is based only on that one sample trip home, and therefore could be very inaccurate. Also, you have to wait till you get home to assign state-value to the new parking lot. But the more important difference is that in TD your update is based on all past samples relavent to the route, but in Monte Carlo they are not.
# 
# **Exercise 6.2: In figure 6.6, explain why only V(A) changes and say exactly how much**
# 
# In the scenario where the agent starts at A and immediately goes right, the only update that would occur would be as follows. The resulting change is by $-0.05$
# \begin{equation}
# \begin{split}
# V(A) &= V(A) - 0.1 ( r + V(L)- V(A)) \\
#      &= 0.5 - 0.1 ( 0 - 0.5) \\
#      &= 0.45 \\
# \end{split}
# \end{equation}
# 
# **Exercise 6.3: **

# In[ ]:



