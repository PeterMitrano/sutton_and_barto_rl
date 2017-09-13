
# coding: utf-8

# # Exercise 5.6
# 
# Derive the weighted-average update rule (5.5) from (5.4). Follow the pattern of the derivation of the unweighted rule (2.4) from (2.1).

# In[4]:

from IPython.display import Image
from IPython.core.display import HTML 
Image(url="https://media.giphy.com/media/BmmfETghGOPrW/giphy.gif")


# ## LETS DO THIS!
# 
# \begin{equation}
# \begin{split}
# V_n &= \frac{\sum^n_{k=1} w_k R_k}{\sum^n_{k=1} w_k} \\
# V_{n+1} &= \frac{\sum^{n+1}_{k=1} w_k R_k}{\sum^{n+1}_{k=1} w_k} \\
#         &= \frac{1}{\sum^{n+1}_{k=1} w_k} \sum^{n+1}_{k=1} w_k R_k \\
#         &= \frac{1}{\sum^{n+1}_{k=1} w_k} [w_{n+1} R_{n+1} + \sum^{n}_{k=1} w_k R_k] &&\text{(expand the sum one step)} \\
#         &= \frac{1}{\sum^{n+1}_{k=1} w_k} [w_{n+1} R_{n+1} + \sum^n_{k=1} w_k * V_n] &&\text{(use the original formula for $V_n$)} \\
#         &= \frac{1}{\sum^{n+1}_{k=1} w_k} [w_{n+1} R_{n+1} + \sum^n_{k=1} w_k*V_n + w_{n+1}V_n - w_{n+1}V_n] \quad && \text{(add and subtract $w_{n+1}V_n$. Just sprinkling a litle bit of  magic on it)} \\
#         &= \frac{1}{\sum^{n+1}_{k=1} w_k} [w_{n+1} R_{n+1} + (\sum^n_{k=1} w_k + w_{n+1})V_n - w_{n+1}V_n]  &&\text{(now you see where I've going with this)}\\
#         &= \frac{1}{\sum^{n+1}_{k=1} w_k} [(\sum^n_{k=1} w_k + w_{n+1})*V_n + w_{n+1} R_{n+1} - w_{n+1}V_n] \\
#         &= \frac{1}{\sum^{n+1}_{k=1} w_k} [\sum^{n+1}_{k=1} w_k * V_n + w_{n+1} R_{n+1} - w_{n+1}*V_n] &&\text{(ohhhh yeaaaa. Combine the sum!)} \\
#         &= V_n + \frac{1}{\sum^{n+1}_{k=1} w_k} [w_{n+1} R_{n+1} - w_{n+1}*V_n] \\
#         &= V_n + \frac{w_{n+1}}{\sum^{n+1}_{k=1} w_k} [R_{n+1} -V_n] \\
#         &= V_n + \frac{w_{n+1}}{W_{n+1}} [R_{n+1} -V_n] &&\text{(replace with $W_{n+1}$)} \\
# \end{split}
# \end{equation}
