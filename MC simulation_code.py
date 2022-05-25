'''
Create on : 2021.11.5
@author   : ivy
'''

from math import exp, sqrt
from random import gauss, seed
import matplotlib.pyplot as plt
seed(200)
S_0 = 4.27   # the initial price of stock;
K1 = 0.1       # strike price, whose underlying is call option
K2 = 5           # strike price, whose underlying is stock
T2 = 1/12        # maturity
T1 = 0.5
r = 0.0291       # risk-free interest rate
div = 0.061732
sigma = 0.221843    # volatility

I = 100      # number of simulation
J = 100


path2 = []
for j in range(J):
    path1 = []
    for i in range(I):
        z = gauss(0.0, 1.0)
        S_t = S_0 * exp((r-div - 0.5 * sigma ** 2) * (T1 + T2) + sigma * sqrt(T1 + T2) * z)
        PREMIUM = exp(-r * T2) * max(S_t - K2, 0)
        path1.append(PREMIUM)
    C_0 = sum(path1) / I
    payoff = max(C_0-K1, 0)
    path2.append(payoff)
# pv of call option on date T1
C_Y = exp(-r * T1) * sum(path2)/J
print(C_Y)
# print price of derivative Y
'''
#change the parameters
#initial
r=0.0291           #riskfree-rate
sigma=0.221843
S0=4.27
K1=0.1
K2=5
T1=0.5
T2=0.5

#list of change
K1_list=[0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2]
K2_list=[4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.4, 5.6, 5.8, 6]
T1_list=[1/12,2/12,3/12,6/12,9/12,12/12]
T2_list=[1/12,2/12,3/12,6/12,9/12,12/12]

'''


