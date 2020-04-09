import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from IPython.core.debugger import set_trace

from pyDOE import *
import scipy.stats as stats

from farmcost import farmcost
from distributions import *


def solve(x):
    global alpha, beta, delta

    yback = x #np.array([2., 5.])

    cost = farmcost(yback, alpha, beta, delta, plotit=False)
    # print(cost)
    return cost

def constraintAll(x):

    return x[1] - x[0]



# print(farmcost(fit.x, alpha, beta, delta, plotit=True))

def ouusolve(x, randa_l, randb_l, randd_l, n):

    # initialize for Monte Carlo
    costvec = np.zeros((n, 1));

    for j in range(n):  # loop over all of the trials
        costvec[j] = farmcost(x, randa_l[j], randb_l[j], randd_l[j]);


    # TODO: compute some statistic using costvec (e.g., mean, prctile, etc.)
    # cost = np.mean(costvec)
    # cost[i] = np.std(costvec) + np.mean(costvec)
    cost = np.percentile(costvec, 95)
    # print(x,cost)
    return cost

alpha = 0.1
beta = 9
delta = 5

y0 = [3, 9]
bounds = [(0,10),(0,10)]
all_constraints = {'type':'ineq', 'fun':constraintAll}


D_fit = minimize(solve,
    y0,
    method="SLSQP",
    constraints=all_constraints,
    options={'disp':True},
    # callback=callback,
    bounds=bounds)

n = 3200
randa_l = []
randb_l = []
randd_l = []
for i in range(n):
    randa_l.append(randa())
    randb_l.append(randb())
    randd_l.append(randd())

U_fit = minimize(ouusolve,
    y0,
    method="SLSQP",
    constraints=all_constraints,
    options={'disp':True},
    args=(randa_l, randb_l, randd_l, n),
    # callback=callback,
    bounds=bounds)


print("DD:", farmcost(D_fit.x, alpha, beta, delta, plotit=False))
print("DU:", ouusolve(D_fit.x, randa_l, randb_l, randd_l, n))

print("UD:", farmcost(U_fit.x, alpha, beta, delta, plotit=False))
print("UU:", ouusolve(U_fit.x, randa_l, randb_l, randd_l, n))
