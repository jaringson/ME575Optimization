import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from IPython.core.debugger import set_trace

from pyDOE import *
import scipy.stats as stats

from farmcost import farmcost
from distributions import *

y0 = [1, 2]



# plt.figure()
# x = np.linspace(-0.2,1,200)
# ascale = 0.1
# ashape = 1.0
# pdf = weibull(x, ascale, ashape)
# # plt.plot(x, pdf) # including h here is crucial
# count, bins, ignored = plt.hist(np.random.weibull(ashape,10000) * ascale, 100)
# plt.show()

# print(weibull(np.array([0]), 1, ashape))

# try increasing large samples
nvec = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

# nvec = [25600*4]
# initialize the array of results from each sample size
num = 10
plt.figure()

for k in range(num):
    cost = np.zeros(len(nvec))
    for i in range(len(nvec)):  # repeat Monte Carlo for many different sample sizes
        n = nvec[i]


        # initialize for Monte Carlo
        costvec = np.zeros((n, 1));

        for j in range(n):  # loop over all of the trials
            costvec[j] = farmcost(y0, randa(), randb(), randd());


        # TODO: compute some statistic using costvec (e.g., mean, prctile, etc.)
        # cost[i] = np.mean(costvec)
        # cost[i] = np.std(costvec) + np.mean(costvec)
        cost[i] = np.percentile(costvec, 95)

        # print(cost)

        # plt.figure()
        # plt.hist(costvec,500)
        # plt.show()

    plt.semilogx(nvec, cost)

plt.title('95th Percentile Convergence')
plt.xlabel('Number of Samples')
plt.ylabel('95th Percentile')
plt.show()
