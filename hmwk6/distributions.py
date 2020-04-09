import numpy as np


def weibull(x, lam, k):

    # lam = 0.1 # lambda
    # k = 1.0 # k

    out = np.zeros_like(x)
    # if x >= 0:
    out[x >= 0] = (k/lam)*(x[x>=0]/lam)**(k-1)*np.exp(-(x[x>=0]/lam)**k)

    return out

def randa():
    ascale = 0.1
    ashape = 1.0
    out = np.random.weibull(ashape) * ascale

    # x = np.random.rand((1))
    # out = weibull(x, ascale, ashape)[0]
    # print(x, out)
    # set_trace()

    return out

def randb():
    bmean = 9
    bsd = 1
    out = np.random.normal(bmean, bsd)
    return out

def randd():
    dmean = 5
    dsd = 0.4
    out = np.random.normal(dmean, dsd)
    return out
