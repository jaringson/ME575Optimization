import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt
import time

from optimizer import Optimizer
from uncon import uncon

fun_evel = 0

# def matyas(x):
#     global fun_evel
#     fun_evel += 1
#     f = 0.26*(x[0]**2+x[1]**2)-0.48*x[0]*x[1]
#     g = np.zeros_like(x)
#     g[0] = 0.52*x[0]-0.48*x[1]
#     g[1] = 0.52*x[1]-0.48*x[0]
#     return f, g
# print(uncon(matyas, np.array([2.0, 2.0]), 1e-6))

def rosenbrock(x):
    global fun_evel
    fun_evel += 1
    f = (1.0-x[0])**2+100.0*(x[1]-x[0]**2)**2
    g = np.zeros_like(x)
    g[0] = -400*x[0]*(x[1]-x[0]**2)-2*(1-x[0])
    g[1] = 200*(x[1]-x[0]**2)
    return f, g

print(uncon(rosenbrock, np.array([-1.0,-1.0]), 1e-6))

# def fun(x):
#     return x[0]**2
# opt = Optimizer(fun, np.array([2.2]))


# opt.minimize()
print(fun_evel)
# print(opt.Xk_1)
