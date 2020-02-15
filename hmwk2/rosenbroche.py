import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt
import time

from optimizer import Optimizer

fun_evel = 0

# def matyas(x):
#     return 0.26*(x[0]**2+x[1]**2)-0.48*x[0]*x[1]
# opt = Optimizer(matyas, np.array([1.0,1.0]))

def rosenbrock(x):
    global fun_evel
    fun_evel += 1
    return (1.0-x[0])**2+100.0*(x[1]-x[0]**2)**2
opt = Optimizer(rosenbrock, np.array([-1.0,-1.0]))



# def fun(x):
#     return x[0]**2
# opt = Optimizer(fun, np.array([2.2]))


opt.minimize()
print(fun_evel)
