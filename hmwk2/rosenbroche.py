import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt
import time

from optimizer import Optimizer

# def matyas(x):
#     return 0.26*(x[0]**2+x[1]**2)-0.48*x[0]*x[1]
# opt = Optimizer(matyas, np.array([1.0,1.0]))

def rosenbrock(x):
    return (1.0-x[0])**2+100.0*(x[1]-x[0]**2)**2
opt = Optimizer(rosenbrock, np.array([0.0,0.0]))



# def fun(x):
#     return x[0]**2
# opt = Optimizer(fun, np.array([2.2]))


opt.minimize()
