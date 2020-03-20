import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt
import time
from copy import deepcopy

fun_evals = 0

def obj_rosenbrock(x, n):
    global fun_evals
    fun_evals += 1
    f = 0
    g = np.zeros_like(x)
    for i in range(n-1):
        f += 100.0*(x[i+1]-x[i]**2)**2+(1.0-x[i])**2
        g[i] += -400*x[i]*(x[i+1]-x[i]**2)-2*(1-x[i])
        g[i+1] += 200*(x[i+1]-x[i]**2)
    return f, g


def rosenbrock(x, n, grad, finite):

    f, g = obj_rosenbrock(x, n)
    perturb = 1e-6

    if finite:
        for i in range(n):
            Xp = deepcopy(x)
            Xp[i] += perturb
            f_p, _ = obj_rosenbrock(Xp, n)

            # Xm = deepcopy(x)
            # Xm[i] -= perturb
            # f_m, _ = obj_rosenbrock(Xm, n)

            # g[i] = (f_p - f_m) / (2.0 * perturb)
            g[i] = (f_p - f) / (perturb)


    if grad or finite:
        return f, g
    return f

n = 64

x0 = np.zeros(n)

grad = False
finite = False

fit = minimize(rosenbrock,
    x0,
    #method="BFGS",
    method="Nelder-Mead",
    args=(n, grad, finite),
    jac=grad,
    options={'disp':True, 'maxiter': 5000000, 'adaptive':True})
print(fit.x)

# import nevergrad as ng
#
# def ng_rosenbrock(x):
#     global fun_evals
#     fun_evals += 1
#     f = 0
#
#     for i in range(len(x)-1):
#         f += 100.0*(x[i+1]-x[i]**2)**2+(1.0-x[i])**2
#
#     return f
#
# # set_trace()
#
# optimizer = ng.optimizers.DE(parametrization=n, budget=100)
# recommendation = optimizer.minimize(ng_rosenbrock)  # best value
# print(recommendation.value)


print(fun_evals)
