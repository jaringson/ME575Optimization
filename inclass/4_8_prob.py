'''
The provided function takes in a vector x of length 3
and returns a scalar objective f to be minimized,
a vector of constraints c to be enforced as c <= 0,
a vector gf containing the gradients of f,
and a matrix gc containing the gradients of the constraints
(gc[i, j] = partial c_i / partial x_j).
In addition to the nonlinear constraints, you should also
enforce the constraint that x >= 0.
The deterministic optimum of this problem is:
x^* = [0.06293586, 2.91716569, 0.07491246], f^*=1.158963

Using the simplified reliability-based optimization methodology
discussed in the text, find the reliable optimum under the
assumption that the inputs are normally distributed and have
a standard deviation of
sigma_1 = 0.01, sigma_2 = 0.1, sigma_3 = 0.05.
Each constraint should be satisfied with a target
reliability of 99.5%.
Briefly describe the how/why of your approach
and turn in your code.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from IPython.core.debugger import set_trace

# function [f, c, gf, gc] = prob(x)
def prob(x):

    f = -x[0]*x[1]*x[2] + (x[1] - 4)**2 + 10*x[0]**3*x[2]

    # c = np.zeros((2, 1))
    # c[0] = 2*x[0]**2 + x[1] + x[2] - 3.0;
    # c[1] = -2*x[0] - 4*x[1]*x[2] + 1.0;

    gf = np.zeros(3);
    gf[0] = -x[1]*x[2] + 30*x[0]**2*x[2];
    gf[1] = -x[0]*x[2] + 2*(x[1] - 4);
    gf[2] = -x[0]*x[1] + 10*x[0]**3;

    # gc = np.zeros((2, 3));
    #
    # gc[0, 1] = 4*x[0];
    # gc[0, 1] = 1.0;
    # gc[0, 2] = 1.0;
    #
    # gc[1, 0] = -2.0;
    # gc[1, 1] = -4*x[2];
    # gc[1, 2] = -4*x[1];

    # set_trace()
    return f, gf

def constraints(x):
    c = np.zeros(2)
    c[0] = 2*x[0]**2 + x[1] + x[2] - 3.0
    c[1] = -2*x[0] - 4*x[1]*x[2] + 1.0
    return -c

def grad_contraints(x):
    global k, sig_g
    gc = np.zeros((2, 3));

    gc[0, 0] = 4*x[0];
    gc[0, 1] = 1.0;
    gc[0, 2] = 1.0;

    gc[1, 0] = -2.0;
    gc[1, 1] = -4*x[2];
    gc[1, 2] = -4*x[1];
    if k > 0:
        # set_trace()
        gc[:,0] += - k * np.squeeze(sig_g)
        gc[:,1] += - k * np.squeeze(sig_g)
        gc[:,2] += - k * np.squeeze(sig_g)
    return -gc


# print(constraints([0.06293586, 2.91716569, 0.07491246]))

k = 0
sig_g = 0

y0 = [0, 0, 0]
bounds = [(0,100), (0,100), (0,100)]
all_constraints = {'type':'ineq', 'fun':constraints, 'jac':grad_contraints}


fit = minimize(prob,
    y0,
    method="SLSQP",
    constraints=all_constraints,
    options={'disp':True},
    jac=True,
    # args=(randa_l, randb_l, randc_l, n),
    # callback=callback,
    bounds=bounds)
print(fit.x)
print(constraints(fit.x))

con_grad = grad_contraints(fit.x)

sig_g = np.zeros((1, 2))
sig_x = np.array([0.01, 0.1, 0.05])
for i in range(3):
    # set_trace()
    sig_g += (con_grad[:,i] * sig_x[i])**2


k = 0.995**3
print(k)

fit = minimize(prob,
    fit.x,
    method="SLSQP",
    constraints=all_constraints,
    options={'disp':True},
    jac=True,
    # args=(randa_l, randb_l, randc_l, n),
    # callback=callback,
    bounds=bounds)
print(fit.x)
print(constraints(fit.x))

# n = 25600
# randa_l = []
# randb_l = []
# randc_l = []
# for i in range(n):
#     randa_l.append(np.random.normal(0, 0.01))
#     randb_l.append(np.random.normal(0, 0.1))
#     randc_l.append(np.random.normal(0, 0.05))
#
# U_fit = minimize(ouu_prob,
#     y0,
#     method="SLSQP",
#     constraints=all_constraints,
#     options={'disp':True},
#     args=(randa_l, randb_l, randc_l, n),
#     # callback=callback,
#     bounds=bounds)
# print(U_fit.x)
