from __future__ import print_function


import numpy as np


from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt

from truss import truss



import numpy
import argparse
from pyoptsparse import Optimization, OPT


parser = argparse.ArgumentParser()
parser.add_argument("--opt",help="optimizer",type=str, default='SLSQP')
args = parser.parse_args()
optOptions = {}

#### Objective function
def objfunc(xx):
    x = xx['xvars']
    mass = -1
    stress = [-1]*10
    try:
        fail = False
        mass, stress = truss(x)
    except:
        print('error')
        fail = True
    # set_trace()

    funcs = {}
    funcs['obj'] = mass

    conval = [0]*10
    for i in range(10):
        conval[i] = stress[i]

    funcs['con'] = conval

    return funcs, fail

#### Constraint Function
def constraintAll(x):
    mass, stress = truss(x)
    global num_evals
    num_evals += 1

    area = x - 0.1
    stress_ub = 25e3 - stress
    stress_ub[8] = 75e3 - stress[8]
    stress_lb = stress + 25e3
    stress_lb[8] = stress[8] + 75e3

    return np.hstack((area,stress_ub,stress_lb))

#### Callback function
def callbackF2(Xi):
    temp = 1
    global Nfeval, all_functions, max_const_vio
    print(solve(Xi))
    all_functions.append(solve(Xi))
    max = np.max(np.abs(constraintAll(Xi)))

    max_const_vio.append(max)



#### Solve
# Optimization Object
optProb = Optimization('Truss Problem', objfunc)

# Design Variables
optProb.addVarGroup('xvars', 10, 'c',lower=[0.1]*10, upper=None, value=[0.1]*10)

lower = [-25e3]*10
lower[8] = -75e3
upper = [25e3]*10
upper[8] = 75e3

# set_trace()

# Constraints
optProb.addConGroup('con', 10, lower=lower, upper=upper)

# Objective
optProb.addObj('obj')

# Check optimization problem:
print(optProb)

# Optimizer
opt = OPT(args.opt, options=optOptions)

# set_trace()

# Solution
sol = opt(optProb, sens='FD')

# Check Solution
print(sol)

# start = np.ones(10) * 0.1
#
# fit = minimize(solve,
#     start,
#     method="SLSQP",
#     constraints=all_constraints,
#     options={'disp':True},
#     callback=callback)
#
# #### Output values for debugging
# print(truss(fit.x))
# print(fit.x)
# print(num_evals)
#
#
# #### Plot Results
# fig = plt.figure()
# ax = fig.add_subplot(211)
# ax.set_title("Truss Optimization")
# ax.plot(range(len(all_functions)), all_functions)
# ax.set_ylabel("Mass (lbs)")
#
# ax = fig.add_subplot(212)
# ax.plot(range(len(max_const_vio)), max_const_vio)
# plt.gcf().subplots_adjust(left=0.15)
# ax.set_ylabel("Max Constraint")
# ax.set_xlabel("Iterations")
#
# plt.show()
