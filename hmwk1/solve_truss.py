import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt

from truss import truss

num_evals = 0

mass = 0
stress = 0

all_functions = []
max_const_vio = []

#### Solve function
def solve(x):
    global num_evals
    num_evals += 1
    mass, stress = truss(x)
    return mass

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


all_constraints = {'type':'ineq', 'fun':constraintAll}
callback = callbackF2

#### Solve
start = np.ones(10) * 0.1

fit = minimize(solve,
    start,
    method="SLSQP",
    constraints=all_constraints,
    options={'disp':True},
    callback=callback)

#### Output values for debugging
print(truss(fit.x))
print(fit.x)
print(num_evals)


#### Plot Results
fig = plt.figure()
ax = fig.add_subplot(211)
ax.set_title("Truss Optimization")
ax.plot(range(len(all_functions)), all_functions)
ax.set_ylabel("Mass (lbs)")

ax = fig.add_subplot(212)
ax.plot(range(len(max_const_vio)), max_const_vio)
plt.gcf().subplots_adjust(left=0.15)
ax.set_ylabel("Max Constraint")
ax.set_xlabel("Iterations")

plt.show()
