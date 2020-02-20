import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import time
from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt

from truss_Adjoint import truss

num_evals = 0

mass = 0
stress = 0

all_functions = []
max_const_vio = []

num_rand = 25
all_As = np.random.uniform(0.1,20,(num_rand,10))

### # Solve function
def solve(x, count=True):
    if count:
        global num_evals
        num_evals += 1
    mass, dmdA, stress, dsdA = truss(x)
    return mass, dmdA

def solve2(x, count=True):
    if count:
        global num_evals
        num_evals += 1
    mass, dmdA, stress, dsdA = truss(x)
    return mass

#### Constraint Function
def constraintAll(x, count=True):
    # if count:
    #     global num_evals
    #     num_evals += 1
    mass, dmdA, stress, dsdA = truss(x)

    area = x - 0.1
    stress_ub = 25e3 - stress
    stress_ub[8] = 75e3 - stress[8]
    stress_lb = stress + 25e3
    stress_lb[8] = stress[8] + 75e3

    # set_trace()
    return np.hstack((area,stress_ub,stress_lb))

def get_dsdA(x):
    # global num_evals
    # num_evals += 1
    mass, dmdA, stress, dsdA = truss(x)
    return np.vstack((np.eye(10),-dsdA,dsdA))

#### Callback function
def callbackF2(Xi):
    global all_functions, max_const_vio
    # print(solve(Xi, count=False)[0])
    all_functions.append(solve(Xi, count=False)[0])
    max = np.max(np.abs(constraintAll(Xi, count=False)))

    max_const_vio.append(max)


def run():
    all_data = []

    all_constraints = {'type':'ineq', 'fun':constraintAll, 'jac':get_dsdA}
    all_constraints2 = {'type':'ineq', 'fun':constraintAll}
    callback = callbackF2




    global all_As

    for A0 in all_As:
        #### Solve

        start = time.time()
        fit = minimize(solve,
            A0,
            method="SLSQP",
            jac=True,
            constraints=all_constraints,
            options={'disp':True},
            callback=callback)
        end = time.time()

        # fit = minimize(solve2,
        #     start,
        #     method="SLSQP",
        #     constraints=all_constraints2,
        #     options={'disp':True},
        #     callback=callback)

        #### Output values for debugging
        # print(truss(fit.x))
        # print('Optimum: ', fit.x)
        # print('Function Calls: ', num_evals)
        # print('Time: ', end-start)
        global num_evals, all_functions, max_const_vio
        data = {'time':end-start, 'num_evals':num_evals, 'funct_evals':all_functions, 'const_vio': max_const_vio}
        all_data.append(data)

        num_evals = 0
        all_functions = []
        max_const_vio = []
    return all_data




# for i in range(2):
#     if i == 0:
#         from truss_FD import truss
#     else:
#         from truss_Adjoint import truss


from truss_FD import truss
FD_all_data = run()
from truss_Adjoint import truss
Adjoint_all_data = run()

FD_ave_time = 0.0
FD_ave_eval = 0.0
Adjoint_ave_time = 0.0
Adjoint_ave_eval = 0.0
for i in range(len(FD_all_data)):
    FD_ave_time += FD_all_data[i]['time']
    FD_ave_eval += FD_all_data[i]['num_evals']
    Adjoint_ave_time += Adjoint_all_data[i]['time']
    Adjoint_ave_eval += Adjoint_all_data[i]['num_evals']

FD_ave_time /= num_rand
FD_ave_eval *= 2
FD_ave_eval *= 10
FD_ave_eval /= num_rand
Adjoint_ave_time /= num_rand
Adjoint_ave_eval /= num_rand

print('FD:')
print('\tAverage Time: ', FD_ave_time)
print('\tAverage Number of Evals: ', FD_ave_eval)

print('Adjoint:')
print('\tAverage Time: ', Adjoint_ave_time)
print('\tAverage Number of Evals: ', Adjoint_ave_eval)

#### Plot Results
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
plt.gcf().subplots_adjust(left=0.15)
# fig = plt.figure()
# ax = fig.add_subplot(211,)
axes[0].set_title("Truss Optimization")
axes[0].plot(range(0,len(FD_all_data[0]['funct_evals'])*20,20), FD_all_data[0]['funct_evals'], color='green', label='Finite-Difference')
axes[0].plot(range(len(Adjoint_all_data[0]['funct_evals'])), Adjoint_all_data[0]['funct_evals'], color='orange', label='Adjoint')
for i in range(1,len(FD_all_data)):
    axes[0].plot(range(0,len(FD_all_data[i]['funct_evals'])*20,20), FD_all_data[i]['funct_evals'], color='green')
    axes[0].plot(range(len(Adjoint_all_data[i]['funct_evals'])), Adjoint_all_data[i]['funct_evals'], color='orange')
axes[0].set_ylabel("Mass (lbs)")
axes[0].set_ylim([0,2000])
axes[0].legend()


axes[1].plot(range(0,len(FD_all_data[0]['const_vio'])*20,20), FD_all_data[0]['const_vio'], color='green', label='Finite-Difference')
axes[1].plot(range(len(Adjoint_all_data[0]['const_vio'])), Adjoint_all_data[0]['const_vio'], color='orange', label='Adjoint')
for i in range(1,len(FD_all_data)):
    axes[1].plot(range(0,len(FD_all_data[i]['const_vio'])*20,20), FD_all_data[i]['const_vio'], color='green')
    axes[1].plot(range(len(Adjoint_all_data[i]['const_vio'])), Adjoint_all_data[i]['const_vio'], color='orange')
axes[1].set_ylabel("Max Constraint")
axes[1].set_xlabel("Function Calls")
axes[1].set_ylim([0,2e6])
axes[1].legend()

plt.show()

set_trace()

#### 20 More ####
# FD:
# 	Average Time:  4.094418354034424
# 	Average Number of Evals:  1014.4
# Adjoint:
# 	Average Time:  0.5342223644256592
# 	Average Number of Evals:  43.36


#### 10 More ####
# FD:
# 	Average Time:  3.3085175704956056
# 	Average Number of Evals:  386.0
# Adjoint:
# 	Average Time:  0.25411544799804686
# 	Average Number of Evals:  32.96
