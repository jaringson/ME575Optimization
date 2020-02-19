import numpy as np
from scipy.optimize import minimize

from uncon import uncon
from test_cases import *
from alg_types import *

from IPython.core.debugger import set_trace

option1 = {'line_type':LineType.BACKTRACK, 'dir_type':DirType.STEEPEST}
option2 = {'line_type':LineType.BACKTRACK, 'dir_type':DirType.QUASI}
option3 = {'line_type':LineType.BRACKET, 'dir_type':DirType.STEEPEST}
option4 = {'line_type':LineType.BRACKET, 'dir_type':DirType.QUASI}
# all_options = [option1, option2, option3, option4]
all_options = [option4]

labels = ['Steepest Backtrack', 'Quasi-Newton Backtrack', 'Steepest N-W', 'Quasi-Newton N-W']

num_of_sp = 10
# starting_points = np.random.uniform(-100,100,(num_of_sp,2))
starting_points = np.array([[-62.30386999, -25.83768453],
[-21.0403251 , -14.02330778],
[  9.70891713, -67.46296481],
[-96.17539178, -80.59439791],
[-1.84957355, 25.32791886],
[63.62334595, 76.05005117],
[-15.88931528, -85.03483885],
[19.42054044, 67.07047287],
[ 47.82754075, -57.23680417],
[85.25710413, 26.2181083 ]])

yint = np.linspace(1, 0, 256)
yint = yint[1:-1]

starting_points = [yint]

num = 0
for option in all_options:
    print(labels[num])
    # print(num)
    total = 0
    num += 1
    for start in starting_points:
        print(start)
        # set_trace()
        xopt, fopt, outputs = uncon(brachistochrone, start, 1e-6, options=None)
        # xopt, fopt, outputs = uncon(brachistochrone, start, 1e-6, options=None)
        total += outputs['function_calls']
        print("Calls: ", outputs['function_calls'])
    average = total / num_of_sp
    print("Average: ", average)

total_scipy = 0
for start in starting_points:
    fit = minimize(brachistochrone, start,  method="BFGS", jac=True, options={'gtol':1e-6})
    print(fit.nfev)
    # print(fit.x)
    total_scipy += fit.nfev
print('Scipy Average: ', total_scipy / num_of_sp)
