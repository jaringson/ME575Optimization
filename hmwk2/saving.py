import numpy as np

from uncon import uncon
from test_cases import *
from alg_types import *

from IPython.core.debugger import set_trace

option1 = {'line_type':LineType.BACKTRACK, 'dir_type':DirType.STEEPEST}
option2 = {'line_type':LineType.BACKTRACK, 'dir_type':DirType.QUASI}
option3 = {'line_type':LineType.BRACKET, 'dir_type':DirType.STEEPEST}
option4 = {'line_type':LineType.BRACKET, 'dir_type':DirType.QUASI}
all_options = [option1, option2, option3, option4]
all_options = [option4]

results = {}
results['name'] = 'rosenbrock'
results['outputs'] = []

yint = np.linspace(1, 0, 60)
yint = yint[1:-1]

num = 1
for option in all_options:
    print(num)
    num += 1
    # xopt, fopt, outputs = uncon(matyas, [-2.0,2.0], 1e-6, option)
    # xopt, fopt, outputs = uncon(rosenbrock, [-2.0,2.0], 1e-6, option)
    xopt, fopt, outputs = uncon(rosenbrock, [85.25710413, 26.2181083 ], 1e-6, option)
    # xopt, fopt, outputs = uncon(brachistochrone, yint, 1e-6, option)
    results['outputs'].append(outputs)
    print(xopt)
    print(fopt)
    print(outputs['function_calls'])
    # set_trace()

# np.savez(results['name'], results, allow_pickle=True)
