import numpy as np
from scipy.optimize import minimize

from test_cases import *

yint = np.linspace(1, 0, 256)
yint = yint[1:-1]

fit = minimize(rosenbrock, yint,  method="BFGS", jac=True)
print(fit.nfev)
print(fit.x)
