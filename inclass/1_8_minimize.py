import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

def fun(x):
    return (1-x[0])**2+100*(x[1]-x[0]**2)**2

def constraint1(x):
    return 1-x[0]**2-x[1]**2

def constraint2(x):
    return 5-x[0]-3*x[1]

# c1 = NonlinearConstraint()


c1 = {'type':'ineq', 'fun':constraint1}
c2 = {'type':'ineq', 'fun':constraint2}
start = [1,1]

fit = minimize(fun, start,  method="SLSQP", constraints=(c1,c2), options={'disp':True})
print(fit.x)
