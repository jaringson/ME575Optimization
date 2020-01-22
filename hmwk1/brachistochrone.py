import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt

start_pt = np.array([0,1])
end_pt = np.array([1,0])

num_pts = 64
# all_x1 = np.linspace(start_pt[0], 0.1, num_pts//4)
# all_x2 = np.linspace(0.1, end_pt[0], num_pts - all_x1.shape[0])
# all_x = np.concatenate((all_x1,all_x2))


all_x = np.linspace(start_pt[0], end_pt[0], num_pts)

mu = 0.3
H = start_pt[1]
# set_trace()

def fun(y, g=0):

    all_y = np.zeros(num_pts)
    all_y[0] = start_pt[1]
    all_y[1:-1] = y
    all_y[-1] = end_pt[1]
     # = np.concatenate((start_pt[1], x, end_pt[1]))
    sum = 0
    # set_trace()
    for i in range(0,num_pts-1):
        # print(i)
        xi = all_x[i]
        yi = all_y[i]
        xip1 = all_x[i+1]
        yip1 = all_y[i+1]
        dx = xip1-xi
        dy = yip1-yi
        if g == 0:
            sum += np.sqrt(dx**2+dy**2)/(np.sqrt(H-yip1-mu*xip1)+np.sqrt(H-yi-mu*xi))
        else:
            sum += np.sqrt(2/g) * np.sqrt(dx**2+dy**2)/(np.sqrt(H-yip1-mu*xip1)+np.sqrt(H-yi-mu*xi))
        # sum += yi - 1
    # print(sum)
    return sum

def constraint1(x):
    return 1-x[0]**2-x[1]**2

def constraint2(x):
    return 5-x[0]-3*x[1]

# c1 = NonlinearConstraint()


c1 = {'type':'ineq', 'fun':constraint1}
c2 = {'type':'ineq', 'fun':constraint2}

# y0 = np.ones((num_pts-2))*0
y0 = np.linspace(0.9,0.1,num_pts-2)

# for i in range(num_pts//2-2):
#     y0[2*i] = intermediate[i]
#     y0[2*i+1] = intermediate[i]

# fit = minimize(fun, start,  method="SLSQP", constraints=(c1,c2), options={'disp':True})
fit = minimize(fun, y0,  method="BFGS", options={'disp':True})
# print(fit.x)
print(fun(fit.x))
print(fun(fit.x, g = 9.81))

all_y = np.zeros(num_pts)
all_y[0] = start_pt[1]
all_y[1:-1] = fit.x
all_y[-1] = end_pt[1]

theta = np.linspace(0,2.412,100)
a = 0.572917
truth_x = a*(theta-np.sin(theta))
truth_y = -a*(1-np.cos(theta))+1

fig = plt.figure()
ax = fig.add_subplot(111)
# ax.set_title('Velocity')
ax.scatter(all_x, all_y)
ax.plot(all_x, all_y)
ax.plot(truth_x, truth_y)

plt.show()
