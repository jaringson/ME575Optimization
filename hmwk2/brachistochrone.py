import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt
import time

from optimizer import Optimizer

start_pt = np.array([0,1])
end_pt = np.array([1,0])

# num_pts = [4,8,16,32,64,128,256]
num_pts = [60,4,8,16,32,64,128]
# num_pts = []

linear_start = True

mu = 0.3
H = start_pt[1]


#### Function to evaluate Brachistochrone time
def fun(y, g=0):

    all_y = np.zeros(num)
    all_y[0] = start_pt[1]
    all_y[1:-1] = y
    all_y[-1] = end_pt[1]

    sum = 0
    for i in range(0,num-1):

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

    return sum


fig = plt.figure()
ax = fig.add_subplot(111)

y0 = np.ones((num_pts[0]-2))*0

all_times = []
all_wall_times = []
all_function_evals = []
all_nit = []



#### Iterate over number of points
for num in num_pts:
    if linear_start:
        y0 = np.linspace(1-1.0/num,1.0/num,num-2)

    all_x = np.linspace(start_pt[0], end_pt[0], num)

    start = time.time()
    opt = Optimizer(fun, y0)
    opt.minimize()
    set_trace()
    # fit = minimize(fun, y0,  method="BFGS")
    end = time.time()

    #### Saving Values
    all_wall_times.append(end-start)
    all_function_evals.append(fit.nfev)
    all_nit.append(fit.nit)

    #### Output for debugging
    print("-------------")
    print(num)
    all_times.append(fun(fit.x, g = 9.81))

    if not linear_start:
        y0 = []
        for i in range(len(fit.x)+1):
            if i == 0:
                y0.append((start_pt[1]+fit.x[i])/2.0)
                y0.append(fit.x[i])
                continue
            if i == len(fit.x):
                y0.append((fit.x[i-1]+end_pt[1])/2.0)
                y0.append((fit.x[i-1]+end_pt[1])/4.0)
            else:
                y0.append((fit.x[i-1]+fit.x[i])/2.0)
                y0.append(fit.x[i])

    #### Adding to plot
    all_y = np.zeros(num)
    all_y[0] = start_pt[1]
    all_y[1:-1] = fit.x
    all_y[-1] = end_pt[1]

    ax.scatter(all_x, all_y,s=3)
    ax.plot(all_x, all_y, label=num)

#### Truth Data for no friction
theta = np.linspace(0,2.412,100)
a = 0.572917
truth_x = a*(theta-np.sin(theta))
truth_y = -a*(1-np.cos(theta))+1


ax.set_title('Brachistochrone')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
# ax.plot(truth_x, truth_y)
ax.legend()

print(all_times)
print(np.average(all_times))

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(all_x, all_y)
ax.plot(all_x, all_y, label=num)
ax.set_title('Brachistochrone 128 Points')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')


fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
plt.gcf().subplots_adjust(left=0.15)

axes[0].set_title("Brachistochrone Dimensionality")
axes[0].plot(num_pts, all_times)
axes[0].scatter(num_pts, all_times)
axes[0].set_ylabel("Travel\nTime (s)")

axes[1].plot(num_pts, all_wall_times)
axes[1].scatter(num_pts, all_wall_times)
axes[1].set_ylabel("Solve\nTime (s)")

axes[2].plot(num_pts, all_function_evals)
axes[2].scatter(num_pts, all_function_evals)
axes[2].set_ylabel("Function\nEvaluations")

axes[3].plot(num_pts, all_nit)
axes[3].scatter(num_pts, all_nit)
axes[3].set_ylabel("Number\nIterations")
axes[3].set_xlabel('Number of Points')

plt.show()
