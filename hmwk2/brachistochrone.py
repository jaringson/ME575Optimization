import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt
import time

from optimizer import Optimizer

funct_eval = 0

def brachistochrone(yint):
    global funct_eval
    funct_eval += 1
    # global fcalls #This is for Dr. Ning's purposes I believe
    # if fcalls > 1e4:
    #     return

    mu_k = 0.3

    y = np.concatenate(([1], yint, [0]), axis=0) #check this is the same
    n = y.size
    x = np.linspace(0.0, 1.0, n)
    g = np.zeros(n-2,dtype=yint.dtype)

    T = 0.0
    for i in range(n-1): #start from 1 or 0? #can i vectorize this??
        ds = np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)

        if 1 - y[i+1] - mu_k * x[i+1] < 0 or 1 - y[i] - mu_k * x[i] < 0:
            T += 10
        else:
            vbar = np.sqrt(1 - y[i+1] - mu_k * x[i+1]) + np.sqrt(1 - y[i] - mu_k * x[i])

            # gradient
            if i > 0:
                dsdyi = 0.5/ds*2 * (y[i+1] - y[i]) * -1
                dvdyi = 0.5 / np.sqrt(1 - y[i] - mu_k * x[i]) * -1
                dtdyi = (vbar * dsdyi - ds*dvdyi)/(vbar**2)
                g[i-1] += dtdyi
            if i < n-2:
                dsdyip = 0.5/ds*2 * (y[i+1] - y[i])
                dvdyip = 0.5 / np.sqrt(1 - y[i+1] - mu_k * x[i+1]) * -1
                dtdyip = (vbar * dsdyip - ds * dvdyip) / (vbar**2)
                g[i] += dtdyip

            T += ds/vbar

    f = T
    # fcalls += 1

    return f


fig = plt.figure()
ax = fig.add_subplot(111)

num_pts = [60]
y0 = np.linspace(1,0,num_pts[0])
y0 = y0[1:-1]

all_times = []
all_wall_times = []
all_function_evals = []
all_nit = []



#### Iterate over number of points
for num in num_pts:

    all_x = np.linspace(0, 1, num)

    start = time.time()
    opt = Optimizer(brachistochrone, y0)
    # print(opt.gradient_func(y0))
    opt.minimize()
    print(funct_eval)
    set_trace()


    # fit = minimize(fun, y0,  method="BFGS")
    end = time.time()

    #### Saving Values
    all_wall_times.append(end-start)
    # all_function_evals.append(fit.nfev)
    # all_nit.append(fit.nit)

    #### Output for debugging
    print("-------------")
    print(num)
    all_times.append(brachistochrone(opt.Xk_1))


    #### Adding to plot
    all_y = np.zeros(num)
    all_y[0] = 1
    all_y[1:-1] = opt.Xk_1
    all_y[-1] = 0

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

# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.scatter(all_x, all_y)
# ax.plot(all_x, all_y, label=num)
# ax.set_title('Brachistochrone 128 Points')
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')


# fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
# plt.gcf().subplots_adjust(left=0.15)
#
# axes[0].set_title("Brachistochrone Dimensionality")
# axes[0].plot(num_pts, all_times)
# axes[0].scatter(num_pts, all_times)
# axes[0].set_ylabel("Travel\nTime (s)")
#
# axes[1].plot(num_pts, all_wall_times)
# axes[1].scatter(num_pts, all_wall_times)
# axes[1].set_ylabel("Solve\nTime (s)")
#
# axes[2].plot(num_pts, all_function_evals)
# axes[2].scatter(num_pts, all_function_evals)
# axes[2].set_ylabel("Function\nEvaluations")
#
# axes[3].plot(num_pts, all_nit)
# axes[3].scatter(num_pts, all_nit)
# axes[3].set_ylabel("Number\nIterations")
# axes[3].set_xlabel('Number of Points')

plt.show()
