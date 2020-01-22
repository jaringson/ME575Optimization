import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt

import time

start_pt = np.array([0,1])
end_pt = np.array([1,0])

num_pts = [4,8,16,32,64,128]
# all_x1 = np.linspace(start_pt[0], 0.1, num_pts//4)
# all_x2 = np.linspace(0.1, end_pt[0], num_pts - all_x1.shape[0])
# all_x = np.concatenate((all_x1,all_x2))




mu = 0.3
H = start_pt[1]
# set_trace()

def fun(y, g=0):

    all_y = np.zeros(num)
    all_y[0] = start_pt[1]
    all_y[1:-1] = y
    all_y[-1] = end_pt[1]
     # = np.concatenate((start_pt[1], x, end_pt[1]))
    sum = 0
    for i in range(0,num-1):
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


fig = plt.figure()
ax = fig.add_subplot(111)

y0 = np.ones((num_pts[0]-2))*0

all_times = []
all_wall_times = []
# all_x = []
for num in num_pts:
    y0 = np.linspace(1-1.0/num,1.0/num,num-2)
    # set_trace()

    all_x = np.linspace(start_pt[0], end_pt[0], num)

    start = time.time()
    fit = minimize(fun, y0,  method="BFGS") #, options={'disp':True})
    end = time.time()
    all_wall_times.append(end-start)
    print("-------------")
    print(num)
    # print(fit)
    # print(fun(fit.x))
    # print(fun(fit.x, g = 9.81))
    # all_x.append(fit.x)
    all_times.append(fun(fit.x, g = 9.81))

    # y0 = []
    # for i in range(len(fit.x)+1):
    #     if i == 0:
    #         y0.append((start_pt[1]+fit.x[i])/2.0)
    #         y0.append(fit.x[i])
    #         continue
    #     if i == len(fit.x):
    #         y0.append((fit.x[i-1]+end_pt[1])/2.0)
    #         y0.append((fit.x[i-1]+end_pt[1])/4.0)
    #         # y0.append()
    #     else:
    #         y0.append((fit.x[i-1]+fit.x[i])/2.0)
    #         y0.append(fit.x[i])

    all_y = np.zeros(num)
    all_y[0] = start_pt[1]
    all_y[1:-1] = fit.x
    all_y[-1] = end_pt[1]

    ax.scatter(all_x, all_y)
    ax.plot(all_x, all_y, label=num)

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

fig = plt.figure()
ax = fig.add_subplot(211)
ax.set_title("Brachistochrone Dimensionality")
ax.plot(num_pts, all_wall_times)
ax.set_ylabel("Wall Time (sec)")
ax = fig.add_subplot(212)
ax.plot(num_pts, all_times)
ax.set_xlabel('Number of Points')
ax.set_ylabel("Travel Time (sec)")

plt.show()
