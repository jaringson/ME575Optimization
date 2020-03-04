import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from IPython.core.debugger import set_trace
import time
from copy import deepcopy

from FixedWing import *


u = np.array([-6.26582972e-16, -2.20643104e-16,  5.00000000e-01,  1.13548318e-15,
        2.00133390e+00,  2.00109970e+00,  9.65912516e-01, -2.05127475e+00,
        2.00071652e+00,  2.00048092e+00,  1.00608311e+00, -2.04085816e+00,
        1.50219630e+00,  1.87019685e+00,  9.78954594e-01, -2.03146546e+00,
       -9.31421230e-02, -2.56793778e-01,  1.00037087e+00, -2.02293361e+00,
       -9.00224637e-01, -1.12652035e+00,  9.80526149e-01, -2.01684386e+00,
       -1.06882553e+00, -1.15833237e+00,  9.90724185e-01, -2.01101954e+00,
       -7.55526051e-01, -7.82346831e-01,  9.98959707e-01, -2.00679216e+00,
       -1.61585830e-01, -3.19436736e-01,  1.00041207e+00, -2.00352845e+00,
        1.10035443e+00,  2.09429077e-01,  9.81309584e-01, -2.00105683e+00])

def obj_func(u, dt, fw):
    global num_pts, horizon
    cost = 0
    # u = u.reshape((4,2))
    fw._state = deepcopy(fw._start)

    Q = 0.1*np.diag([0,0,5, 5,1,1, 50,50,50, 0,0,0])
    Qx = np.zeros((fw.numStates-1,fw.numStates))

    spot = 0
    counter = 0
    u_now = u[4*spot: 4*spot+4]

    all_x = np.array([])
    all_u = np.array([])
    time = 0
    all_time = []

    for i in range(horizon):
        counter += 1
        if counter >= horizon // num_pts and i != horizon-1:
            spot += 1
            # set_trace()
            u_now = u[4*spot: 4*spot+4]
            counter = 0
        # print(4*i,4*i+4)
        # u_now = u[4*i:4*i+4]
        x = fw.forward_simulate_dt(fw._state, u_now, dt)
        if all_x.size == 0:
            all_x = deepcopy(x)
            all_u = deepcopy(np.atleast_2d(np.array(u_now))).T
        else:
            all_x = np.append(all_x,deepcopy(x),axis=1)
            all_u = np.append(all_u,deepcopy(np.atleast_2d(np.array(u_now))).T,axis=1)
        all_time.append(time)
        time += dt
    return all_x, all_u, all_time


horizon = 10
num_pts = 10
dt = 0.01

fw = FixedWing()

all_x, all_u, all_time = obj_func(u, dt, fw)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(all_x[0,:], all_x[1,:], all_x[2,:])

fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
axes[0].set_title("Fixed Wing Positions")
axes[0].plot(all_time, all_x[0,:])
axes[1].plot(all_time, all_x[1,:])
axes[2].plot(all_time, all_x[2,:])

fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
axes[0].set_title("Fixed Wing Velocities")
axes[0].plot(all_time, all_x[3,:])
axes[1].plot(all_time, all_x[4,:])
axes[2].plot(all_time, all_x[5,:])

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
axes[0].set_title("Fixed Wing Inputs")
axes[0].plot(all_time, all_u[0,:])
axes[1].plot(all_time, all_u[1,:])
axes[2].plot(all_time, all_u[2,:])
axes[3].plot(all_time, all_u[3,:])

plt.show()
