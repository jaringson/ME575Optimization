import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from IPython.core.debugger import set_trace
import time
from copy import deepcopy

from FixedWing import *
# from FixedWingOptimizeFull import *

from tools import Euler2Quaternion
from tools import boxminus

def obj_func(u, dt, fw, xgoal):
    global num_pts, horizon
    cost = 0
    # u = u.reshape((4,2))
    fw._state = deepcopy(fw._start)

    Q = 0.1*np.diag([0,0,100, 1,0.1,0.1, 50,50,50, 0,0,0])
    Qx = np.zeros((fw.numStates-1,fw.numStates))

    spot = 0
    counter = 0
    u_now = u[4*spot: 4*spot+4]

    for i in range(horizon):
        counter += 1
        if counter >= horizon // num_pts and i != horizon-1:
            spot += 1
            # print(spot,i)
            # set_trace()
            u_now = u[4*spot: 4*spot+4]
            counter = 0
        # print(4*i,4*i+4)
        # u_now = u[4*i:4*i+4]
        x = fw.forward_simulate_dt(fw._state, u_now, dt)
        # Qx[0:6] = np.abs(Q[0:6,0:6].dot(x[0:6]-xgoal[0:6]))
        # Qx[6:9] = np.abs(Q[6:9,6:9].dot(boxminus(x[6:10],xgoal[6:10])))
        # Qx[9:] = np.abs(Q[9:,9:].dot(x[10:]-xgoal[10:]))

        one = (x[0:6]-xgoal[0:6]).T.dot(Q[0:6,0:6].dot(x[0:6]-xgoal[0:6]))
        two =  boxminus(deepcopy(x[6:10]),deepcopy(xgoal[6:10])).T @ Q[6:9,6:9] @ boxminus(deepcopy(x[6:10]),deepcopy(xgoal[6:10]))
        three = (x[10:]-xgoal[10:]).T.dot(Q[9:,9:].dot(x[10:]-xgoal[10:]))
        # print(one,two,three)
        cost += one[0,0] + two[0,0] + three[0,0]

    # print(fw._state)
    # print(u)
    return cost

def constrained_obj_func(u, dt, fw, xgoal, mu, ub, lb):
    fx_cost = obj_func(u, dt, fw, xgoal)


    upper = u - ub # <= 0
    lower = lb - u # <= 0

    upper = upper[upper>0]**2
    lower = lower[lower>0]**2
    cost = fx_cost + mu/2 * (np.sum(upper) + np.sum(lower))
    print(cost)
    # set_trace()

    return cost


horizon = 10
num_pts = 10
dt = 0.01

fw = FixedWing()

u0 = np.zeros((num_pts,4))
u0[:,2] = 0.5
u0 = u0.flatten()


# bounds = [(-1,1),(-1,1),(0,1),(-1,1)]*num_pts
ub = 2*np.ones((num_pts,4))
ub[:,2] = 1
ub = ub.flatten()

lb = -2*np.ones((num_pts,4))
lb[:,2] = 0
lb = lb.flatten()

mu = 10

# u0 = np.zeros(4)


phi0 = 0.  # roll angle
theta0 =  0.  # pitch angle
psi0 = np.pi/2  # yaw angle

e = Euler2Quaternion(phi0, theta0, psi0, 1)
e0 = e.item(0)
e1 = e.item(1)
e2 = e.item(2)
e3 = e.item(3)

xgoal = np.array([[0],  # (0)
                   [0],   # (1)
                   [-100],   # (2)
                   [15],    # (3)
                   [0],    # (4)
                   [0],    # (5)
                   [e0],    # (6)
                   [e1],    # (7)
                   [e2],    # (8)
                   [e3],    # (9)
                   [0],    # (10)
                   [0],    # (11)
                   [0]])   # (12)



out = minimize(constrained_obj_func,
    u0,
    args=(dt,fw,xgoal,mu,ub,lb),
    method="SLSQP",
    options={'disp':True})
print(out)
print(fw._state)
