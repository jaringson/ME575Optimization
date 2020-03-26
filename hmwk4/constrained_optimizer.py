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

    Q = 0.1*np.diag([0,0,100, 1,0,0, 50,50,50, 0,0,0])
    Qx = np.zeros((fw.numStates-1,fw.numStates))

    spot = 0
    counter = 0
    u_now = deepcopy(u[4*spot: 4*spot+4])

    for i in range(horizon):
        counter += 1
        if counter >= horizon // num_pts and i != horizon-1:
            spot += 1
            # print(spot,i)
            # set_trace()
            u_now = deepcopy(u[4*spot: 4*spot+4])

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
        cost += one[0,0] + two[0,0] + three[0,0]

    # print(one,two,three)
    # set_trace()
    # print(fw._state)
    # print(u)
    # print(cost)
    # set_trace()
    return cost

def constrained_obj_func(u, dt, fw, xgoal, mu, ub, lb):
    fx_cost = obj_func(u, dt, fw, xgoal)


    upper = u - ub # <= 0
    lower = lb - u # <= 0

    upper = upper[upper>0]**2
    lower = lower[lower>0]**2
    cost = fx_cost + mu/2 * (np.sum(upper) + np.sum(lower))
    # print(cost)
    # set_trace()

    return cost


horizon = 100
num_pts = 10
dt = 0.01

fw = FixedWing()

u0 = np.zeros((num_pts,4))
u0[:,2] = 0.5
u0 = u0.flatten()


bounds = [(-2,2),(-2,2),(0,1),(-2,2)]*num_pts
ub = 2*np.ones((num_pts,4))
ub[:,2] = 1
ub = ub.flatten()

lb = -2*np.ones((num_pts,4))
lb[:,2] = 0
lb = lb.flatten()

mu = 10

# u0 = np.zeros(4*num_pts)

# u0 = np.array([ 1.99968847, -1.99937726,  0.99963555, -1.99974936, -1.85635716,
#        -1.23882894,  0.99955841, -2.00000374,  0.18832958,  1.06771522,
#         0.99962614,  1.260342  , -0.53428481, -2.00000832,  0.9997045 ,
#         0.02068783, -0.28880833, -0.6296831 ,  0.99978117, -0.42219638,
#        -0.24789309, -1.16855602,  0.99984204, -0.32701288, -0.29349161,
#        -1.04229027,  0.99988301, -0.54846785, -0.22825965, -1.09610842,
#         0.99997923, -0.39563048, -0.32289351, -1.35541227,  0.99995413,
#        -0.41654092, -0.21861447, -0.89845965,  0.99999614, -0.41061412])

phi0 = 0.0  # roll angle
theta0 =  0.  # pitch angle
psi0 = 0.0  # yaw angle

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
start = time.time()
out = minimize(obj_func,
    u0,
    args=(dt,fw,xgoal),
    method="SLSQP",
    bounds=bounds,
    options={'disp':True})
print(out)
# print(fw._state)
print("SCIPY: ", time.time()-start)

start = time.time()
while True:
    out = minimize(constrained_obj_func,
        u0,
        args=(dt,fw,xgoal,mu,ub,lb),
        method="SLSQP",
        options={'disp':True})
    # print(out)
    # print(fw._state)

    # set_trace()
    if np.linalg.norm(u0-out.x) < 1e-2:
        break
    u0 = deepcopy(out.x)
    mu *= 10
print("Penality: ", time.time()-start)
