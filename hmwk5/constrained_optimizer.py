import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from IPython.core.debugger import set_trace
import time
from copy import deepcopy

import sys
sys.path.append('../hmwk4')

from FixedWing import *
# from FixedWingOptimizeFull import *

from tools import Euler2Quaternion
from tools import boxminus


sys.path.append('../NEMPC')

from NonlinearEMPC import NonlinearEMPC

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

    print(one+two+three)
    # set_trace()
    # print(fw._state)
    # print(u)
    print(cost)
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


def genetic_obj_function(x,u,xgoal,ugoal,final_timestep=False):
    Q = 0.1*np.diag([0,0,100, 1,0,0, 50,50,50, 0,0,0])
    Qx = np.zeros((fw.numStates-1,fw.numStates))

    cost = np.zeros(u.shape[1])


    one =  np.sum(Q[0:6,0:6].dot(np.square(x[0:6]-xgoal[0:6])), axis=0)
    two = np.sum(Q[6:9,6:9] @ np.square(boxminus(deepcopy(x[6:10]),deepcopy(xgoal[6:10]))), axis=0)
    three = np.sum(Q[9:,9:].dot(np.square(x[10:]-xgoal[10:])), axis=0)

    # for i in range(u.shape[0]):
    cost = one + two + three
    # set_trace()

    # two =  boxminus(deepcopy(x[6:10]),deepcopy(xgoal[6:10])).T @ Q[6:9,6:9] @ boxminus(deepcopy(x[6:10]),deepcopy(xgoal[6:10]))
    # three = (x[10:]-xgoal[10:]).T.dot(Q[9:,9:].dot(x[10:]-xgoal[10:]))
    # cost += one[0,0] + two[0,0] + three[0,0]

    # print(cost)
    return cost



horizon = 100
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

u0 = np.zeros(4*num_pts)


phi0 = 0.0 #np.pi/4 # roll angle
theta0 =  0.0  # pitch angle
psi0 = 0.0 #np.pi/2  # yaw angle

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
print(xgoal)

u0 = np.array([ 0.20433147, -1.94372848,  0.70895769, -0.4871615 ,  1.8798103 ,
       -0.58387123,  0.05376949, -0.2297951 , -1.46098918, -0.92195947,
        0.76607728, -1.43007503, -0.82581917, -0.34715923,  0.51921409,
        0.20791168, -1.20392724, -1.05418708,  0.38166481, -1.12907951,
        1.85122421, -1.64858145,  0.90911387,  0.50796873, -1.84218198,
        1.40862561,  0.54977426, -0.06817422,  0.50568081,  1.18168001,
        0.08837746, -1.64071432,  0.77057864,  0.52146495,  0.11003808,
       -1.99409457, -1.83051271, -1.5862883 ,  0.9714796 ,  0.77167708])

obj_func(u0, dt, fw, xgoal)

# start = time.time()
# while True:
#     out = minimize(constrained_obj_func,
#         u0,
#         args=(dt,fw,xgoal,mu,ub,lb),
#         method="Nelder-Mead",
#         options={'disp':True, 'maxfev': 4000, 'adaptive':True})
#     print(out)
#     # print(fw._state)
#
#     # set_trace()
#     print("Nelder-Mead: ", time.time()-start)
#     print('Error: ', np.linalg.norm(u0-out.x))
#     if np.linalg.norm(u0-out.x) < 1e-2:
#         break
#     u0 = deepcopy(out.x)
#     mu *= 10




# controller = NonlinearEMPC(fw.forward_simulate_dt,
#                            genetic_obj_function,
#                            fw.numStates,
#                            fw.numInputs,
#                            umin=[-2., -2., 0., -2.],
#                            umax=[2., 2., 1., 2.],
#                            horizon=100,
#                            numKnotPoints=10,
#                            numSims=50,
#                            numParents=5,
#                            numStrangers=5)
#
# start = time.time()
# u = controller.solve_for_next_u(deepcopy(fw._state),xgoal,ulast=np.array([0.,0.,0.,0.]),ugoal=np.array([0,0,0,0]))
# print(u)
# print("Genetic: ", time.time()-start)
