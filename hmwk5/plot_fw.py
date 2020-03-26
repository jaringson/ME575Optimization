import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from IPython.core.debugger import set_trace
import time
from copy import deepcopy

import sys
sys.path.append('../hmwk4')

from FixedWing import *

''' Turning '''
u_pen = np.array([ 1.99968847, -1.99937726,  0.99963555, -1.99974936, -1.85635716,
       -1.23882894,  0.99955841, -2.00000374,  0.18832958,  1.06771522,
        0.99962614,  1.260342  , -0.53428481, -2.00000832,  0.9997045 ,
        0.02068783, -0.28880833, -0.6296831 ,  0.99978117, -0.42219638,
       -0.24789309, -1.16855602,  0.99984204, -0.32701288, -0.29349161,
       -1.04229027,  0.99988301, -0.54846785, -0.22825965, -1.09610842,
        0.99997923, -0.39563048, -0.32289351, -1.35541227,  0.99995413,
       -0.41654092, -0.21861447, -0.89845965,  0.99999614, -0.41061412])

u_sci = np.array([ 2.00000000e+00, -2.00000000e+00,  2.67703744e-13, -2.00000000e+00,
       -2.00000000e+00, -1.29893099e+00,  1.58182494e-14, -2.00000000e+00,
        2.69946409e-01,  1.12027000e+00,  1.14238260e-14,  1.49416937e+00,
       -5.45670571e-01, -2.00000000e+00,  9.15567965e-14, -1.91047052e-01,
       -2.77824821e-01, -5.74670373e-01,  2.62099680e-05, -3.81942442e-01,
       -2.24118554e-01, -1.12806707e+00,  6.68414528e-15, -3.21549673e-01,
       -2.72779323e-01, -1.00151239e+00,  2.32793200e-14, -5.69961328e-01,
       -1.92305398e-01, -1.04904120e+00,  1.33946667e-13, -3.97364044e-01,
       -3.13485551e-01, -1.37535528e+00,  0.00000000e+00, -4.24995448e-01,
       -1.72517377e-01, -8.43206082e-01,  0.00000000e+00, -4.29127996e-01])

u_nm = np.array([-1.84161018e-02, -1.99483692e+00,  6.96283120e-04, -2.00004038e+00,
        1.04990092e-01, -1.10016670e-02, -1.99809985e-05,  4.84036379e-03,
        7.03573982e-02, -3.13537917e-02,  4.57449377e-04, -1.51344132e+00,
       -1.31926810e+00, -7.11379348e-01,  4.36115563e-04, -3.79435243e-07,
       -7.93707950e-02, -1.76152612e+00, -1.54937212e-04, -2.65641895e-04,
       -1.68023875e-02,  2.13825796e-03,  2.33775354e-03,  7.44666259e-05,
       -4.36472981e-02,  1.89506029e-04,  3.72667683e-06,  2.62970079e-03,
        1.63659550e-02, -1.54920172e-04,  7.18803404e-04,  1.04782328e-02,
       -3.04816746e-04,  3.94181348e-03,  2.10738312e-03,  1.75953751e-04,
       -3.63322863e-03,  2.45470432e-04,  9.29109554e-04,  2.66587846e-04])

u_ga = np.array([ 1.09693564, -1.26508089,  0.94215434, -1.71752379, -1.36331596,
       -1.46012808,  0.73298087, -1.35807449, -1.37555382, -0.41979344,
        0.12543043,  1.98362114,  1.83587683,  0.20665375,  0.304645  ,
       -0.6594078 ,  0.62120132, -1.40607086,  0.90107703,  0.47599804,
       -0.28190247, -0.85979452,  0.06371539,  0.01600364,  1.1330222 ,
       -1.67450575,  0.02834489, -0.99027705, -1.58429152, -0.56823755,
        0.24060194,  0.54986005,  0.89949179, -0.33906867,  0.94550225,
       -1.90851775, -1.57097071, -0.36258255,  0.19593639,  1.40942505])

u_ga = np.array([ 0.20433147, -1.94372848,  0.70895769, -0.4871615 ,  1.8798103 ,
       -0.58387123,  0.05376949, -0.2297951 , -1.46098918, -0.92195947,
        0.76607728, -1.43007503, -0.82581917, -0.34715923,  0.51921409,
        0.20791168, -1.20392724, -1.05418708,  0.38166481, -1.12907951,
        1.85122421, -1.64858145,  0.90911387,  0.50796873, -1.84218198,
        1.40862561,  0.54977426, -0.06817422,  0.50568081,  1.18168001,
        0.08837746, -1.64071432,  0.77057864,  0.52146495,  0.11003808,
       -1.99409457, -1.83051271, -1.5862883 ,  0.9714796 ,  0.77167708])

''' Straight Ahead '''
# u_pen = np.array([ 1.87363681e-04, -2.00006705e+00,  1.00002660e+00, -2.33882312e-04,
#        -3.17333717e-04, -1.10245442e+00,  1.00003213e+00,  8.27633534e-05,
#         2.57947815e-04,  1.07423706e+00,  1.00002629e+00,  1.58974966e-03,
#        -3.14965417e-04, -1.70142671e+00,  1.00001652e+00, -3.92501427e-03,
#         4.49127749e-04,  3.24570761e-01,  1.00001007e+00,  5.22999769e-03,
#        -4.62517403e-04, -9.14842233e-01,  1.00000702e+00, -6.10806468e-03,
#        -3.50761633e-05, -1.11973168e-01,  1.00000570e+00,  7.45938596e-03,
#         7.29422455e-04, -2.06469285e-01,  9.99995137e-01, -7.54311938e-03,
#        -6.12713791e-04,  7.72797519e-02,  1.00000036e+00,  4.82927089e-03,
#         5.42388529e-04, -3.54129731e-01,  1.86100734e-02, -2.16677653e-04])
#
#
# u_sci = np.array([ 6.14669381e-04, -9.99999996e-01,  9.99999999e-01, -2.05823018e-03,
#        -6.13579804e-04, -1.00000000e+00,  9.99999999e-01,  5.24138728e-03,
#         1.11727898e-04, -1.00000000e+00,  9.99999999e-01, -5.59979887e-03,
#        -4.79853153e-04, -1.59228654e-01,  1.00000000e+00,  2.83511273e-03,
#         1.06615507e-03, -9.99999890e-01,  1.00000000e+00,  3.05383687e-04,
#        -1.35143616e-03, -1.03345335e-01,  1.00000000e+00, -3.77528113e-03,
#         1.27016587e-03, -7.67814052e-02,  1.00000000e+00,  8.51543590e-03,
#        -5.83279918e-04, -3.70966284e-01,  1.00000000e+00, -1.17545768e-02,
#         4.95564007e-05, -2.21164060e-02,  1.00000000e+00,  8.09421382e-03,
#        -2.88587635e-04, -2.93187969e-01,  7.67200137e-01,  2.50612842e-03])
#
#
# u_nm = np.array([-1.10197584e-02, -2.00000617e+00,  3.61728634e-03, -1.07676630e-02,
#        -1.53420396e-02, -1.06206822e+00,  2.78968964e-04,  3.60258140e-03,
#         3.36486849e-03,  7.10242506e-01,  1.42047174e-03, -9.45916260e-03,
#        -6.29877336e-04, -8.80320218e-01,  8.83550658e-06,  1.01188610e-02,
#         2.43927336e-02, -1.21964905e+00,  1.88133993e-03, -7.78886687e-03,
#         2.15318748e-03, -1.93155281e-06,  6.34238576e-03,  1.40103287e-02,
#         3.23181884e-05, -1.05143326e-01,  2.33111250e-03,  1.13151499e-03,
#        -7.72364184e-04, -6.90478719e-04,  1.32502621e-02, -1.93151558e-05,
#         6.04935512e-03, -7.91969605e-04,  6.16191302e-04,  5.64580853e-02,
#        -6.99735105e-05, -9.24710800e-04,  2.22675940e-05,  8.44726243e-03])
#
# u_ga = np.array([ 0.71510705, -1.45924637,  0.00903116,  0.42335229, -1.61526648,
#        -0.47244311,  0.37739723,  1.06130301,  1.68152659, -1.92555773,
#         0.65227272, -1.01546236, -0.47939183, -0.39403306,  0.05737855,
#         1.00794696,  1.58677878, -1.59349678,  0.54176559, -0.18696258,
#         1.08029018,  1.16713375,  0.38280761, -1.66703202, -1.59309395,
#        -0.91179755,  0.96378061, -0.49550966, -0.73333638, -1.70133151,
#         0.65212684,  0.14966897, -1.53863898,  1.06064198,  0.70816568,
#        -0.22275053,  1.478807  , -0.55494244,  0.81421542, -1.66361029])
#
# u_ga = np.array([-1.51545392e+00, -1.60051032e+00,  1.63218045e-01, -1.84104688e-01,
#         1.75571571e+00, -7.63354184e-01,  9.80480245e-01,  1.86311871e-01,
#        -1.87059652e+00, -6.16912731e-01,  6.08973910e-01, -1.36213714e+00,
#         1.93445765e+00, -4.27425619e-01,  6.77945488e-02,  9.07128597e-04,
#        -8.59268716e-02, -2.78713920e-01,  2.23971392e-01,  5.83886904e-01,
#        -6.37673833e-01,  3.47171742e-01,  8.81084704e-01, -5.88808982e-01,
#        -7.63841618e-02, -1.89133123e+00,  1.79047775e-01,  8.36874183e-01,
#        -1.35166542e+00,  1.50903005e+00,  2.21878219e-01, -5.67939327e-02,
#        -9.99474660e-01, -1.03806502e+00,  7.80621241e-01, -2.51662839e-01,
#         1.69721188e+00,  1.92443398e+00,  8.20535732e-01,  7.08750102e-01])


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


horizon = 100
num_pts = 10
dt = 0.01

fw_pen = FixedWing()
fw_sci = FixedWing()
fw_nm = FixedWing()
fw_ga = FixedWing()


all_x_pen, all_u_pen, all_time_pen = obj_func(u_pen, dt, fw_pen)
all_x_sci, all_u_sci, all_time_sci = obj_func(u_sci, dt, fw_sci)
all_x_nm, all_u_nm, all_time_nm = obj_func(u_nm, dt, fw_nm)
all_x_ga, all_u_ga, all_time_ga = obj_func(u_ga, dt, fw_ga)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(all_x[0,:], -all_x[1,:], -all_x[2,:])
# ax.set_xlabel("X (m)")
# ax.set_ylabel("Y (m)")
# ax.set_zlabel("Z (m)")
# ax.set_zlim3d(90,110)


fig = plt.figure()
gs = fig.add_gridspec(3,1)
ax1 = fig.add_subplot(gs[0:2, 0])
plt.tight_layout()

ax1.set_title("Fixed Wing Positions")
ax1.plot(all_x_pen[0,:], -all_x_pen[1,:], label="Penalty")
ax1.plot(all_x_sci[0,:], -all_x_sci[1,:], label="Scipy")
ax1.plot(all_x_nm[0,:], -all_x_nm[1,:], label="Nelder-Mead")
ax1.plot(all_x_ga[0,:], -all_x_ga[1,:], label="Genetic")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_ylim(-4,1)
ax1.legend()

ax2 = fig.add_subplot(gs[2, 0])
ax2.plot(all_time_pen, -all_x_pen[2,:])
ax2.plot(all_time_sci, -all_x_sci[2,:])
ax2.plot(all_time_nm, -all_x_nm[2,:])
ax2.plot(all_time_ga, -all_x_ga[2,:])
ax2.set_ylabel('Z (m)')
ax2.set_xlabel('Time (s)')
ax2.set_ylim(98,101)
plt.tight_layout()

# fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True)
# axes.set_title("Fixed Wing Positions")
# axes.plot(all_x[0,:], all_x[1,:])
# axes.set_xlabel("X (m)")
# axes.set_ylabel("Y (m)")

# fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
# axes[0].set_title("Fixed Wing Positions")
# axes[0].plot(all_time_pen, all_x_pen[0,:], label="Penalty")
# axes[0].plot(all_time_sci, all_x_sci[0,:], label="Scipy")
# axes[0].plot(all_time_nm, all_x_nm[0,:], label="Nelder-Mead")
# axes[0].plot(all_time_ga, all_x_ga[0,:], label="Genetic")
# axes[0].set_ylabel("X (m)")
# axes[0].legend()
# axes[1].plot(all_time_pen, -all_x_pen[1,:])
# axes[1].plot(all_time_sci, -all_x_sci[1,:])
# axes[1].plot(all_time_nm, -all_x_nm[1,:])
# axes[1].plot(all_time_ga, -all_x_ga[1,:])
# axes[1].set_ylabel("Y (m)")
# axes[1].set_ylim(-10,10)
# axes[2].plot(all_time_pen, -all_x_pen[2,:])
# axes[2].plot(all_time_sci, -all_x_sci[2,:])
# axes[2].plot(all_time_nm, -all_x_nm[2,:])
# axes[2].plot(all_time_ga, -all_x_ga[2,:])
# axes[2].set_ylabel("Z (m)")
# axes[2].set_ylim(90,110)
# axes[2].set_xlabel("Time (s)")

# fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
# axes[0].set_title("Fixed Wing Velocities")
# axes[0].plot(all_time_pen, all_x_pen[3,:], label="Penalty")
# axes[0].plot(all_time_sci, all_x_sci[3,:], label="Scipy")
# axes[0].plot(all_time_nm, all_x_nm[3,:], label="Nelder-Mead")
# axes[0].plot(all_time_ga, all_x_ga[3,:], label="Genetic")
# axes[0].set_ylabel("X (m/s)")
# axes[0].legend()
# axes[1].plot(all_time_pen, -all_x_pen[4,:])
# axes[1].plot(all_time_sci, -all_x_sci[4,:])
# axes[1].plot(all_time_nm, -all_x_nm[4,:])
# axes[1].plot(all_time_ga, -all_x_ga[4,:])
# axes[1].set_ylabel("Y (m/s)")
# axes[2].plot(all_time_pen, -all_x_pen[5,:])
# axes[2].plot(all_time_sci, -all_x_sci[5,:])
# axes[2].plot(all_time_nm, -all_x_nm[5,:])
# axes[2].plot(all_time_ga, -all_x_ga[5,:])
# axes[2].set_ylabel("Z (m/s)")
# axes[2].set_xlabel("Time (s)")

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
plt.gcf().subplots_adjust(left=0.15)
axes[0].set_title("Fixed Wing Inputs")
axes[0].plot(all_time_pen, all_u_pen[0,:], label="Penalty")
axes[0].plot(all_time_sci, all_u_sci[0,:], label="Scipy")
axes[0].plot(all_time_nm, all_u_nm[0,:], label="Nelder-Mead")
axes[0].plot(all_time_ga, all_u_ga[0,:], label="Genetic")
axes[0].set_ylabel("Aileron")
axes[0].legend()
axes[1].plot(all_time_pen, all_u_pen[1,:])
axes[1].plot(all_time_sci, all_u_sci[1,:])
axes[1].plot(all_time_nm, all_u_nm[1,:])
axes[1].plot(all_time_ga, all_u_ga[1,:])
axes[1].set_ylabel("Elevator")
axes[2].plot(all_time_pen, all_u_pen[2,:])
axes[2].plot(all_time_sci, all_u_sci[2,:])
axes[2].plot(all_time_nm, all_u_nm[2,:])
axes[2].plot(all_time_ga, all_u_ga[2,:])
axes[2].set_ylabel("Throttle")
axes[3].plot(all_time_pen, all_u_pen[3,:])
axes[3].plot(all_time_sci, all_u_sci[3,:])
axes[3].plot(all_time_nm, all_u_nm[3,:])
axes[3].plot(all_time_ga, all_u_ga[3,:])
axes[3].set_ylabel("Rudder")
axes[3].set_xlabel("Time (s)")

plt.show()
