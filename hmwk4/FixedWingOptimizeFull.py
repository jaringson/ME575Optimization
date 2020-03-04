# from Model import Model
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from copy import deepcopy
# import aerosonde_parameters as MAV
# import vaporlite_parameters as MAV

from tools import normalize
from tools import Quaternion2Euler
import pdb
import warnings

def R(q):
    wx = q[0]*q[1]
    wy = q[0]*q[2]
    wz = q[0]*q[3]
    xx = q[1]*q[1]
    xy = q[1]*q[2]
    xz = q[1]*q[3]
    yy = q[2]*q[2]
    yz = q[2]*q[3]
    zz = q[3]*q[3]
    out = np.array([ [ 1.0 - 2.0*yy - 2.0*zz, 2.0*xy + 2.0*wz,      2.0*xz - 2.0*wy],
           [2.0*xy - 2.0*wz,      1.0 - 2.0*xx - 2.0*zz,    2.0*yz + 2.0*wx],
           [2.0*xz + 2.0*wy,      2.0*yz - 2.0*wx,      1.0 - 2.0*xx - 2.0*yy]])
    return out;

class FixedWing():

    def __init__(self):
        self.numStates = 13
        self.numInputs = 4

        self._state = np.array([[0],  # (0)
                               [0],   # (1)
                               [0],   # (2)
                               [0],    # (3)
                               [0],    # (4)
                               [0],    # (5)
                               [0],    # (6)
                               [3],    # (7)
                               [0],    # (8)
                               [0],    # (9)
                               [0],    # (10)
                               [0],    # (11)
                               [0]])   # (12)

        self.state_max = np.array([[5],  # (0)
                               [5],   # (1)
                               [5],   # (2)

                               [1],    # (3)
                               [1],    # (4)
                               [1],    # (5)
                               [1],    # (6)

                               [6],    # (7)
                               [1],    # (8)
                               [1],    # (9)

                               [np.pi/1],    # (10)
                               [np.pi/1],    # (11)
                               [np.pi/1]])   # (12)

        self.uMax = 1.0
        self.uTMax = 1.0
        # self.uMax = 0.9
        # self.uTMax = 0.9
        self.uTMin = 0
        self._Va = 3
        self._alpha = 0
        self._beta = 0

        self.plotObjects = []

        # plane in ned
        self.plane = np.array([[0,0,0],
                  [0.5,0,0],
                  [0.1,0,0],
                  [0,0.5,-0.1], #left wing
                  [0.1,0,0],
                  [0,-0.5,-0.1], #right wing
                  [0.1,0,0],
                  [-0.5,0,0],
                  [-0.5,0,-0.25],
                  [-0.5,0.1,-0.25],
                  [-0.5,-0.1,-0.25]]).T

        self.create_coeficients(np.zeros((32,1)))


    def create_coeficients(self, coefList):
        # print("Coef List", coefList)

        ######################################################################################
                        #   Physical Parameters
        ######################################################################################
        self.mass = 0.015 #kg
        Jx = 0.00002 #kg m^2
        self.Jy = 0.0001
        Jz = 0.00011
        Jxz = 0.00001
        self.S_wing = 0.043
        self.b = 0.3429
        self.c = 0.1397
        self.S_prop = 0.1143**2*np.pi/4.0
        self.rho = 1.2682
        self.e = 1
        self.AR = (self.b**2) / self.S_wing
        self.gravity = 9.8

        ######################################################################################
                        #   Longitudinal Coefficients
        ######################################################################################
        self.C_L_0 = coefList[0] #0.1762
        self.C_L_alpha = coefList[1] #3.4329
        self.C_L_q = coefList[2] #4.8975
        self.C_L_delta_e = coefList[3] #0.000026274

        C_D_0 = coefList[4] #0.07
        C_D_alpha = coefList[5] #0.04
        self.C_D_q = coefList[6] #0.0
        self.C_D_p = coefList[7] #0.0
        self.C_D_delta_e = coefList[8] #0.0

        self.C_m_0 = coefList[9] #0.0
        self.C_m_alpha = coefList[10] #-0.26771
        self.C_m_q = coefList[11] #-2.3065
        self.C_m_delta_e = coefList[12] #-0.71205

        M = 50.0
        self.alpha0 = 0.47
        epsilon = 0.16


        ######################################################################################
                        #   Lateral Coefficients
        ######################################################################################
        self.C_Y_0 = coefList[13] #0.0
        self.C_Y_beta = coefList[14] #-0.19942
        self.C_Y_p = coefList[15] #-0.193
        self.C_Y_r = coefList[16] #0.27923
        self.C_Y_delta_a = coefList[17] #0.0
        self.C_Y_delta_r = coefList[18] #0.182201

        self.C_ell_0 = coefList[19] #0.0
        self.C_ell_beta = coefList[20] #-0.077793
        self.C_ell_p = coefList[21] #-0.27114
        self.C_ell_r = coefList[22] #0.14918
        self.C_ell_delta_a = coefList[23] #0.0
        self.C_ell_delta_r = coefList[24] #0.016043

        self.C_n_0 = coefList[25] #0.0
        self.C_n_beta = coefList[26] #0.074453
        self.C_n_p = coefList[27] #-0.07788
        self.C_n_r = coefList[28] #-0.098776
        self.C_n_delta_a = coefList[29] #0.0
        self.C_n_delta_r = coefList[30] # -0.090527

        ######################################################################################
                        #   Propeller thrust / torque parameters (see addendum by McLain)
        ######################################################################################
        self.C_prop = 1.0
        self.S_prop = 0.1143**2*np.pi/4.0
        self.k_motor = coefList[31]#6
        self.kTp = 0.
        self.kOmega = 0.

        # self.C_prop = coefList[31] #1.0
        # self.S_prop = coefList[32] #0.1143**2*np.pi/4.0
        # self.k_motor = coefList[33]#6
        # self.kTp = coefList[34] #0.
        # self.kOmega = coefList[35] #0.


        ######################################################################################
                        #   Calculation Variables
        ######################################################################################
        #   gamma parameters pulled from page 36 (dynamics)
        gamma = Jx * Jz - (Jxz**2)
        self.gamma1 = (Jxz * (Jx - self.Jy + Jz)) / gamma
        self.gamma2 = (Jz * (Jz - self.Jy) + (Jxz**2)) / gamma
        self.gamma3 = Jz / gamma
        self.gamma4 = Jxz / gamma
        self.gamma5 = (Jz - Jx) / self.Jy
        self.gamma6 = Jxz / self.Jy
        self.gamma7 = ((Jx - self.Jy) * Jx + (Jxz**2)) / gamma
        self.gamma8 = Jx / gamma

        #   C values defines on pag 62
        self.C_p_0         = self.gamma3 * self.C_ell_0      + self.gamma4 * self.C_n_0
        self.C_p_beta      = self.gamma3 * self.C_ell_beta   + self.gamma4 * self.C_n_beta
        self.C_p_p         = self.gamma3 * self.C_ell_p      + self.gamma4 * self.C_n_p
        self.C_p_r         = self.gamma3 * self.C_ell_r      + self.gamma4 * self.C_n_r
        self.C_p_delta_a    = self.gamma3 * self.C_ell_delta_a + self.gamma4 * self.C_n_delta_a
        self.C_p_delta_r    = self.gamma3 * self.C_ell_delta_r + self.gamma4 * self.C_n_delta_r
        self.C_r_0         = self.gamma4 * self.C_ell_0      + self.gamma8 * self.C_n_0
        self.C_r_beta      = self.gamma4 * self.C_ell_beta   + self.gamma8 * self.C_n_beta
        self.C_r_p         = self.gamma4 * self.C_ell_p      + self.gamma8 * self.C_n_p
        self.C_r_r         = self.gamma4 * self.C_ell_r      + self.gamma8 * self.C_n_r
        self.C_r_delta_a    = self.gamma4 * self.C_ell_delta_a + self.gamma8 * self.C_n_delta_a
        self.C_r_delta_r    = self.gamma4 * self.C_ell_delta_r + self.gamma8 * self.C_n_delta_r


    def draw_plane_nwu(self, plane_in):
        R = np.array([[1,0,0],
                      [0,-1,0],
                      [0,0,-1]])
        p = R.dot(plane_in)
        return p[0,:], p[1,:], p[2,:]


    def forward_simulate_dt(self,x,u,coefList,dt=.01):
        self.create_coeficients(coefList)
        self._state = deepcopy(x)
        # u = deepcopy(u.clip(-self.uMax,self.uMax))
        u = deepcopy(u)
        # x = x.reshape([self.numStates,-1])
        # xdot = np.zeros(x.shape)
        forces_moments = self._forces_moments(u)
        # xdot = self._derivatives(x, forces_moments).reshape((-1,13)).T
        # xdot[6:10] = normalize(xdot[6:10])
        # xdot[1,:] = x[0,:]
        # x = x + xdot*dt

        self._state[self._state<-1e100]=0
        self._state[self._state>1e100]=0
        k1 = self._derivatives(self._state, forces_moments)
        k1[k1<-1e10]=0
        k1[k1>1e10]=0
        k2 = self._derivatives(self._state + dt/2.0*k1, forces_moments)
        k2[k2<-1e10]=0
        k2[k2>1e10]=0
        k3 = self._derivatives(self._state + dt/2.0*k2, forces_moments)
        k3[k3<-1e10]=0
        k3[k3>1e10]=0
        k4 = self._derivatives(self._state + dt*k3, forces_moments)
        k4[k4<-1e10]=0
        k4[k4>1e10]=0
        self._state += dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)
        # print(k1,k2,k3,k4)
        # print(dt)
        # pdb.set_trace()
        self._state[self._state<-1e10]=0
        self._state[self._state>1e10]=0

        # pdb.set_trace()
        self._state[3:7] = normalize(self._state[3:7])
        x = deepcopy(self._state)



        # print('u',u)
        # print('x',x)
        # print('xdot',xdot)

        # if wrapAngle==True:
        #     x[1,:] = (x[1,:] + np.pi) % (2*np.pi) - np.pi

        # pdb.set_trace()

        return x

    # def calc_discrete_A_B_w(self,x,u,dt=.01):
    #     x = deepcopy(x)
    #     u = deepcopy(u)
    #     x = x.reshape([self.numStates,-1])
    #     A = np.matrix([[-self.b/self.I, 0],
    #                    [1.0, 0]])
    #     B = np.matrix([[1.0/self.I],
    #                    [0.0]])
    #     w = np.matrix([self.m*self.g*np.sin(x[1,:])/self.I,
    #                    [0.0]])
    #
    #     [Ad,Bd] = self.discretize_A_and_B(A,B,dt)
    #     wd = w*dt
    #
    #     return Ad,Bd,wd

    def visualize(self,x,ax,color='red'):
        # CoM = [-0.5*np.sin(x[1]),0.5*np.cos(x[1])]
        # theta = x[1]
        #
        # x = [CoM[0] + self.l/2.0*np.sin(theta),CoM[0] - self.l/2.0*np.sin(theta)]
        # y = [CoM[1] - self.l/2.0*np.cos(theta),CoM[1] + self.l/2.0*np.cos(theta)]
        #
        # massX = CoM[0] - self.l/2.0*np.sin(theta)
        # massY = CoM[1] + self.l/2.0*np.cos(theta)

        for plot in self.plotObjects:
            plot[0].remove()
        self.plotObjects = []

        # # self.plotObjects.append(ax.scatter(x[0], x[1], -x[2], 'bo', c='blue'))
        # self.plotObjects.append(ax.plot(*self.draw_plane_nwu(self.plane), linewidth=2, color='red'))
        phi, theta, psi = Quaternion2Euler(x[3:7])

        Rphi = np.array([[1,0,0],
                  [0,np.cos(phi),np.sin(phi)],
                  [0,-np.sin(phi),np.cos(phi)]])
        Rtheta = np.array([[np.cos(theta),0,-np.sin(theta)],
                      [0,1,0],
                      [np.sin(theta),0,np.cos(theta)]])
        Rpsi = np.array([[np.cos(psi),np.sin(psi),0],
                      [-np.sin(psi),np.cos(psi),0],
                      [0,0,1]])

        T = np.array([x[0],-x[1],-x[2]])

        Rot = Rphi.dot(Rtheta).dot(Rpsi)
        # print(Rot)
        # print(np.squeeze(R(x[3:7])))
        # pdb.set_trace()

        # Rot = np.squeeze(R(x[3:7]))

        # plt.clf()
        xs, ys, zs = self.draw_plane_nwu(Rot.dot(2.5*self.plane)+T)

        # xs, ys, zs = Rot.dot(1.5*self.plane)+T
        self.plotObjects.append(ax.plot(xs, ys, zs, linewidth=2, color=color))

        # plt.draw()
        # plt.plot(x[0],x[1], 'bo')
        # # ax.scatter(x[0], x[1], x[2], 'bo')
        # # plt.scatter(massX,massY,50,'r')
        # plt.axis([-20,20,-20,20])
        ax.set_xlim3d([-6, 6])
        ax.set_ylim3d([-6, 6])
        ax.set_zlim3d([-10, 20])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.ion()
        plt.show()
        plt.pause(.0000001)

    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        pn = state[0]
        pe = state[1]
        pd = state[2]
        e0 = state[3]
        e1 = state[4]
        e2 = state[5]
        e3 = state[6]
        u = state[7]
        v = state[8]
        w = state[9]
        # state[6:10] = normalize(state[6:10])
        p = state[10]
        q = state[11]
        r = state[12]
        #   extract forces/moments
        fx = forces_moments[0]
        fy = forces_moments[1]
        fz = forces_moments[2]
        l = forces_moments[3]
        m = forces_moments[4]
        n = forces_moments[5]


        # with warnings.catch_warnings():
        #     warnings.filterwarnings('error')
        #     try:
        #     # position kinematics
        #     except Warning as e:
        #         pdb.set_trace()
        #         print(e)

        pn_dot = (e1**2+e0**2-e2**2-e3**2)*u + 2*(e1*e2-e3*e0)*v + 2*(e1*e3+e2*e0)*w
        pe_dot = 2*(e1*e2+e3*e0)*u + (e2**2+e0**2-e1**2-e3**2)*v + 2*(e2*e3-e1*e0)*w
        pd_dot = 2*(e1*e3-e2*e0)*u + 2*(e2*e3+e1*e0)*v + (e3**2+e0**2-e1**2-e2**2)*w

        # pn_dot = (e0**2+e1**2-e2**2-e3**2)*u + 2*(e1*e2+e3*e0)*v + 2*(e1*e3-e2*e0)*w
        # pe_dot = 2*(e1*e2-e3*e0)*u + (e0**2-e1**2+e2**2-e3**2)*v + 2*(e2*e3+e1*e0)*w
        # pd_dot = 2*(e1*e3+e2*e0)*u + 2*(e2*e3-e1*e0)*v + (e0**2-e1**2-e2**2+e3**2)*w

        # pdb.set_trace()

        # position dynamics
        mass = self.mass
        u_dot = (r*v-q*w)+fx/mass
        v_dot = (p*w-r*u)+fy/mass
        w_dot = (q*u-p*v)+fz/mass

        # rotational kinematics
        e0_dot = 0.5*(-p*e1-q*e2-r*e3)
        e1_dot = 0.5*(p*e0+r*e2-q*e3)
        e2_dot = 0.5*(q*e0-r*e1+p*e3)
        e3_dot = 0.5*(r*e0+q*e1-p*e2)

        # rotatonal dynamics
        p_dot = self.gamma1*p*q - self.gamma2*q*r + self.gamma3*l + self.gamma4*n
        q_dot = self.gamma5*p*r - self.gamma6*(p**2-r**2) + m/self.Jy
        r_dot = self.gamma7*p*q - self.gamma1*q*r + self.gamma4*l + self.gamma8*n

        # collect the derivative of the states
        x_dot = np.array([pn_dot, pe_dot, pd_dot, e0_dot, e1_dot, e2_dot, e3_dot,
            u_dot, v_dot, w_dot, p_dot, q_dot, r_dot])
        # pdb.set_trace()


        # print(x_dot)
        return x_dot

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_t, delta_r)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        # assert delta.shape == (4,1)
        da = delta[0]
        de = delta[1]
        dt = delta[2]
        dr = delta[3]

        e0 = self._state[3]
        e1 = self._state[4]
        e2 = self._state[5]
        e3 = self._state[6]
        u = self._state[7]
        v = self._state[8]
        w = self._state[9]
        p = self._state[10]
        q = self._state[11]
        r = self._state[12]

        self._Va = np.sqrt(u**2 + v**2 + w**2)
        self._alpha = np.arctan(1.0*w/u)
        self._beta = np.arcsin(1.0*v/self._Va)



        Fg = self.mass*self.gravity*np.array([2*(e1*e3-e2*e0),
                                            2*(e2*e3 + e1*e0),
                                            e3**2 + e0**2 - e1**2 - e2**2,
                                            ])

        # Fg = self.mass*self.gravity*np.array([2*(e1*e3 - e2*e0),
        #                                     2*(e2*e3 + e1*e0),
        #                                     e3**2 + e0**2 - e1**2 - e2**2,
        #                                     ])

        M_e = 25
        sig = lambda a: (1+np.exp(-M_e*(a-self.alpha0))+np.exp(M_e*(a+self.alpha0)))/((1+np.exp(-M_e*(a-self.alpha0)))*(1+np.exp(M_e*(a+self.alpha0))))
        cla = lambda a: (1-sig(a))*(self.C_L_0+self.C_L_alpha*a)+sig(a)*(2*np.sign(a)*np.sin(a)**2*np.cos(a))
        cda = lambda a: self.C_D_p + (self.C_L_0+self.C_L_alpha*a)**2/(np.pi*self.e*self.AR)

        cxa = lambda a: -(cda(a)) * np.cos(a) + (cla(a)) * np.sin(a)

        cxq = lambda a: -self.C_D_q * np.cos(a) +self.C_L_q * np.sin(a)

        cxde = lambda a: -self.C_D_delta_e * np.cos(a) + self.C_L_delta_e * np.sin(a)

        cza = lambda a: -(cda(a)) * np.sin(a) - (cla(a)) * np.cos(a)

        czq = lambda a: -self.C_D_q * np.sin(a) - self.C_L_q * np.cos(a)

        czde = lambda a: -self.C_D_delta_e * np.sin(a) - self.C_L_delta_e * np.cos(a)

        c = self.c/(2.0*self._Va)
        b = self.b/(2.0*self._Va)



        one = 0.5*self.rho*self._Va**2*self.S_wing
        # two = np.array([[1,0,0],[0,1,0],[0,0,1]])
        three = np.array([[cxa(self._alpha)+cxq(self._alpha)*c*q+cxde(self._alpha)*de],
            [self.C_Y_0+self.C_Y_beta*self._beta+self.C_Y_p*b*p+self.C_Y_r*b*r+self.C_Y_delta_a*da+self.C_Y_delta_r*dr],
            [cza(self._alpha)+czq(self._alpha)*c*q+czde(self._alpha)*de]])

        Fa = np.squeeze(three) * one
        # pdb.set_trace()
        Fa = Fa.reshape((3,-1))

        F = Fg + Fa
        #
        # print("Fa:",Fa)

        Fp = 0.5*self.rho*self.S_prop*self.C_prop*((self.k_motor*dt)**2-self._Va**2)

        # print("FP:", Fp)

        fx = F[0] + Fp
            # + 0.5*MAV.rho*self._Va**2*MAV.S_wing*(\
            #     +cxa(self._alpha)\
            #     + cxq(self._alpha)*c*q\
            #     + cxde(self._alpha)*de
            #     )

        fy = F[1]
        fz = F[2]

        #  Moment time!!!
        one = 0.5*self.rho*self._Va**2*self.S_wing
        two = np.array([\
            [self.b*(self.C_ell_0+self.C_ell_beta*self._beta+self.C_ell_p*b*p+self.C_ell_r*b*r+self.C_ell_delta_a*da+self.C_ell_delta_r*dr)],
            [self.c*(self.C_m_0+(self.C_m_alpha*self._alpha)+(self.C_m_q*c*q)+(self.C_m_delta_e*de))],
            [self.b*(self.C_n_0+(self.C_n_beta*self._beta)+(self.C_n_p*b*p)+(self.C_n_r*b*r)+(self.C_n_delta_a*da)+(self.C_n_delta_r*dr))]
            ])
        Ma = one * np.squeeze(two)
        # print("\nMa:", Ma)
        # pdb.set_trace()
        Ma = Ma.reshape((3,-1))

        size = Ma.shape[1]

        Mp = np.block([[np.ones(size)*-self.kTp*(self.kOmega*dt)**2],
                       [np.zeros(size)],
                       [np.zeros(size)]
                       ])

        M = Mp + Ma

        Mx = M[0]
        My = M[1]
        Mz = M[2]

        # self._forces[0] = fx
        # self._forces[1] = fy
        # self._forces[2] = fz
        # pdb.set_trace()
        # print(fx, fy, fz, Mx, My, Mz)

        return np.array([fx, fy, fz, Mx, My, Mz])
