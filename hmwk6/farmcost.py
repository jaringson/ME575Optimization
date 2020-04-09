import numpy as np
import matplotlib.pyplot as plt

from IPython.core.debugger import set_trace

def jensenwake(x,y,alpha,beta,delta):
    if x < 0: return 1.0

    theta = np.arctan(y/(x+delta))
    f = 0.0
    if np.abs(theta) < np.pi/beta:
        f = 0.5*(1+np.cos(beta*theta))

    # normalized by Uinf
    V = 1 - 2/3*(0.5/(0.5+alpha*x))**2 * f
    return V

def farmcost(yback, alpha, beta, delta, plotit=False):
    x = np.array([0.0, 4.0, 8.0])
    y = np.hstack([0, yback]) # append 0 in front for lead turbine
    CP = 0.38
    rho = 1.2

    u12 = jensenwake(x[1]-x[0], y[1]-y[0], alpha, beta, delta)
    u13 = jensenwake(x[2]-x[0], y[2]-y[0], alpha, beta, delta)
    u23 = jensenwake(x[2]-x[1], y[2]-y[1], alpha, beta, delta)

    u = np.array([1.0, u12, 1-np.sqrt((1-u13)**2 + (1-u23)**2)])
    P = CP*0.5*rho*u**3*np.pi*1.0**2/4

    area = np.mean((x - np.mean(x))**2 + (y - np.mean(y))**2)
    othercost = 1.0

    power = np.sum(P)

    coe = (area/100 + othercost)/power

    if plotit:
        nx = 200
        ny = 250

        xvec = np.linspace(-2,12,nx)
        yvec = np.linspace(-5,10,ny)

        Y,X = np.meshgrid(yvec, xvec)
        U1 = np.zeros([nx,ny])
        U2 = np.zeros([nx,ny])
        U3 = np.zeros([nx,ny])

        for i in range(nx):
            for j in range(ny):
                U1[i,j] = jensenwake(X[i,j]-x[0], Y[i,j]-y[0], alpha,beta,delta)
                U2[i,j] = jensenwake(X[i,j]-x[1], Y[i,j]-y[1], alpha,beta,delta)
                U3[i,j] = jensenwake(X[i,j]-x[2], Y[i,j]-y[2], alpha,beta,delta)

        Udef = 1 - np.sqrt((1-U1)**2 + (1-U2)**2 + (1-U3)**2)

        plt.figure()
        plt.contourf(X, Y, Udef, 200)
        plt.colorbar()
        plt.xlabel('x/D')
        plt.ylabel('y/D')

        plt.show()

    return coe

if __name__ == '__main__':
    alpha = 0.1
    beta = 9
    delta = 5

    yback = np.array([2., 5.])

    print(farmcost(yback, alpha, beta, delta, plotit=False))
