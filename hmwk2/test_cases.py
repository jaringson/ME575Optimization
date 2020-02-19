import numpy as np

def matyas(x):
    f = 0.26*(x[0]**2+x[1]**2)-0.48*x[0]*x[1]
    g = np.zeros_like(x)
    g[0] = 0.52*x[0]-0.48*x[1]
    g[1] = 0.52*x[1]-0.48*x[0]
    return f, g

def rosenbrock(x):
    f = (1.0-x[0])**2+100.0*(x[1]-x[0]**2)**2
    g = np.zeros_like(x)
    g[0] = -400.0*x[0]*(x[1]-x[0]**2)-2.0*(1.0-x[0])
    g[1] = 200.0*(x[1]-x[0]**2)
    return f, g

def brachistochrone(yint):
    # global fcalls #This is for Dr. Ning's purposes I believe
    # if fcalls > 1e4:
    #     return

    mu_k = 0.3

    y = np.concatenate(([1], yint, [0]), axis=0) #check this is the same
    n = y.size
    x = np.linspace(0.0, 1.0, n)
    g = np.zeros(n-2)

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

    return f, g

if __name__=="__main__":
    yint = np.linspace(1, 0, 60)
    yint = yint[1:-1]

    f, g = brachistochrone(yint)
    print('f: ', f)
    print('g: ', g)
