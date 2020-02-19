import numpy as np
import algopy

from IPython.core.debugger import set_trace

'''
Computes mass and stress for the 10-bar truss structure
Parameters:
A: array of length 10 w/ the cross sectional area of each bar
    (see image in hw writeup for number order if interested)
​
Outputs:
mass: float mass of the entire structure
stress: array of length 10 with corresponding stress in each bar
'''

def truss(A):
    x = algopy.UTPM.init_jacobian(A)
    mass, stress  = _truss(x)
    dmdA = algopy.UTPM.extract_jacobian(mass)
    dsdA = algopy.UTPM.extract_jacobian(stress)
    dsdA = np.squeeze(dsdA)
    return mass.data[0,0], dmdA, np.squeeze(stress.data[0,0]), dsdA
    # print('jacobian = ',dmdA)
    # print('jacobian = ',dsigdA)


def _truss(A):
    P = 1e5 # applied loads
    Ls = 360 # length of sides
    Ld = np.sqrt(360**2 * 2) # length of diagonals

    # start = algopy.zeors(10, dtype=)
    start = np.array([5, 3, 6, 4, 4, 2, 5, 6, 3, 4]) - 1
    finish = np.array([3, 1, 4, 2, 3, 1, 4, 3, 2, 1]) - 1
    phi = np.deg2rad(np.array([0, 0, 0, 0, 90, 90, -45, 45, -45, 45]))
    L = np.concatenate(([Ls] * 6, [Ld] * 4), axis=0)

    nbar = np.size(A)
    E = 1e7 * np.ones( nbar) # modulus of elasticity
    rho = 0.1 * np.ones(nbar) # material density

    Fx = algopy.zeros(6, dtype=float)
    Fy = np.array([0.0, -P, 0.0, -P, 0.0, 0.0])
    rigidx = np.array([0, 0, 0, 0, 1, 1])
    rigidy = np.array([0, 0, 0, 0, 1, 1])

    n = np.size(Fx) #number of nodes
    DOF = 2

    # compute mass
    mass = sum(rho * A * L)

    # assemble global matrices
    K = algopy.zeros((DOF * n, DOF * n), dtype=A)
    S = algopy.zeros((nbar, DOF * n), dtype=A)

    for i in range(nbar): #loop through each bar
        Ksub, Ssub = bar(E[i], A[i], L[i], phi[i])
        idx = node2idx([start[i], finish[i]], DOF)
        idxx, idxy = np.meshgrid(idx,idx)

        for j in range(4):
            S[i, int(idx[j])] = Ssub[j]
            for k in range(4):
                K[int(idxy[j,k]), (idxx[j,k])] += Ksub[j,k]
        # set_trace()

    #setup applied loads
    # F = algopy.zeros(n * DOF, dtype=A)
    F = np.zeros(n * DOF)

    for i in range(n):
        idx = node2idx([i], DOF)
        # F[int(idx[0])] = Fx[i]
        # F[int(idx[1])] = Fy[i]
        F[idx[0]] = Fx[i]
        F[idx[1]] = Fy[i]

    #setup boundary condition
    idxx = np.argwhere(rigidx).squeeze()
    idxy = np.argwhere(rigidy).squeeze()
    removex = node2idx(idxx.tolist(), DOF)
    tempx = np.reshape(removex, (2,-1), order='F')
    removey = node2idx(idxy.tolist() , DOF)
    tempy = np.reshape(removey, (2,-1), order='F')
    removex = tempx[0,:]
    removey = tempy[1,:]

    remove = np.concatenate((removex, removey), axis=0)

    # K = np.delete(K, remove, axis=0)
    # K = np.delete(K, remove, axis=1)
    # F = np.delete(F, remove)
    # S = np.delete(S, remove, axis=1)
    K = K[:8, :8]
    F = F[:8]
    S = S[:, :8]

    d = algopy.solve(K, F.reshape((8,1)))
    # d = F.solve(K)
    # d = K.solve(F)
    stress = algopy.dot(S, d)

    return mass, stress

'''
Compute the stiffness and stress matrix for one element
Parameters:
E: float: modulus of elasticity
A: float: cross sectional area
L: float: length of element
phi: float: orientation of element

Outputs:
K: 4 x 4 ndarray: stiffness matrix
S: 1x4 ndarray: stress matrix
'''
def bar(E, A, L, phi):
    c = np.cos(phi)
    s = np.sin(phi)

    k0 = np.array([[c**2, c*s], [c*s, s**2]])

    K = algopy.zeros((4,4), dtype=A)
    for i in range(2):
        for j in range(2):
            coef = 1
            if (i == 0 and j == 1) or (i == 1 and j == 0):
                coef = -1
            K[2*i,2*j] = coef*c**2
            K[2*i,2*j+1] = coef*c*s
            K[2*i+1,2*j] = coef*c*s
            K[2*i+1,2*j+1] = coef*s**2
    K = E * A / L * K

    S = algopy.zeros(4, dtype=A)
    S[0] = -c
    S[1] = -s
    S[2] = c
    S[3] = s
    S = E / L * S

    return K, S

'''
Computes the appropriate indices in the global matrix for
the corresponding node numbers. You pass in the number of the node
(either as a scalar or an array of locations), and the DOF per node
and it returns the corresponding indices in the global matrices
'''
def node2idx(node, DOF):
    idx = np.empty(0)

    for i in range(len(node)):
        start = DOF * (node[i]-1) + 2
        finish = DOF * node[i] + 1

        idx = np.hstack((idx, np.arange(start, finish+1)))

    return idx.astype(int)

if __name__=="__main__":

    temp = np.ones(10) * .1
    mass, dmdA, stress, dsdA = truss(temp)
    print(mass, dmdA, stress, dsdA)
