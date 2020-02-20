import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt

from truss_FD import truss as FD
from truss_CS import truss as CS
from truss_AD import truss as AD
from truss_Adjoint import truss as Adjoint

num_rand = 100

FD_m_error = 0
FD_s_error = 0
CS_m_error = 0
CS_s_error = 0
AD_m_error = 0
AD_s_error = 0
Adjoint_m_error = 0
Adjoint_s_error = 0

all_As = np.random.uniform(0.1,10,(num_rand,10))


for A in all_As:
    # set_trace()

    FD_m, FD_dmdA, FD_s, FD_dsdA = FD(A)
    CS_m, CS_dmdA, CS_s, CS_dsdA = CS(A)
    AD_m, AD_dmdA, AD_s, AD_dsdA = AD(A)
    Adjoint_m, Adjoint_dmdA, Adjoint_s, Adjoint_dsdA = Adjoint(A)

    Compare_dmdA = Adjoint_dmdA
    Compare_dsdA = Adjoint_dsdA

    FD_m_error += np.linalg.norm(FD_dmdA-Compare_dmdA)
    FD_s_error += np.linalg.norm(FD_dsdA-Compare_dsdA)

    CS_s_error += np.linalg.norm(CS_dmdA-Compare_dmdA)
    CS_s_error += np.linalg.norm(CS_dsdA-Compare_dsdA)

    AD_s_error += np.linalg.norm(AD_dmdA-Compare_dmdA)
    AD_s_error += np.linalg.norm(AD_dsdA-Compare_dsdA)

    Adjoint_s_error += np.linalg.norm(Adjoint_dmdA-Compare_dmdA)
    Adjoint_s_error += np.linalg.norm(Adjoint_dsdA-Compare_dsdA)

print("Average Relative Error")
# print("Input A: ", A)
print("FD: ")
print("\tMass: ", 1.0*FD_m_error/num_rand)
print("\tStress: ", 1.0*FD_s_error/num_rand)
print("CS: ")
print("\tMass: ", 1.0*CS_m_error/num_rand)
print("\tStress: ", 1.0*CS_s_error/num_rand)
print("AD: ")
print("\tMass: ", 1.0*AD_m_error/num_rand)
print("\tStress: ", 1.0*AD_s_error/num_rand)
print("Adjoint: ")
print("\tMass: ", 1.0*Adjoint_m_error/num_rand)
print("\tStress: ", 1.0*Adjoint_s_error/num_rand)
