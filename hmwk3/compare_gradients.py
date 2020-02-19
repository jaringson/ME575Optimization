import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt

from truss_FD import truss as FD
from truss_CS import truss as CS
from truss_AD import truss as AD
from truss_Adjoint import truss as Adjoint



A = np.ones(10) * 1

FD_m, FD_dmdA, FD_s, FD_dsdA = FD(A)
CS_m, CS_dmdA, CS_s, CS_dsdA = CS(A)
AD_m, AD_dmdA, AD_s, AD_dsdA = AD(A)
Adjoint_m, Adjoint_dmdA, Adjoint_s, Adjoint_dsdA = Adjoint(A)

Compare_dmdA = Adjoint_dmdA
Compare_dsdA = Adjoint_dsdA

print("Relative Error")
print("Input A: ", A)
print("FD: ")
print("\tMass: ", np.linalg.norm(FD_dmdA-Compare_dmdA))
print("\tStress: ", np.linalg.norm(FD_dsdA-Compare_dsdA))
print("CS: ")
print("\tMass: ", np.linalg.norm(CS_dmdA-Compare_dmdA))
print("\tStress: ", np.linalg.norm(CS_dsdA-Compare_dsdA))
print("AD: ")
print("\tMass: ", np.linalg.norm(AD_dmdA-Compare_dmdA))
print("\tStress: ", np.linalg.norm(AD_dsdA-Compare_dsdA))
print("Adjoint: ")
print("\tMass: ", np.linalg.norm(Adjoint_dmdA-Compare_dmdA))
print("\tStress: ", np.linalg.norm(Adjoint_dsdA-Compare_dsdA))
