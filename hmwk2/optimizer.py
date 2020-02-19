import numpy as np
from copy import deepcopy
from alg_types import *

from IPython.core.debugger import set_trace

'''Optimitation Class'''
'''Please provide a float (double) for the start_X'''
class Optimizer():
    def __init__(self, function, start_X, alg_type, dir_type):
        self.num_variables = len(start_X)
        self.function = function
        self.gradientk = np.zeros(self.num_variables)
        self.gradientk_1 = np.zeros(self.num_variables)

        self.perturb = 1e-8
        self.Vk_1 = np.eye(self.num_variables)
        self.Xk = np.zeros_like(start_X)
        self.Xk_1 = np.array(start_X)

        self.tau_converge = 1e-6
        self.mu1 = 1e-4
        self.mu2 = 0.8 #0.8

        self.alpha_max = 100.0
        self.alphak = 0.0
        self.phi0 = []
        self.phi0_prime = []

        self.pk = 0

        self.init = True

        self.function_calls = 0
        self.iterations = 0

        self.line_type = alg_type
        self.dir_type = dir_type

        self.list_funct_calls = []
        self.list_norm = []
        self.list_funct_values = []

    '''For Calculating Gradients (Not using in this assignment)'''
    # def gradient_func(self, X, update=True):
    #     y = self.function(X)
    #     gradient = np.empty_like(self.gradientk_1)
    #     for i in range(self.num_variables):
    #         # Xp = deepcopy(X)
    #         # Xp[i] += self.perturb
    #         # yp = self.function(Xp)
    #         #
    #         # Xm = deepcopy(X)
    #         # Xm[i] -= self.perturb
    #         # ym = self.function(Xm)
    #
    #         Xim = deepcopy(X.astype(complex))
    #         # set_trace()
    #         Xim[i] = np.complex(X[i],self.perturb)
    #         yim = self.function(Xim)
    #
    #         # gradient[i] = (yp - y) / self.perturb
    #         # gradient[i] = (yp - ym) / (2.0 * self.perturb)
    #         gradient[i] = np.imag(yim)/self.perturb
    #     if update:
    #         self.gradientk = deepcopy(self.gradientk_1)
    #         self.gradientk_1 = gradient
    #     return gradient

    '''Inverse Hessian Function'''
    def inverse_hessian(self):
        sk = self.Xk_1 - self.Xk
        sk = np.atleast_2d(sk).T
        yk = self.gradientk_1 - self.gradientk
        yk = np.atleast_2d(yk).T
        if (sk == 0).any() or (yk == 0).any():
            self.Vk_1 = np.eye(self.num_variables)
        else:
            t1 = (np.eye(self.num_variables) - sk@yk.T / (sk.T@yk))
            t2 = self.Vk_1 @ (np.eye(self.num_variables) - yk@sk.T/(sk.T@yk))
            t3 = sk@sk.T / (sk.T@yk)
            self.Vk_1 = t1 @ t2 + t3


    '''Approximate Quadratic Min'''
    def analytical_min(self, alpha_1, alpha_2, phi_1, phi_2, phi_prime_1):
        t1 = 2.0*alpha_1*(phi_2-phi_1)
        t2 = phi_prime_1*(alpha_1**2-alpha_2**2)
        t3 = phi_2-phi_1
        t4 = phi_prime_1*(alpha_1-alpha_2)
        return (t1+t2) / (2.0*(t3+t4))

    '''Algorithm #7: Pinpointing'''
    def pinpoint(self, alpha_low, alpha_high):
        alpha_low_end = deepcopy(alpha_low)
        alpha_high_end = deepcopy(alpha_high)
        j = 0
        while True:

            phi_low, phi_prime_low = self.function(self.Xk_1+alpha_low*self.pk)
            self.function_calls += 1
            phi_prime_low = phi_prime_low @ self.pk
            phi_high, _ = self.function(self.Xk_1+alpha_high*self.pk)
            self.function_calls += 1

            alpha_j = self.analytical_min(alpha_low, alpha_high, phi_low, phi_high, phi_prime_low)

            phi_j, phi_prime_j = self.function(self.Xk_1+alpha_j*self.pk)
            self.function_calls += 1
            phi_prime_j = phi_prime_j @ self.pk

            if phi_j > self.phi0 + self.mu1*alpha_j*self.phi0_prime or phi_j > phi_low:
                # set_trace()
                alpha_high = deepcopy(alpha_j)
            else:

                if np.abs(phi_prime_j) <= -self.mu2*self.phi0_prime:
                    return alpha_j
                elif phi_prime_j*(alpha_high-alpha_low) >= 0:
                    alpha_high = deepcopy(alpha_low)

                alpha_low = deepcopy(alpha_j)

            j = j + 1
            if j > 5:
                #### If we are in here too long, revert to Backtracking ###
                return self.simple_backtracking()

    '''Alogirhtm #6: Bracketing'''
    def bracketing(self):
        alphak = deepcopy(self.alpha_max)
        alpha_step = 1.0
        alpha_array = [0.0]
        alpha_i = alpha_step #np.max([alphak,1e-20])

        i = 1
        phi_i_1 = self.phi0
        while True:
            # print("Bracketing: ", i)
            phi_i, phi_prime_i = self.function(self.Xk_1+alpha_i*self.pk)
            self.function_calls += 1
            phi_prime_i = phi_prime_i @ self.pk

            if phi_i > self.phi0 + self.mu1*alpha_i*self.phi0_prime or (phi_i > phi_i_1):
                return self.pinpoint(alpha_array[i-1], alpha_i)


            if np.abs(phi_prime_i) <= -self.mu2*self.phi0_prime:
                return alpha_i
            elif phi_prime_i >= 0:
                return self.pinpoint(alpha_i, alpha_array[i-1])
            else:
                alpha_array.append(deepcopy(alpha_i))
                alpha_i += alpha_step
                phi_i_1 = deepcopy(phi_i)
                if alpha_i > self.alpha_max:
                    # set_trace()
                    return self.simple_backtracking()
            i = i + 1

    '''Simple backtracking algorithm'''
    def simple_backtracking(self):
        alphak = deepcopy(self.alpha_max)
        if self.init or (self.gradientk == 0.0).all():
            self.init = False
        else:
            if self.alphak > 1e-4:
                alphak = self.alphak * (np.linalg.norm(self.gradientk) / np.linalg.norm(self.gradientk_1))**2

        phi_alpha, gradient = self.function(self.Xk_1+alphak*self.pk)
        self.function_calls += 1
        while phi_alpha > self.phi0 + self.mu1 * alphak * self.phi0_prime:
            alphak *= 0.5
            phi_alpha, gradient = self.function(self.Xk_1+alphak*self.pk)
            self.function_calls += 1

            #### Make sure that alpha doesn't get too small ####
            if alphak < 1e-8:
                break

        return alphak

    '''Minimize function gets called to start the optimization routine'''
    def minimize(self):
        self.phi0, self.gradientk_1 = self.function(self.Xk_1)
        self.function_calls += 1

        self.list_norm.append(np.linalg.norm(self.gradientk_1, np.inf))
        self.list_funct_calls.append(self.function_calls)
        self.list_funct_values.append(self.phi0)

        #### Convergence criteria is based on the infinity norm ####
        while np.linalg.norm(self.gradientk_1, np.inf) > self.tau_converge:
            self.iterations += 1


            #### Get a driction ###
            if self.dir_type==DirType.STEEPEST:
                self.pk = -deepcopy(self.gradientk_1)
            elif self.dir_type==DirType.QUASI:
                self.inverse_hessian()
                self.pk = -self.Vk_1 @ self.gradientk_1
            else:
                raise Exception("Invalid Direction Option")

            self.pk /= np.linalg.norm(self.pk)


            self.phi0_prime = self.gradientk_1.T @ self.pk
            # set_trace()


            if self.line_type == LineType.BACKTRACK:
                self.alphak = self.simple_backtracking()

            elif self.line_type == LineType.BRACKET:
                self.alphak = self.bracketing()
            else:
                raise Exception("Invalid Line Search Option")


            self.gradientk = deepcopy(self.gradientk_1)
            self.phi0, self.gradientk_1 = self.function(self.Xk_1+ self.alphak * self.pk)
            self.function_calls += 1
            self.Xk = deepcopy(self.Xk_1)
            self.Xk_1 = self.Xk + self.alphak * self.pk

            #### For debugging: Print out norm ###
            # print(np.linalg.norm(self.gradientk_1, np.inf))

            self.list_norm.append(np.linalg.norm(self.gradientk_1, np.inf))
            self.list_funct_calls.append(self.function_calls)
            self.list_funct_values.append(self.phi0)
