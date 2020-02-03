import numpy as np
from copy import deepcopy
from IPython.core.debugger import set_trace


class Optimizer():
    def __init__(self, function, start_X):
        self.num_variables = len(start_X)
        self.function = function
        self.gradientk = np.zeros(self.num_variables)
        self.gradientk_1 = np.zeros(self.num_variables)

        self.perturb = 1e-8
        self.Vk = np.eye(self.num_variables)
        self.Vk_1 = self.Vk
        self.Xk = np.zeros_like(start_X)
        self.Xk_1 = start_X

        self.tau_converge = 1e-6
        self.mu1 = 1e-4
        self.mu2 = 0.99 #0.8

        self.alpha_max = 1.0
        self.phi0 = []
        self.phi0_prime = []

        self.pk = 0

        self.init = True

    def gradient_func(self, X, update=True):
        y = self.function(X)
        gradient = np.empty_like(self.gradientk_1)
        for i in range(self.num_variables):
            # Xp = deepcopy(X)
            # Xp[i] += self.perturb
            # yp = self.function(Xp)
            #
            # Xm = deepcopy(X)
            # Xm[i] -= self.perturb
            # ym = self.function(Xm)

            Xim = deepcopy(X.astype(complex))
            # set_trace()
            Xim[i] = np.complex(X[i],self.perturb)
            yim = self.function(Xim)

            # gradient[i] = (yp - y) / self.perturb
            # gradient[i] = (yp - ym) / (2.0 * self.perturb)
            gradient[i] = np.imag(yim)/self.perturb
        if update:
            self.gradientk = deepcopy(self.gradientk_1)
            self.gradientk_1 = gradient
        return gradient

    def inverse_hessian(self):
        sk = self.Xk_1 - self.Xk
        sk = np.atleast_2d(sk).T
        yk = self.gradientk_1 - self.gradientk
        yk = np.atleast_2d(yk).T
        if (sk == 0).any() or (yk == 0).any():
            self.Vk = deepcopy(self.Vk_1)
            self.Vk_1 = np.eye(self.num_variables)
            set_trace()
        else:
            t1 = (np.eye(self.num_variables) - sk@yk.T / (sk.T@yk))
            t2 = self.Vk @ (np.eye(self.num_variables) - yk@sk.T/(sk.T@yk))
            t3 = sk@sk.T / (sk.T@yk)
            self.Vk = deepcopy(self.Vk_1)
            self.Vk_1 = t1 @ t2 + t3
        # print(self.Vk_1)
        # set_trace()

    def analytical_min(self, alpha_1, alpha_2, phi_1, phi_2):
        phi_prime_1 = self.gradient_func(self.Xk_1+alpha_1*self.pk, update=False) @ self.pk
        t1 = 2.0*alpha_1*(phi_2-phi_1)
        t2 = phi_prime_1*(alpha_1**2-alpha_2**2)
        t3 = phi_2-phi_1
        t4 = phi_prime_1*(alpha_1-alpha_2)
        return (t1+t2) / (2.0*(t3+t4))

    def pinpoint(self, alpha_low, alpha_high):
        alpha_low_end = deepcopy(alpha_low)
        alpha_high_end = deepcopy(alpha_high)
        j = 0
        while True:
            # print("Pinpoint: ", j)
            # print(alpha_low, alpha_high)
            # set_trace()

            phi_low = self.function(self.Xk_1+alpha_low*self.pk)
            phi_high = self.function(self.Xk_1+alpha_high*self.pk)

            alpha_j = self.analytical_min(alpha_low, alpha_high, phi_low, phi_high)


            phi_j = self.function(self.Xk_1+alpha_j*self.pk)

            if phi_j > self.phi0 + self.mu1*alpha_j*self.phi0_prime or phi_j > phi_low:
                # set_trace()
                alpha_high = deepcopy(alpha_j)
            else:
                phi_prime_j = self.gradient_func(self.Xk_1+alpha_j*self.pk, update=False) @ self.pk

                if np.abs(phi_prime_j) <= -self.mu2*self.phi0_prime:
                    return alpha_j
                elif phi_prime_j*(alpha_high-alpha_low) >= 0:
                    alpha_high = deepcopy(alpha_low)

                alpha_low = deepcopy(alpha_j)

            j = j + 1
            if j > 5:
                # print(alpha_low, alpha_high)
                # set_trace()
                return self.simple_backtracking()
            #     # if alpha_j < 1e-4:
            #     #     return 1e-4
            #     print(alpha_j)
            #     set_trace()
            #     return alpha_j

    def bracketing(self):
        alpha_step = 0.1
        alpha_array = [0.0]
        alpha_i = alpha_step
        # alpha_array = np.linspace(alpha0,self.alpha_max,100)
        i = 1
        while True:
            # print("Bracketing: ", i)
            phi_i = self.function(self.Xk_1+alpha_i*self.pk)
            phi_i_1 = self.function(self.Xk_1+alpha_array[i-1]*self.pk)

            if phi_i > self.phi0 + self.mu1*alpha_i*self.phi0_prime or (phi_i > phi_i_1):
                try:
                    return self.pinpoint(alpha_array[i-1], alpha_i)
                except:
                    set_trace()


            phi_prime_i = self.gradient_func(self.Xk_1+alpha_i*self.pk, update=False) @ self.pk
            if np.abs(phi_prime_i) <= -self.mu2*self.phi0_prime:
                return alpha_i
            elif phi_prime_i >= 0:
                # set_trace()
                return self.pinpoint(alpha_i, alpha_array[i-1])
            else:
                alpha_array.append(deepcopy(alpha_i))
                alpha_i += alpha_step
                if alpha_i > self.alpha_max:
                    # set_trace()
                    return self.simple_backtracking()
            i = i + 1

    def simple_backtracking(self):
        alphak = deepcopy(self.alpha_max)
        if self.init:
            self.init = False
        else:
            alphak = self.alphak * (np.linalg.norm(self.gradientk) / np.linalg.norm(self.gradientk_1))**2
        phi_alpha = self.function(self.Xk_1+alphak*self.pk)
        while phi_alpha > self.phi0 + self.mu1 * alphak * self.phi0_prime:
            alphak *= 0.5
            phi_alpha = self.function(self.Xk_1+alphak*self.pk)

        return alphak


    def minimize(self):
        self.gradient_func(self.Xk_1)
        while np.linalg.norm(self.gradientk_1, np.inf) > self.tau_converge:

            # if self.init:
            #     # self.init = False
            #     self.pk = -deepcopy(self.gradientk_1)
            # else:
            self.inverse_hessian()
            self.pk = -self.Vk_1 @ self.gradientk_1



            # self.pk = -deepcopy(self.gradientk_1)
            self.pk /= np.linalg.norm(self.pk)

            self.phi0 = self.function(self.Xk_1)
            self.phi0_prime = self.gradientk_1.T @ self.pk


            self.alphak = self.simple_backtracking()
            # self.alphak = self.bracketing()
            # if(alphak > 0.9):
            #     print(self.pk)
            #     set_trace()

            self.Xk = deepcopy(self.Xk_1)
            self.Xk_1 = self.Xk + self.alphak * self.pk
            # set_trace()

            self.gradient_func(self.Xk_1)
            # print(self.Xk_1)
            # print(self.alphak)
            # # print(self.gradientk_1)
            print(np.linalg.norm(self.gradientk_1, np.inf))
