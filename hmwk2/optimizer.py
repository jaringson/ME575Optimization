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
        self.Xk = start_X
        self.Xk_1 = start_X

        self.tau_converge = 1e-6
        self.mu1 = 1e-4
        self.mu2 = 0.9 #0.8

        self.alpha0 = 1.0

        self.init = True

    def gradient_func(self, X, update=True):
        y = self.function(X)
        for i in range(self.num_variables):
            Xp = deepcopy(X)
            Xp[i] += self.perturb
            yp = self.function(Xp)

            Xm = deepcopy(X)
            Xm[i] -= self.perturb
            ym = self.function(Xm)

            # Xim = deepcopy(X)
            # set_trace()
            # Xim[i] = np.complex(X[i],self.perturb)
            # yim = self.function(Xim)

            if update:
                self.gradientk[i] = deepcopy(self.gradientk_1[i])
                # self.gradientk_1[i] = (yp - y) / self.perturb
                self.gradientk_1[i] = (yp - ym) / (2 * self.perturb)
                # self.gradientk_1[i] = np.imag(yim)/self.perturb
        return self.gradientk_1

    def inverse_hessian(self):
        sk = self.Xk_1 - self.Xk
        yk = self.gradientk_1 - self.gradientk
        t1 = (np.eye(self.num_variables) - sk@yk.T / (sk.T@yk))
        t2 = self.Vk @ (np.eye(self.num_variables) - yk@sk.T/(sk.T@yk))
        t3 = sk@sk.T / (sk.T@yk)
        self.Vk = deepcopy(self.Vk_1)
        self.Vk_1 = t1 @ t2 + t3
        # print(self.Vk_1)
        # set_trace()

    def minimize(self):
        self.gradient_func(self.Xk_1)
        while np.linalg.norm(self.gradientk_1, np.inf) > self.tau_converge:
            pk = 0
            alphak = self.alpha0
            if self.init:
                self.init = False
                pk = -deepcopy(self.gradientk_1)
            else:
                self.inverse_hessian()
                pk = -self.Vk_1 @ self.gradientk_1


            # pk = -deepcopy(self.gradientk_1)
            pk /= np.linalg.norm(pk)

            phi0 = self.function(self.Xk_1)
            phi0_prime = self.gradientk_1.T @ pk
            phi_alpha = self.function(self.Xk_1+alphak*pk)
            # set_trace()
            # print(alphak)
            while phi_alpha > phi0 + self.mu1 * alphak * phi0_prime:
                alphak *= 0.5
                phi_alpha = self.function(self.Xk_1+alphak*pk)

            self.Xk = deepcopy(self.Xk_1)
            self.Xk_1 = self.Xk + alphak * pk
            print(self.Xk_1)
            # set_trace()

            self.gradient_func(self.Xk_1)
