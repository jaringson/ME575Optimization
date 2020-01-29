import numpy as np
from copy import deepcopy

class Optimizer():
    def __init__(self, funtion, start_X):
        self.num_variables = len(start_X)
        self.function = function
        self.gradient = np.zeros(self.num_variables)

        self.perturb = 0.01
        self.Vk = self.gradient(start_X)
        self.Xk = start_X
        # self.Xk_1 = ??

    def gradient(self, X):
        cost_1 = self.function(X)
        for i in self.num_variables:
            temp_X = deepcopy(X)
            temp_X[i] += self.perturb

            cost_2 = self.function(temp_X)
            self.gradient[i] = (cost_1 - cost_2) / self.perturb

    def hessian(self, Xk_1):
        sk = 9
        Vk_1 = (np.eye(self.num_variables) - 9)

    def minimize():
        
