import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from optimizer import Optimizer

from IPython.core.debugger import set_trace


def func(x):
    return np.exp(x)/np.sqrt(np.sin(x)**3+np.cos(x)**3)

opt = Optimizer(func, np.array([0]))

x_all = np.linspace(-0.5,2,1000)
y = []
dx = []


np.set_printoptions(15)

opt.perturb = 0.1
print(opt.gradient_func(np.array([2.0])))

opt.perturb = 10**-6
print(opt.gradient_func(np.array([2.0])))

opt.perturb = 10**-15
print(opt.gradient_func(np.array([2.0])))

opt.perturb = 10**-30
print(opt.gradient_func(np.array([2.0])))

# for x in x_all:
#     # set_trace()
#     opt.gradient_func(np.array([x]))
#     y.append(func(np.array([x])))
#     dx.append(opt.gradientk_1[0])
#
# fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
# axes[0].plot(x_all,y)
# axes[1].plot(x_all,dx)
#
# plt.show()
