import numpy as np
import time
import matplotlib.pyplot as plt

from IPython.core.debugger import set_trace

x = [-2.0000, -1.7895, -1.5789, -1.3684, -1.1579, -0.9474, -0.7368, -0.5263, -0.3158, -0.1053, 0.1053, 0.3158, 0.5263, 0.7368, 0.9474, 1.1579, 1.3684, 1.5789, 1.7895, 2.0000]
f = [7.7859, 5.9142, 5.3145, 5.4135, 1.9367, 2.1692, 0.9295, 1.8957, -0.4215, 0.8553, 1.7963, 3.0314, 4.4279, 4.1884, 4.0957, 6.5956, 8.2930, 13.9876, 13.5700, 17.7481]

fig = plt.figure()
plt.plot(x,f)

plt.show()


def error(x,f):
    cost = 0
    for i in range(len(x)):
        cost += x[i] - f[i]
    return cost
