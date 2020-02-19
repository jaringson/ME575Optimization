import numpy as np
import matplotlib.pyplot as plt

from IPython.core.debugger import set_trace

name = "Rosenbrock"
data = np.load('rosenbrock.npz', allow_pickle=True)
data = data['arr_0'].item()
outputs=data['outputs']
# set_trace()

labels = ['Steepest Backtrack', 'Quasi-Newton Backtrack', 'Steepest N-W', 'Quasi-Newton N-W']
colors = ['grey', 'green', 'red', 'orange']

fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
plt.gcf().subplots_adjust(left=0.15)

# plt.semilo
# plt.semilogy()
axes[0].set_title(name)
axes[0].set_ylabel('Gradient Norm\n(inf)')
axes[1].set_ylabel('Number of\nFunction Calls')
axes[2].set_ylabel('Function\nEvaulations')
axes[2].set_xlabel('Iterations')
for i in range(4):
    if i == 0 or i == 2:
        axes[0].semilogy(range(outputs[i]['iterations']+1), outputs[i]['list_norms'], color=colors[i], label=labels[i])
        axes[1].semilogy(range(outputs[i]['iterations']+1), outputs[i]['list_function_calls'], color=colors[i])
        if data['name'] == 'brachistochrone':
            axes[2].plot(range(outputs[i]['iterations']+1), outputs[i]['list_function_values'], color=colors[i])
        else:
            axes[2].semilogy(range(outputs[i]['iterations']+1), outputs[i]['list_function_values'], color=colors[i])

axes[0].legend()


fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
plt.gcf().subplots_adjust(left=0.15)

# plt.semilo
# plt.semilogy()
axes[0].set_title(name)
axes[0].set_ylabel('Gradient Norm\n(inf)')
axes[1].set_ylabel('Number of\nFunction Calls')
axes[2].set_ylabel('Function\nEvaulations')
axes[2].set_xlabel('Iterations')
for i in range(4):
    if i == 1 or i == 3:
        axes[0].semilogy(range(outputs[i]['iterations']+1), outputs[i]['list_norms'], color=colors[i], label=labels[i])
        axes[1].semilogy(range(outputs[i]['iterations']+1), outputs[i]['list_function_calls'], color=colors[i])
        if data['name'] == 'brachistochrone':
            axes[2].plot(range(outputs[i]['iterations']+1), outputs[i]['list_function_values'], color=colors[i])
        else:
            axes[2].semilogy(range(outputs[i]['iterations']+1), outputs[i]['list_function_values'], color=colors[i])


axes[0].legend()
# axes[1].plot(num_pts, all_wall_times)
# axes[1].scatter(num_pts, all_wall_times)
# axes[1].set_ylabel("Solve\nTime (s)")

plt.show()
