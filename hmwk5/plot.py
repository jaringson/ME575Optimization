import numpy as np
from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import gridspec

n = [2,4,8,16,32,64]
# gf = [96, 270, 1592, 2070, 9872, 27192]
gf = [146, 504, 1910, 10637, 126959, 1622668]
gb_fd = [75, 750, 1485, 3247, 7326, 32825]
gb_exact = [25, 46, 70, 113, 225, 421]

# fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True)
fig = plt.figure()
plt.gcf().subplots_adjust(left=0.15)
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 0])
ax1 = plt.subplot(gs[0])
ax1.plot(n,gf,label='Nelder-Mead')
ax1.plot(n,gb_fd,label='Finite Difference')
ax1.plot(n,gb_exact,label='Exact Gradient')
plt.yscale('log')
ax1.legend()

# ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

ax1.set_title('Rosenbrock Dimensionality')
ax1.set_ylabel('Function Calls')

# axins = zoomed_inset_axes(ax1,10.0,loc=1) # zoom=6
# axins.plot(n,gf,label='Nelder-Mead')
# axins.plot(n,gb_fd,label='Finite Difference')
# axins.plot(n,gb_exact,label='Exact Gradient')
# x1,x2,y1,y2 = 0,70,0,3000
# axins.set_xlim(x1,x2)
# axins.set_ylim(y1,y2)

# mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")

# ax2 = plt.axes([.65, .6, .2, .2], facecolor='w')
# ax2 = plt.subplot(gs[1])
# ax2.plot(n,gf,label='Nelder-Mead')
# ax2.plot(n,gb_fd,label='Finite Difference')
# ax2.plot(n,gb_exact,label='Exact Gradient')
# plt.yscale('log')
# x1,x2,y1,y2 = -1,67,-1000,40000
# ax2.set_xlim(x1,x2)
# ax2.set_ylim(y1,y2)
ax1.set_xlabel('Number of Variables (n)')
# ax2.set_ylabel('Function Calls')
# plt.setp(ax2, xticks=[], yticks=[])

# ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))


# plt.subplots_adjust(bottom=0.1, top=1.0)

plt.show()
