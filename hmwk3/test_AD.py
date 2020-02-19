import numpy, algopy
from IPython.core.debugger import set_trace

x = algopy.UTPM(numpy.ones((2,1)))
y = algopy.UTPM(2*numpy.ones((2,1)))


# right
z = algopy.zeros(2, dtype=x)
z[0] = x
z[1] = y

set_trace()
