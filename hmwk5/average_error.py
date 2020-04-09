import numpy as np

true = [1.951342700001178e-06,
    3.3984701319945737e-09,
    3.8443061689139e-08,
    9.27136011483163e-09,
    9.61654837215035e-08,
    3.4540291314071e-08]

finite = [0.000672304473533783,
    0.0009224790665583137,
    0.0010146070019833487,
    0.0010019998638573636,
    0.0010395307305557599,
    0.0010302674734721951]

nelder = [1.1509422658126704e-05,
    5.458262999457207e-06,
    4.4392206944672734e-05,
    1.0640780656233718e-05,
    1.7003304695157417e-05,
    7.160801932020965e-06]


print(np.average(true))
print(np.average(finite))
print(np.average(nelder))