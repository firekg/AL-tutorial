import numpy as np


L = np.linalg.cholesky(k)
# `beta` is `alpha` in [GP for ML] book
# `t` is `y` in [GP for ML] book
beta = np.linalg.solve(L.transpose(), np.linalg.solve(L,t))
kstar = kernel(data, xstar, theta, wantderiv=False, measnoise=0)
f = np.dot(kstar.transpose(), beta)
v = np.linalg.solve(L, kstar)
V = kernel(xstar, xstar, theta, wantderiv=False, measnoise=0 - np.dot(v.transpose(), v))
