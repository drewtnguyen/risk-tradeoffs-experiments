import numpy as np
from numpy.random import standard_normal
from numpy.linalg import svd
from numba import njit

# Very unsafe. In particular, does not check whether covariance is PSD.
@njit
def multivariate_normal(mean, cov, size=None, check_valid='warn',
                        tol=1e-8):
    final_shape = (size, mean.shape[0])
    x = np.empty(final_shape,dtype=np.float64)
    N, M = x.shape
    for i in range(N):
        for j in range(M):
            x[i, j] = standard_normal()

    cov = cov.astype(np.double)
    (u, s, v) = svd(cov)

    x = np.dot(x, np.sqrt(s)[:, None] * v)
    x += mean
    return x
