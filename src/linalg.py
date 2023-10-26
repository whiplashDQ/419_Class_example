import numpy as np


def solve(a, b):
    """Solve a linear system using Gauss elimination.

    Parameters
    ----------
    a : numpy.ndarray, shape=(M,M)
        The coefficient matrix
    b : numpy.ndarray, shape=(M,)
        The dependent variable values

    Returns
    -------
    numpy.ndarray, shape=(M,)
        The solution vector
    """
    # ensures that the coef matrix and rhs vector are ndarrays
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    # make sure that the coef matrix is square
    m = len(a)
    ndim = len(a.shape)
    if ndim != 2:
        raise ValueError(f"A has {ndim} dimensions"
                         + ", should be 2")
    if a.shape[1] != m:
        raise ValueError(f"A has {m} rows and {a.shape[1]} cols"
                         + ", should be square")
    # make sure the rhs vector is a 1d array
    ndimb = len(b.shape)
    if ndimb != 1:
        raise ValueError(f"b has {ndimb} dimensions"
                         + ", should be 1D")
    # make sure number of rhs values matches number of eqns
    if len(b) != m:
        raise ValueError(f"A has {m} rows, b has {len(b)} values"
                         + ", dimensions incompatible")
    # form the augmented matrix [a | b]
    aug = np.hstack([a, np.reshape(b, (m, 1))])
    # solve by the GE algorithm
    for k in range(m):
        # calculate elimination coefficients
        # slice = start(:stop(:step))
        aug[k+1:, k] /= aug[k, k]
        # eliminate below the pivot
        for j in range(k+1, m):
            aug[j, k+1:] -= aug[j, k] * aug[k, k+1:]
    # perform backward substitution
    for k in range(m-1, -1, -1):
        aug[k, -1] = ((aug[k, -1]
                      - np.dot(aug[k, k+1:m],
                               aug[k+1:, -1]))
                      / aug[k, k])
    # return the solution
    return aug[:, -1]