import numpy as np

def solve(A,b):
    """solve a linear system using Gauss elimination 
    parameters
    ----------
    a : numpy.ndarray, shape=(M,M)
        coefficient matrix 
    b : numpy.ndarray, shape=(M,)
        the dependent variable values
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m = len(A)
    ndim = len(A.shape)
    if ndim != 2:
        raise ValueError("a has {ndim} dimensions, expected 2")
    if A.shape[1]!= m:
        raise ValueError(f"A has {m} rows and {A.shape[1]} columns, expected square matrix")
    ndimb = len(b.shape)
    if ndimb != 1:
        raise ValueError(f"b has {ndimb} dimensions, expected 1D")
    if b.shape[0] != m:
        raise ValueError(f"A has {m} rows, b has {b.shape[0]} rows, expected same")
    b = np.reshape(b, (len(b),1))
    aug = np.hstack([A,np.reshape(b,(m,1))])
    
