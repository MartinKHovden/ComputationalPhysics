import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy.linalg import lu, solve
import time

def lu_solver(A, w):
    """
    Function for solving the tridiagonal system, using lu-decomposition.
    A is a tridiagonal matrix. w is the rhs.
    The function returns the solution of the system.
    """
    (P,L,U) = lu(A)
    y = solve(L, w)
    x = solve(U, y)
    return x

def main():
    n = 10**3

    A = 2*np.eye(n) - np.eye(n, k = -1) - np.eye(n, k=1)

    x = np.linspace(0,1,n)
    h = 1./n
    f = h**2*100*np.exp(-10*x)
    u = 1 - (1 - np.exp(-10))*x - np.exp(-10*x)

    plt.plot(x, u)
    start = time.time()
    plt.plot(x, lu_solver(A, f))
    end = time.time()
    plt.legend(["exact", "num"])

    plt.show()

    print(end-start)

if __name__ == "__main__":
    main()
