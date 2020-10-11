from scipy.linalg import lu, solve
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time
import sys

@jit
def thomas_solver(a, b, c, f_values):
    """
    Solver of general tridiagonal systems using the Thomas method. a, b and c are
    vectors that contain the diagonal elements and the 1st off-diagonal elements
    in each direction. a, b and c should be numpy arrays with the same length.
    The funtion returns the solution of the system, v.
    a, b and c have length n, while f should have length n+2.

    Parameters
    ----------
    a,b,c : numpy array
    vectors that contains the diagonal elements of the tridiagonal matrix

    f_values : numpy array
    vector that contains the rhs. of the equation

    Returns
    ------------
    v : numpy array
    vector that contains the solution at the grid points

    """
    a[0] = c[-1] = 0
    n = len(a)
    b_tilde = np.zeros(n)
    b_tilde[0] = b[0]
    f_tilde = np.zeros(n)
    f_tilde[0] = f_values[1]
    v = np.zeros(n+2) #contains the solution
    # Forward substitution
    for i in range(1, n+1):
        b_tilde[i] = b[i] - (a[i]*c[i-1])/b_tilde[i-1]
        f_tilde[i] = f_values[i+1] - (a[i]*f_tilde[i-1])/b_tilde[i-1]

    # v[-1] = f_tilde[-1]/b_tilde[-1]

    k = n+1
    #Backward substitution
    while k >= 2:
        v[k-1] = (f_tilde[k-2] - c[k-2]*v[k])/b_tilde[k-2]
        k-=1

    return v


@jit
def solver_symmetric(a,b,c,f_values):
    """
    Solver of symmetric tridiagonal systems. a, b and c are
    vectors that contain the diagonal elements and the 1st off-diagonal elements
    in each direction. a, b and c should be numpy arrays with the same length.
    The funtion returns the solution of the system, v.
    a, b and c have length n, while f should have length n+2.

    Parameters
    -----------
    a,b,c : numpy array
    vectors that contains the diagonal elements of the tridiagonal matrix

    f_values : numpy array
    vector that contains the rhs. of the equation

    Returns
    ------------
    v : numpy array
    vector that contains the solution at the grid points

    """
    n = len(a)
    d_tilde = np.zeros(n)
    d_tilde[0] = 2
    f_values = f_values[1:-1]
    f_tilde = np.zeros(n)
    f_tilde[0] = f_values[0]
    for i in range(1,n+1):
        d_tilde[i] = ((i+1)+1)/(i+1)
        f_tilde[i] = f_values[i] + f_tilde[i-1]/d_tilde[i-1]

    v = np.zeros(n+2)
    # v[-2] = f_tilde[-1]/d_tilde[-1]

    k = n+1

    while k >= 2:
        v[k-1] = (f_tilde[k-2] + v[k])/d_tilde[k-2]
        k-=1
    return v


@jit
def relative_error(u, v):
    """
    Computes the relative error in the open interval.
    Returns an array of relative errors for each grid-point.
    v is approximate solution.
    u is exact solution.
    """
    return np.log10(abs((v[1:-2]-u[1:-2])/u[1:-2]))

def lu_solver(A, w):
    """
    Function for solving the tridiagonal system, using lu-decomposition.
    A is a tridiagonal matrix. w is the rhs.
    The function returns the solution of the system.
    A should have dimension n x n, and w should have length n+2.

    Parameters
    ----------
    A : numpy array
        Matrix

    w : numpy array
        Vector

    Returns
    -------
    v : numpy array
        Vector
    """
    v = np.zeros(len(w))
    (P,L,U) = lu(A)
    y = solve(L, w[1:-1])
    x = solve(U, y)
    v[1:-1] = x
    return v


def main():
    """
    Program takes two inputs. First input is exponent and second
    argument are chosen method. If no arguments are given, the Program
    chooses predefined methods and n.
    """

    if len(sys.argv) >=2:
        n = 10**int(sys.argv[1])
    else:
        print("default power of n: 4. Add one command line argument to change")
        print("Usage: arg1: exponent, arg2: preffered method")
        n = 10**4
    #Vector representation:
    a = -np.ones(n)
    b = 2*np.ones(n)
    c = -np.ones(n)

    x = np.linspace(0,1,n+2)
    h = 1./(n+1)
    f = h*h*100*np.exp(-10*x)

    #matrix representation:
    A = 2*np.eye(n) - np.eye(n, k = -1) - np.eye(n, k=1)

    start = time.time()
    # print(f)

    if len(sys.argv) == 3:
        if sys.argv[2].lower() == "lu":
            v = lu_solver(A, f)              #lu-solver

        elif sys.argv[2].lower() == "thomas":
            v = thomas_solver(a, b, c, f)

        elif sys.argv[2].lower() == "symmetric" or sys.argv[2].lower() == "sym":
            v= (solver_symmetric(a,b,c,f))   #Symmetric-solver
        else:
            print("The argument are not valied. Default method are thomas-solver. \n Use: thomas, lu, sym to choose preferred method ")
            v = thomas_solver(a, b, c, f)

    else:
        v = thomas_solver(a, b, c, f)




        # v = thomas_solver(a, b, c, f)         #Thomas-solver
        # v= (solver_symmetric(a,b,c,f))   #Symmetric-solver
    end = time.time()

    print("Execution time: ", end-start)

    u = 1 - (1 - np.exp(-10))*x - np.exp(-10*x)
    plt.plot(x,v)
    plt.plot(x,u)
    plt.legend(["num", "exact"])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    main()
