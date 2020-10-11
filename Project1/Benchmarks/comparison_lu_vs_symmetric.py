from Project1 import thomas_solver, lu_solver, solver_symmetric
import matplotlib.pyplot as plt
import numpy as np
import time

k = 4

cpu_symmetric = np.zeros(k)
cpu_lu = np.zeros(k)

for i in range(1,k+1):
    n = 10**i
    #Array representation
    a = -np.ones(n)
    b = 2*np.ones(n)
    c = -np.ones(n)

    x = np.linspace(0,1,n+2)
    h = 1./(n+1)
    f = h*h*100*np.exp(-10*x)

    #matrix representation:
    A = 2*np.eye(n) - np.eye(n, k = -1) - np.eye(n, k=1)

    start1 = time.time()
    v = solver_symmetric(a,b,c,f)
    end1 = time.time()

    start2 = time.time()
    v = lu_solver(A, f)
    end2 = time.time()

    cpu_symmetric[i-1] = end1 - start1
    cpu_lu[i-1] = end2 - start2

print(cpu_symmetric, cpu_lu)
