from Project1 import thomas_solver, solver_symmetric, relative_error
import matplotlib.pyplot as plt
import numpy as np
import time

k = 7

#Vectors of cpu-times for different values of n
cpu_times_thomas = np.zeros(k)
cpu_times_symmetric = np.zeros(k)


#Have to compile the code first, so that it does not disturb the
#actual timing of the algorithms. I'm using numba for the functions.
n = 1
a = -np.ones(n)
b = 2*np.ones(n)
c = -np.ones(n)

x = np.linspace(0,1,n+2)
h = 1./(n+1)
f = h*h*100*np.exp(-10*x)

v = thomas_solver(a,b,c,f)
v = solver_symmetric(a,b,c,f)


#Now the code is ready to do the CPU-test, after compiling the code
for i in range(1,k+1):
    n = 10**i

    #vector representation
    a = -np.ones(n)
    b = 2*np.ones(n)
    c = -np.ones(n)

    x = np.linspace(0,1,n+2)
    h = 1./(n+1)
    f = h*h*100*np.exp(-10*x)

    start1 = time.time()
    v = thomas_solver(a,b,c,f)
    end1 = time.time()
    cpu_times_thomas[i-1] = end1 - start1

    start2 = time.time()
    v2 = solver_symmetric(a,b,c,f)
    end2 = time.time()

    cpu_times_symmetric[i-1] = end2-start2


i = 1
for x,y in zip(cpu_times_thomas, cpu_times_symmetric):
    print("n = ",10**i, "cpu:Thomas:", x,"CPU:sym: ", y)
    i+=1
