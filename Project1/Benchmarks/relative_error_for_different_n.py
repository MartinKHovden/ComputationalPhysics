from Project1 import thomas_solver, solver_symmetric, relative_error
import matplotlib.pyplot as plt
import numpy as np
import time

k = 7

#Contains the largest relative error for each value of n
rel_error = np.zeros(k)

for i in range(1,k+1):
    #Computes the relative error for different values of n
    n = 10**i

    #Vector representation
    a = -np.ones(n)
    b = 2*np.ones(n)
    c = -np.ones(n)

    x = np.linspace(0,1,n+2)
    h = 1./(n+1)
    f = h*h*100*np.exp(-10*x)  #RHS

    #Solution
    v = solver_symmetric(a,b,c,f)

    #Exact solution
    u = 1 - (1 - np.exp(-10))*x - np.exp(-10*x)

    rel_error[i-1] = max(relative_error(u,v))

i = 1
for err in rel_error:
    print("n = ", 10**i, "Relative error: " ,err)
    i+=1
