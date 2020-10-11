from Project1 import thomas_solver
import matplotlib.pyplot as plt
import numpy as np

#Compares the thomas algorithm for different numbers of n, and plots the result. 

for i in range(1, 5):
    n = 10**i
    a = -np.ones(n)
    b = 2*np.ones(n)
    c = -np.ones(n)

    x = np.linspace(0,1,n+2)
    h = 1./(n+1)
    f = h*h*100*np.exp(-10*x)

    v = thomas_solver(a,b,c,f)
    u = 1 - (1 - np.exp(-10))*x - np.exp(-10*x)
    plt.subplot(int("22" + str(i)))
    plt.plot(x,v)
    plt.title("n = 10**" + str(i))
    plt.plot(x,u)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend(["numerical", "exact"])

plt.show()
