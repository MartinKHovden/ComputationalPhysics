import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("monte_carlo.txt")

n_values = data["N"]
values = data["value"]
exact_value =np.repeat( (5*np.pi**2)/(16**2), repeats=8)

#Print numerical values vs. exact values as a function of N.
plt.plot(n_values, values, linestyle = "--", marker="o")
plt.plot(n_values, exact_value, linestyle = "--", marker="o")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Integral value")
plt.title("Monte Carlo brute force")
plt.legend(["Numerical value", "Exact value"])
plt.show()

error = abs(values - exact_value)
#Plot error as a function of N.
plt.plot(n_values, error, linestyle = "--", marker="o")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Error")
plt.title("Monte Carlo brute force error")
plt.legend(["Numerical value"])
plt.show()

variance = data["variance"]
#Plot variance as a function of N.
plt.plot(n_values, variance, linestyle = "--", marker="o")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Variance")
plt.title("Monte Carlo brute force variance")
plt.legend(["Numerical value variance"])
plt.show()
