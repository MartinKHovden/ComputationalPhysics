import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("monte_carlo_importance_sampling.txt")
n_values = data["N"]
values = data["value"]
exact_value =np.repeat( (5*np.pi**2)/(16**2), repeats=8)

#Print numerical values vs. exact values as a function of N.
plt.plot(n_values, values, linestyle = "--", marker="o")
plt.plot(n_values, exact_value, linestyle = "--", marker="o")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Integral value")
plt.title("Monte Carlo importance sampling")
plt.legend(["Numerical value", "Exact value"])
plt.show()

error = abs(values - exact_value)
#Print error as a function of N.
plt.plot(n_values, error, linestyle = "--", marker="o")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Error")
plt.title("Monte Carlo importance sampling error")
plt.legend(["Numerical value"])
plt.show()

variance = data["variance"]
#Plot variance as a function of N.
plt.plot(n_values, variance, linestyle = "--", marker="o")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Variance")
plt.title("Monte Carlo importance sampling variance")
plt.legend(["Numerical value variance"])

plt.show()

data_brute_force = pd.read_csv("monte_carlo.txt")

time_brute_force = data_brute_force["time"]
time_importance_sampling = data["time"]

error_importance_sampling = abs(exact_value - values)
error_brute_force = abs(exact_value - data_brute_force["value"])

# print(time_brute_force, time_importance_sampling)
print(error_brute_force, error_importance_sampling)
