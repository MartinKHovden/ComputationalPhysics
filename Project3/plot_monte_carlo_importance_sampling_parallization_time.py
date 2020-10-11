import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_parallel = pd.read_csv("monte_carlo_parallel.txt")
n_values = data_parallel["N"]
values = data_parallel["value"]

data_sequential = pd.read_csv("monte_carlo_importance_sampling.txt")
n_values = data_sequential["N"]
values = data_sequential["value"]

time_parallel = data_parallel["time"]
time_sequential = data_sequential["time"]

factor_upgrade = time_sequential/time_parallel

#Plot time difference.
plt.plot(n_values, time_parallel, linestyle = "--", marker="o")
plt.plot(n_values, time_sequential, linestyle = "--", marker="o")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Time [s]")
plt.title("Time difference for Monte Carlo with importance sampling")
plt.legend(["Parallel code", "Sequential code"])
plt.show()

#Plot speed-up factor.
plt.plot(n_values, factor_upgrade, linestyle = "--", marker="o")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Speed-up factor")
plt.title("Monte Carlo importance sampling speed-up factor parallel")
plt.legend(["speed-up factor"])
plt.show()
