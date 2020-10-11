"""Solves part c of the project

This file contains code that produces the plots and info from running FE, BE and CN
in one dimension. 
"""
from diffusion_solvers import forward_euler, backward_euler, crank_nicolson, analytic_solution_1D
import numpy as np
import matplotlib.pyplot as plt

x_min = 0    # Minimum value of x in the domain.
x_max = 1    # Maximum value of x in the domain.

dx = 0.1    #Spatial step
dt = 1e-3   #Time-step

t = 1  #Max time and where to stop the simulation.

#Calculates number of x-values and number of y-values.
num_x_values = (x_max - x_min)/dx
num_t_values = float(t)/dt

#Sets up the grid values.
x_values = np.linspace(x_min, x_max, num_x_values +1)
t_values = np.linspace(0, t, num_t_values+1)

#Defining the boundary and intitial conditions.
g = (lambda x : 0)
a = (lambda t : 0)
b = (lambda t : 1)

#Choose time to plot and calculate the index of this time.
plot_time = 0.02
plot_index = int(plot_time*t/dt)

#Updates some parameters for matplotlib to get nice plots.
params = {'figure.figsize': (20, 5), "legend.fontsize":15, "axes.labelsize": 15}
plt.rcParams.update(params)

plt.subplot(121)

#Plotting the solution from the different numerical methods.

#Calls the function one time to compile the numba code:
u_forward_euler = forward_euler(x_min, x_max, 1, 0.5, t, g, a, b)

#Then runs it normally to achieve the optimal speedup:
u_forward_euler = forward_euler(x_min, x_max, dx, dt, t, g, a, b)
u_backward_euler = backward_euler(x_min, x_max, dx, dt, t, g, a, b)
u_cn = crank_nicolson(x_min, x_max, dx, dt, t, g, a, b)
analytic_solution = analytic_solution_1D(x_values, t_values[plot_index])


plt.plot(x_values, u_forward_euler[plot_index,:], linestyle="--", marker = "x", alpha = 0.9)
plt.plot(x_values, u_backward_euler[plot_index,:], linestyle="--", marker="^", alpha = 0.9)
plt.plot(x_values, u_cn[plot_index,:], linestyle="--", marker="o", alpha = 0.9)
plt.plot(x_values, analytic_solution, alpha = 0.8)

plt.legend(["Forward Euler", "Backward Euler", "CN", "Analytic"])
plt.title("Solution for dx = %.4f, dt = %.5f, t = %.3f" %(dx,dt, plot_time), fontsize=15)
plt.xlabel("X",fontsize=20)
plt.ylabel("u", fontsize=20)
plt.ylim(-0.1,1.1)

plt.subplot(122)

#Plotting the error for the different numerical methods compared to the analytical solution.
error_forward_euler = analytic_solution - u_forward_euler[plot_index,:]
error_backward_euler = analytic_solution- u_backward_euler[plot_index,:]
error_cn = analytic_solution - u_cn[plot_index,:]

plt.plot(x_values, (error_forward_euler), linestyle="--",marker="x", alpha= 0.9)
plt.plot(x_values, (error_backward_euler), linestyle="--",marker="^",  alpha = 0.9)
plt.plot(x_values, (error_cn), linestyle="--", marker="o", alpha = 0.9)

plt.xlabel("X",fontsize=20)
plt.ylabel("u", fontsize=20)
plt.title("Error for dx = %.4f, dt = %.5f, t = %.3f" %(dx,dt, plot_time),fontsize=15)
plt.legend(["Forward Euler", "Backward Euler", "CN"])
plt.subplots_adjust( wspace=0.5)

filename = ("solution_and_error_1D_dt_%s_dx_%s_t_%.3f.png"%(dt,dx, plot_time)).replace(".","_",3)
# plt.savefig(r"Plots/" + filename)

plt.show()

print("MSE forward Euler = ", np.mean(sum(error_forward_euler**2)))
print("MSE backward Euler = ", np.mean(sum(error_backward_euler**2)))
print("MSE CN = ", np.mean(sum(error_cn**2)))
print("Max error forward euler = ", max(error_forward_euler))
print("Max error backward euler = ", max(error_backward_euler))
print("Max error cn = ", max(error_cn))
