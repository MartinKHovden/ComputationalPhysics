""" Solves part e of the project.

This file contains code for producing plots and results from running the
forward Euler scheme in 2 dimensions.
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time
from mpl_toolkits import mplot3d
from diffusion_solvers import forward_euler_2d, analytic_solution_2D
import matplotlib.ticker as mtick


#Sets up the grid parameters. Uses the same for dx and dy, but this can be changed if wanted.
x_min = 0
x_max = 1
dx = 0.01 #Spatial step length in x-direction.

y_min = x_min
y_max = x_max
dy = dx #Spaial step length in y-direction.

dt = 1e-5 #Step length for time.
t = 0.3   #The last t-value to compute.

#Choose initial condition. The boundaries are set to zero autmatically.
g = lambda x,y: np.sin(np.pi*x)*np.sin(np.pi*y)

# Finds the numerical solution and returns the grid.
u, x_values, y_values = forward_euler_2d(x_min, x_max, y_min, y_max, dx, dy, dt, t, g)

#Sets up a grid where the solution is plotted.
X,Y = np.meshgrid(x_values, y_values)

#Finds the analytical solution.
analytic = analytic_solution_2D(X, Y, t)

#Calculates the error between the numerical solution and the analytical solution.
error = analytic - u

#Updates matplotlib parameters to get nice looking plots.
params = {'figure.figsize': (16, 8)}
plt.rcParams.update(params)

fig = plt.figure()

ax = fig.add_subplot(1,2,1, projection="3d")
ax.plot_surface(X, Y, u, cmap="viridis", alpha = 0.8)
plt.xlabel("X",fontsize=20)
plt.ylabel("Y", fontsize=20)
ax.set_zlabel("u", fontsize=20)
ax.zaxis.labelpad = 20
plt.title("Numerical solution \n dt = %.5f  dx = %.4f dy = %.4f  t = %.3f" %(dt, dx, dy, t), fontsize=13)

ax2 = fig.add_subplot(1,2,2,projection="3d")
ax2.plot_surface(X, Y, error, cmap = "viridis", alpha = 0.8 )
plt.xlabel("X", fontsize=20)
plt.ylabel("Y", fontsize=20)
ax2.set_zlabel("u", fontsize=20)
ax2.zaxis.labelpad = 20
ax2.zaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
plt.setp( ax2.zaxis.get_majorticklabels(), ha="left" )
ax2.zaxis.labelpad = 30
plt.title("Numerical error \n dt = %.5f  dx = %.4f dy = %.4f  t = %.3f" %(dt, dx, dy, t), fontsize=13)

filename = ("solution_and_error_2D_dt_%.5f_dx_%.4f_dy_%.4f_t_%.3f.png"%(dt,dx, dy, t)).replace(".","_",4)
plt.savefig(r"Plots/" + filename)

plt.show()

print("MSE = ", np.mean(error**2))
