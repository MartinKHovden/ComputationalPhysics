import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

from mpl_toolkits import mplot3d

def forward_euler_2d(x_min, x_max, y_min, y_max, dx, dy, dt, t, g, a, b):
    """Solves the heat equation in 2 dimensions using forward euler.

    Parameters
    ----------
    x_min: float
        Lower limit of x-dimension.
    x_max: float
        Upper limit of x-dimension.
    y_min: float
        Lower limit of y-dimension.
    y_max: float
        Upper limit of y-dimension.
    dx: float
        Spacing between each grid point in x-dimension.
    dy: float
        Spacing between each grid point in y-dimension.
    dt: float
        Spacing between time steps.
    t: float
        max time step.
    g: function(x,y)
        Initial condition at time = 0.

    Returns
    -------
    u: numpy matrix
        Contains the solution for the last time step.
        First dimension: y-coordinates
        Second dimension: x-coordinates.
    x_values: numpy vector
        Vector containing the grid points in x-direction.
    y_values: numpy vector
        Vector containing the grid points in y-direction.
    """

    num_t_values = int(float(t)/dt + 1)
    num_x_values = int(float(x_max - x_min)/dx + 1)
    num_y_values = int(float(y_max - y_min)/dy + 1)

    save_num_between = 10

    #Matrix for previous timestep.
    u_prev = np.zeros((num_y_values, num_x_values))

    x_values = np.linspace(x_min, x_max, num_x_values)
    y_values = np.linspace(y_min, y_max, num_y_values)

    t_values = np.linspace(0, t, num_t_values)

    alpha = dt/(dx**2)

    initialize_matrix(num_x_values, x_values, num_y_values, y_values, u_prev, g)


    print(u_prev)

    start_time = time.time()
    u = iterate(num_t_values, num_x_values, u_prev, dt, dx, dy)
    end_time = time.time()
    print("time: ", end_time - start_time, " seconds")

    print(u)

    return u, x_values, y_values


@jit(nopython=True)
def iterate(num_t_values, num_x_values, u_prev, dt, dx, dy):
    # u = np.zeros_like(u_prev)
    for t_index in range(1, num_t_values):
        u_prev[1:-1, 1:-1] = u_prev[1:-1, 1:-1] + dt*(  (u_prev[0:-2, 1:-1] -2*u_prev[1:-1,1:-1] + u_prev[2:,1:-1])/(dx**2) + \
                                                    (u_prev[1:-1, 0:-2] -2*u_prev[1:-1,1:-1] + u_prev[1:-1,2:])/(dy**2)  )
        # u_prev = u
    return u_prev

def initialize_matrix(num_x_values, x_values, num_y_values, y_values, u_prev, g):
    for x_index in range(1, num_x_values-1):
        for y_index in range(1, num_y_values-1):
            u_prev[y_index, x_index] = g(x_values[x_index],y_values[y_index])

    for y_index in range(0, num_y_values):
        u_prev[y_index, 0] = 1
        u_prev[y_index, -1] = 0

    for x_index in range(0, num_x_values):
        u_prev[0, x_index] = 1
        u_prev[-1, x_index] = 1





g = lambda x,y: 0# np.sin(np.pi*x)*np.sin(np.pi*y)
a = lambda t: 0
b = lambda t: 0

u, x_values, y_values = forward_euler_2d(0, 1, 0, 2, 0.01,0.01, 1e-5, 0.1, g, a, b)

fig = plt.figure()
ax = plt.axes(projection="3d")

print(x_values.shape)

X,Y = np.meshgrid(x_values, y_values)
ax.plot_surface(X, Y, u, cmap="viridis")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
