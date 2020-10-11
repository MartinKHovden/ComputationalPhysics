"""
This file contains code for solving the diffusion equation with a heat-source Q.
We will use the code for simulating the temperature in the lithosphere, but the
code can be used for many other problem where there is a heat-source in the domain.
The function solves the equation in one dimension. However, it can also be extended
quite easily to solve 2 dimensional problems also.

The main function is the forward euler method. In addition it contains functions
for calculating the heat in the domain and the second constant.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

def forward_euler(x_min, x_max, dx, dt, t, g, a, b, C_1, C_2, Q, radioactive = False, additional_heat = 0):
    """Solves the heat equation using forward euler.

    Function for solving the diffusion equation in one dimension with
    given boundary conditions and added radioactive elements.

    Parameters
    ----------
    x_min: float
        Lower limit of spacial dimension.
    x_max: float
        Upper limit of spacial dimension.
    dx: float
        Spacing between spacial points.
    dt: float
        Spacing between time steps.
    t: float
        max time step.
    g: function(x)
        Initial condition for u(x,0).
    a: function(t)
        Boundary condition u(0,t)
    b: function(t)
        Boundary condition u(x_max,t)
    C_1: float
        Constant in the scaled equation.
    C_2: function(y)
        Piecewise constant in the scaled equation.
    Q: function(y)
        Heat source function.
    radioactive: bool
        Wheter or not to include heat source in the simulation.
    additional_heat: float
        Additional heat to the normal heat source.

    Returns
    -------
    solutions: numpy matrix
        Contains the solution for each time step up to t.
        First dimension: Time steps
        Second dimension: Spacial steps.
    """

    num_t_values = int(float(t)/dt + 1)
    num_x_values = int(float(x_max - x_min)/dx + 1)

    #Matrix the will contain the solution for each time-step.
    u = np.zeros((num_t_values, num_x_values))

    #Vectors for x-values and t_values in the grid.
    x_values = np.linspace(x_min, x_max, num_x_values)
    t_values = np.linspace(0, t, num_t_values)

    alpha = dt/(dx**2)

    #Loops over the u-matrix and sets the boundary conditions.
    for t_index in range(0, num_t_values):
        u[t_index, 0] = a(t_values[t_index])
        u[t_index, -1] = b(t_values[t_index])

    #Loops over the u-matrix and sets the initial conditions.
    for x_index in range(1, num_x_values-1):
        u[0, x_index] = g(x_values[x_index])

    start_time = time.time()
    u = update_forward_euler(t_values, x_values, u, alpha, dt, C_1, C_2, radioactive, additional_heat)
    end_time = time.time()

    print("Time used forward Euler: ", end_time - start_time, " seconds")

    return u

# @jit(nopython=True)
def update_forward_euler(t_values, x_values, u, alpha, dt, C_1, C_2, radioactive, additional_heat ):
    """Function for finding the u-matrix in forward euler with added heat.

    Parameters
    ----------
    num_t_values: int
        Number of t-values in the grid.
    num_x_values: int
        Number of x-values in the grid.
    u: numpy 2d-array
        Empty matrix that will contain the solution for each time-step.
    alpha: float

    Returns
    -------
    u: numpy 2d-array
        Matrix that contains the solution for each time-step.
    """
    num_t_values = len(t_values)
    num_x_values = len(x_values)
    for t_index in range(1, num_t_values):
        for x_index in range(1, num_x_values-1):
            u[t_index, x_index] = C_1*alpha*(u[t_index-1, x_index + 1] - 2*u[t_index-1, x_index] + u[t_index-1, x_index - 1]) + u[t_index-1, x_index] + dt*C_2(x_values[x_index], Q, radioactive, additional_heat,  t=t_values[t_index])
    return u

def Q(y_bar, radioactive, additional_heat, t):
    """Heat-source function.

    Piece-wise continous function representing the heat produced by the radioactive
    elements in the lithosphere. Can be used without radioactive sources.

    Parameters
    ----------
    y_bar: numpy 1d vector
        Vector containing the scaled y-values
    radioactive: bool
        Wheter or not to include radioactive heat source. If false returns zero.
    additional_heat: float
        If radioactive = true, then we can add more heat in the mantle in addition to the original Q.
        If additional heat = 0, then returns Q as decribed in the report.
    t: double
        Time value.

    Returns
    -------
    Returns the value of Q for the y-values for the given t.

    """
    if radioactive == False:
        return 0
    else:
        y = y_bar
        additional_heat = additional_heat*(0.4*np.exp(-t/4.47) + 0.4*np.exp(-t/14) + 0.2*np.exp(-t/1.25))
        if y < 0 or y > 120/120:
            print("Error out of bounds")
        elif 0 <= y < 80/120:
            return 0.05*(10**(-6)) + additional_heat
        elif 80/120 <= y < 100/120:
            return 0.35*(10**(-6))
        else:
            return 1.4*(10**(-6))


def C_2(y_bar, Q, radioactive, additional_heat, t):
    """Function represents the second constant in the scaled expression. The theory is
    explained in the project report.

    Parameters
    ----------
     y_bar: numpy 1d vector
        Vector containing the scaled y-values
    radioactive: bool
        Wheter or not to include radioactive heat source. If false returns zero.
    additional_heat: float
        If radioactive = true, then we can add more heat in the mantle in addition to the original Q.
        If additional heat = 0, then returns Q as decribed in the report.
    t: double
        Time value.

    Returns
    -------
    temp: double
        Returns the value of the constant for each y at t.
    """
    y = y_bar
    temp = ((t_c)/((rho*c_p)*(T_1-T_0)))*Q(y, radioactive, additional_heat, t=t)
    return temp
