"""
This file contains various methods for solving the diffusion equation in both
one and two dimensions.

In one dimension:
    - Forward-euler
    - Backward-euler
    - Crank-Nicolson

Solves the problem du/dt = d^2u/dx^2
with chosen boundary conditions.

In two dimensions:
    - Forward-euler

Solves the problem du/dt = d^2u/dx^2 + d^2u/dy^2
with chosen boundary conditions.

"""
import numpy as np
from numba import jit
import time
import pytest
import multiprocessing as mp
import scipy as sc
from scipy.sparse.linalg import spsolve

def forward_euler(x_min, x_max, dx, dt, t, g, a, b):
    """Solves the heat equation using forward euler.

    Function for solving the diffusion equation in one dimension with
    given boundary conditions.

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
    #Calls function for solving for all time values and times it.
    u = update_forward_euler(num_t_values, num_x_values, u, alpha)
    end_time = time.time()

    print("Time used forward Euler: ", end_time - start_time, " seconds")

    return u

@jit(nopython=True)
def update_forward_euler(num_t_values, num_x_values, u, alpha):
    """Function for finding the u-matrix in forward euler.

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
    for t_index in range(1, num_t_values):
        for x_index in range(1, num_x_values-1):
            #Updating for the next time-step using vectorization of the forward euler step.
            u[t_index, x_index] = alpha*(u[t_index-1, x_index + 1] - 2*u[t_index-1, x_index] + u[t_index-1, x_index - 1]) + u[t_index-1, x_index]

    return u

def backward_euler(x_min, x_max, dx, dt, t, g, a, b):
    """Solves the heat equation using backward euler.

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

    Returns
    -------
    solutions: numpy matrix
        Contains the solution for each time step up to t.
        First dimension: Time steps
        Second dimension: Spacial steps.

    """

    num_t_values = int(float(t)/dt + 1)
    num_x_values = int(float(x_max - x_min)/dx + 1)

    #Matrix with solution for each time step.
    u = np.zeros((num_t_values, num_x_values))

    x_values = np.linspace(x_min, x_max, num_x_values)
    t_values = np.linspace(0, t, num_t_values)

    #Initializes matrix with boundary conditions.
    for t_index in range(0, num_t_values):
        u[t_index, 0] = a(t_values[t_index])
        u[t_index, -1] = b(t_values[t_index])

    #Initializes matrix with initial conditions.
    for x_index in range(1, num_x_values-1):
        u[0, x_index] = g(x_values[x_index])

    #Sets up the matrix on the l.h.s.
    A = 2*np.eye(num_x_values) - np.eye(num_x_values, k=1) - np.eye(num_x_values, k=-1)

    A = np.eye(num_x_values) + (dt/(dx**2))*A

    A[0,0] = A[-1,-1] = 1
    A[0,1] = A[-1,-2] = 0

    start_time = time.time()
    #Calculates solution for each time step and times it.
    u = update_backward_euler(num_t_values, num_x_values, u, dt, A, a, b)
    end_time = time.time()

    print("Time used backward Euler: ", end_time - start_time, " seconds")

    return u

# @jit(nopython=True)
def update_backward_euler(num_t_values, num_x_values, u, dt, A, a, b):
    """Function for finding the u-matrix in backward euler.


    Uses the scipy sparse functionality to solve the tridiagonal system of equations.
    This speeds up the process significantly.

    Parameters
    ----------
    num_t_values: int
        Number of t-values in the grid.
    num_x_values: int
        Number of x-values in the grid.
    u: numpy 2d-array
        Empty matrix that will contain the solution for each time-step.
    dt: float
        Grid space.
    alpha: float

    Returns
    -------
    u: numpy 2d-array
        Matrix that contains the solution for each time-step.

    """
    A = sc.sparse.csc_matrix(A)
    for i in range(1, num_t_values):
        u[i,:] = spsolve(A , u[i-1, :] )
        u[i,-1] = b(1)
        u[i,0] = a(1)
    return u

def crank_nicolson(x_min, x_max, dx, dt, t, g, a, b):
    """Solves the heat equation using crank nicolson.

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

    Returns
    -------
    solutions: numpy matrix
        Contains the solution for each time step up to t.
        First dimension: Time steps
        Second dimension: Spacial steps.

    """
    num_t_values = int(float(t)/dt + 1)
    num_x_values = int(float(x_max - x_min)/dx + 1)

    #Initializes matrix to contain solution for each time step.
    u = np.zeros((num_t_values, num_x_values))

    x_values = np.linspace(x_min, x_max, num_x_values)
    t_values = np.linspace(0, t, num_t_values)

    #Initializes the boundary conditions.
    for t_index in range(0, num_t_values):
        u[t_index, 0] = a(t_values[t_index])
        u[t_index, -1] = b(t_values[t_index])

    #Initializes the initial value for t = 0.
    for x_index in range(1, num_x_values-1):
        u[0, x_index] = g(x_values[x_index])

    alpha = dt/(dx**2)

    #Sets up the matrix.
    A = 2*np.eye(num_x_values) - np.eye(num_x_values, k=1) - np.eye(num_x_values, k=-1)
    # A = (1/(dx**2))*A
    # A = alpha*A

    A[0,0] = A[-1,-1] = 1
    A[0,1] = A[-1,-2] = 0

    start_time = time.time()
    #Calculates the solution for each time step.
    u = update_cn(num_t_values, num_x_values, u, alpha, A, a, b)
    end_time = time.time()

    print("Time used Crank Nicolson : ", end_time - start_time, " seconds")

    return u

# @jit(nopython=True)
def update_cn(num_t_values, num_x_values, u, alpha, A, a, b):
    """Function for finding the u-matrix in Crank-Nicolson.

    Parameters
    ----------
    num_t_values: int
        Number of t-values in the grid.
    num_x_values: int
        Number of x-values in the grid.
    u: numpy 2d-array
        Empty matrix that will contain the solution for each time-step.
    dt: float
        Grid space.
    A: numpy 2d-array
        The difference scheme matrix. Tridiagonal.

    Returns
    -------
    u: numpy 2d-array
        Matrix that contains the solution for each time-step.

    """
    temp1 = (2*np.eye(num_x_values) + alpha*A)
    temp1[0,0] = temp1[-1,-1] = 1
    temp1[0,1] = temp1[-1,-2] = 0
    temp1 = sc.sparse.csc_matrix(temp1)

    temp2 = 2*np.eye(num_x_values) - alpha*A
    temp2[0,0] = temp2[-1,-1] = 1
    temp2[0,1] = temp2[-1,-2] = 0

    for i in range(1, num_t_values):
        forward_step = np.dot(temp2,u[i-1,:])
        u[i,:] = spsolve(temp1,forward_step)
        u[i,-1] = b(1)
        u[i,0] = a(1)

    return u

def forward_euler_2d(x_min, x_max, y_min, y_max, dx, dy, dt, t, g):
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

    start_time = time.time()
    u = iterate(num_t_values, num_x_values, u_prev, dt, dx, dy)
    end_time = time.time()
    print("time: ", end_time - start_time, " seconds")

    return u, x_values, y_values


@jit(nopython=True)
def iterate(num_t_values, num_x_values, u_prev, dt, dx, dy):
    """Function for finding the solution after num_t_values timesteps
    for the forward euler method in two dimensions.

    Parameters
    ----------
    num_t_values: int
        Number of t_values.
    num_x_values: int
        Number of x-values in the grid.
    u_prev: numpy 2d matrix
        Matrix that contains the solution at the previous timestep.

    Returns
    -------
    u_prev: numpy 2d matrix
        The solution after num_t_values steps from the previous timestep.

    """
    for t_index in range(1, num_t_values):
        u_prev[1:-1, 1:-1] = u_prev[1:-1, 1:-1] + dt*(  (u_prev[0:-2, 1:-1] -2*u_prev[1:-1,1:-1] + u_prev[2:,1:-1])/(dx**2) + \
                                                    (u_prev[1:-1, 0:-2] -2*u_prev[1:-1,1:-1] + u_prev[1:-1,2:])/(dy**2)  )
    return u_prev

def initialize_matrix(num_x_values, x_values, num_y_values, y_values, u_prev, g):
    """Function for initializing the matrix for the forward euler in 2d.

    Updates the matrix in place.

    Parameters
    ----------
    num_x_values: int
        Number of x-values in the grid.
    x_values: numpy vector
        Vector containing the grid points in x-direction.
    num_y_values: int
        Number of y-values in the grid.
    y_values: numpy vector
        Vector containing the grid points in y-direction.
    u_prev: numpy vector
        Solution for the last time-step.
    g: function(x, y), returns int.
        Function for the initial condition that returns the value at x, y.

    """
    for x_index in range(1, num_x_values-1):
        for y_index in range(1, num_y_values-1):
            u_prev[y_index, x_index] = g(x_values[x_index],y_values[y_index])

    for y_index in range(0, num_y_values):
        u_prev[y_index, 0] = 0
        u_prev[y_index, -1] = 0

    for x_index in range(0, num_x_values):
        u_prev[0, x_index] = 0
        u_prev[-1, x_index] = 0

def analytic_solution_1D(x_values, t):
    """Analtytic solution to the diffusion equation with u(x,0) = 0, u(0,t) = 0 and u(1,t) = 1.

    Parameters
    ----------
    x_values: numpy vector
        Vector with x-values in the grid.
    t: float
        The time step to calculate the numerical solution.

    Returns
    -------
    Returns the values of the analytical solution at the grid points for the given
    initial conditions and boundary conditions.
    """
    v = 0

    for k in range(1, 100):
        v += -2/(k*np.pi)*(-1)**(k+1)*np.exp(-(k*np.pi)**2*t)*np.sin(k*np.pi*x_values)

    return x_values + v

def analytic_solution_2D(x_values, y_values, t):
    """Analytic solution to the diffusion equation in two dimensions.

    Solution for the initial condition u(x,y,0) = sin(pi*x)*sin(pi*y) and the boundary
    conditions equal to zero.

    Parameters
    ----------
    x_values: numpy vector
        Vector with x-values in the grid.
    y_values: numpy vector
        Vector with y-values in the grid.
    t: float
        The time step to calculate the numerical solution.

    Returns
    -------
    Returns the values of the analytical solution at the grid points for the given
    initial conditions and boundary conditions.

    """
    return np.sin(np.pi*x_values)*np.sin(np.pi*y_values)*np.exp(-2*(np.pi**2)*t)


##############################
#### TESTING OF THE CODE. ####
##############################

def test_forward_euler_1():
    x_min = 0
    x_max = 1

    dx = 0.01    #Spatial step
    dt = 0.5*(dx**2)#1e-3   #Time-step

    t = 1  #Max time

    num_x_values = (x_max - x_min)/dx
    num_t_values = float(t)/dt

    x_values = np.linspace(x_min, x_max, num_x_values +1)
    t_values = np.linspace(0, t, num_t_values+1)

    #Defining the boundary and intitial conditions for test1.
    g = (lambda x : 0)
    a = (lambda t : 0)
    b = (lambda t : 0)
    u_forward_euler = forward_euler(x_min, x_max, dx, dt, t, g, a, b)

    #checks if the boundary conditions are fulfilled.
    assert(u_forward_euler[0,0] == 0 and u_forward_euler[0,-1] == 0)
    assert(u_forward_euler[-1,0] == 0 and u_forward_euler[-1,-1] == 0)

def test_forward_euler_2():
    x_min = 0
    x_max = 1

    dx = 0.01    #Spatial step
    dt = 0.5*(dx**2)#1e-3   #Time-step

    t = 1  #Max time

    num_x_values = (x_max - x_min)/dx
    num_t_values = float(t)/dt

    x_values = np.linspace(x_min, x_max, num_x_values +1)
    t_values = np.linspace(0, t, num_t_values+1)
    #Defining the boundary conditions and initial conditions for test2.
    g = (lambda x : np.sin(np.pi*x))
    a = (lambda t : 0)
    b = (lambda t : 0)

    u_forward_euler = forward_euler(x_min, x_max, dx, dt, t, g, a, b)

    assert(u_forward_euler[0,0] == 0 and u_forward_euler[0,-1] == 0)
    assert(u_forward_euler[-1,0] == 0 and u_forward_euler[-1,-1] == 0)

def test_forward_euler_3():
    x_min = 0
    x_max = 1

    dx = 0.01    #Spatial step
    dt = 0.5*(dx**2)#1e-3   #Time-step

    t = 1  #Max time

    num_x_values = (x_max - x_min)/dx
    num_t_values = float(t)/dt

    x_values = np.linspace(x_min, x_max, num_x_values +1)
    t_values = np.linspace(0, t, num_t_values+1)

    g = (lambda x : 0)
    a = (lambda t : 0)
    b = (lambda t : 1)

    u_forward_euler = forward_euler(x_min, x_max, dx, dt, t, g, a, b)

    assert(u_forward_euler[0,0] == 0 and u_forward_euler[0,-1] == 1)
    assert(u_forward_euler[-1,0] == 0 and u_forward_euler[-1,-1] ==1)

def test_forward_euler_4():
    x_min = 0
    x_max = 1

    dx = 0.01    #Spatial step
    dt = 0.5*(dx**2)#1e-3   #Time-step

    t = 1  #Max time

    num_x_values = (x_max - x_min)/dx
    num_t_values = float(t)/dt

    x_values = np.linspace(x_min, x_max, num_x_values +1)
    t_values = np.linspace(0, t, num_t_values+1)

    g = (lambda x : 1-x)
    a = (lambda t : 0)
    b = (lambda t : 1)

    u_forward_euler = forward_euler(x_min, x_max, dx, dt, t, g, a, b)

    assert(u_forward_euler[0,0] == 0 and u_forward_euler[0,-1] == 1)
    assert(u_forward_euler[-1,0] == 0 and u_forward_euler[-1,-1] ==1)
    assert(np.mean(u_forward_euler[-1,:] - g(x_values)) < 0.0001)

def test_backward_euler_1():
    x_min = 0
    x_max = 1

    dx = 0.01    #Spatial step
    dt = 0.5*(dx**2)#1e-3   #Time-step

    t = 1  #Max time

    num_x_values = (x_max - x_min)/dx
    num_t_values = float(t)/dt

    x_values = np.linspace(x_min, x_max, num_x_values +1)
    t_values = np.linspace(0, t, num_t_values+1)

    #Defining the boundary and intitial conditions for test1.
    g = (lambda x : 0)
    a = (lambda t : 0)
    b = (lambda t : 0)
    u_forward_euler = backward_euler(x_min, x_max, dx, dt, t, g, a, b)

    #checks if the boundary conditions are fulfilled.
    assert(u_forward_euler[0,0] == 0 and u_forward_euler[0,-1] == 0)
    assert(u_forward_euler[-1,0] == 0 and u_forward_euler[-1,-1] == 0)

def test_backward_euler_2():
    x_min = 0
    x_max = 1

    dx = 0.01    #Spatial step
    dt = 0.5*(dx**2)#1e-3   #Time-step

    t = 1  #Max time

    num_x_values = (x_max - x_min)/dx
    num_t_values = float(t)/dt

    x_values = np.linspace(x_min, x_max, num_x_values +1)
    t_values = np.linspace(0, t, num_t_values+1)
    #Defining the boundary conditions and initial conditions for test2.
    g = (lambda x : np.sin(np.pi*x))
    a = (lambda t : 0)
    b = (lambda t : 0)

    u_backward_euler = backward_euler(x_min, x_max, dx, dt, t, g, a, b)

    assert(u_backward_euler[0,0] == 0 and u_backward_euler[0,-1] == 0)
    assert(u_backward_euler[-1,0] == 0 and u_backward_euler[-1,-1] == 0)

def test_backward_euler_3():
    x_min = 0
    x_max = 1

    dx = 0.01    #Spatial step
    dt = 0.5*(dx**2)#1e-3   #Time-step

    t = 1  #Max time

    num_x_values = (x_max - x_min)/dx
    num_t_values = float(t)/dt

    x_values = np.linspace(x_min, x_max, num_x_values +1)
    t_values = np.linspace(0, t, num_t_values+1)

    g = (lambda x : 0)
    a = (lambda t : 0)
    b = (lambda t : 1)

    u_backward_euler = backward_euler(x_min, x_max, dx, dt, t, g, a, b)

    assert(u_backward_euler[0,0] == 0 and u_backward_euler[0,-1] == 1)
    assert(u_backward_euler[-1,0] == 0 and u_backward_euler[-1,-1] ==1)

def test_backward_euler_4():
    x_min = 0
    x_max = 1

    dx = 0.01    #Spatial step
    dt = 0.5*(dx**2)#1e-3   #Time-step

    t = 1  #Max time

    num_x_values = (x_max - x_min)/dx
    num_t_values = float(t)/dt

    x_values = np.linspace(x_min, x_max, num_x_values +1)
    t_values = np.linspace(0, t, num_t_values+1)

    g = (lambda x : 1-x)
    a = (lambda t : 0)
    b = (lambda t : 1)

    u_backward_euler = backward_euler(x_min, x_max, dx, dt, t, g, a, b)

    assert(u_backward_euler[0,0] == 0 and u_backward_euler[0,-1] == 1)
    assert(u_backward_euler[-1,0] == 0 and u_backward_euler[-1,-1] ==1)
    assert(np.mean(u_backward_euler[-1,:] - g(x_values)) < 0.0001)

def test_cn_1():
    x_min = 0
    x_max = 1

    dx = 0.01    #Spatial step
    dt = 0.5*(dx**2)#1e-3   #Time-step

    t = 1  #Max time

    num_x_values = (x_max - x_min)/dx
    num_t_values = float(t)/dt

    x_values = np.linspace(x_min, x_max, num_x_values +1)
    t_values = np.linspace(0, t, num_t_values+1)

    #Defining the boundary and intitial conditions for test1.
    g = (lambda x : 0)
    a = (lambda t : 0)
    b = (lambda t : 0)
    u_cn = crank_nicolson(x_min, x_max, dx, dt, t, g, a, b)

    #checks if the boundary conditions are fulfilled.
    assert(u_cn[0,0] == 0 and u_cn[0,-1] == 0)
    assert(u_cn[-1,0] == 0 and u_cn[-1,-1] == 0)

def test_cn_2():
    x_min = 0
    x_max = 1

    dx = 0.01    #Spatial step
    dt = 0.5*(dx**2)#1e-3   #Time-step

    t = 1  #Max time

    num_x_values = (x_max - x_min)/dx
    num_t_values = float(t)/dt

    x_values = np.linspace(x_min, x_max, num_x_values +1)
    t_values = np.linspace(0, t, num_t_values+1)
    #Defining the boundary conditions and initial conditions for test2.
    g = (lambda x : np.sin(np.pi*x))
    a = (lambda t : 0)
    b = (lambda t : 0)

    u_cn = crank_nicolson(x_min, x_max, dx, dt, t, g, a, b)

    assert(u_cn[0,0] == 0 and u_cn[0,-1] == 0)
    assert(u_cn[-1,0] == 0 and u_cn[-1,-1] == 0)

def test_cn_3():
    x_min = 0
    x_max = 1

    dx = 0.01    #Spatial step
    dt = 0.5*(dx**2)#1e-3   #Time-step

    t = 1  #Max time

    num_x_values = (x_max - x_min)/dx
    num_t_values = float(t)/dt

    x_values = np.linspace(x_min, x_max, num_x_values +1)
    t_values = np.linspace(0, t, num_t_values+1)

    g = (lambda x : 0)
    a = (lambda t : 0)
    b = (lambda t : 1)

    u_cn = crank_nicolson(x_min, x_max, dx, dt, t, g, a, b)

    assert(u_cn[0,0] == 0 and u_cn[0,-1] == 1)
    assert(u_cn[-1,0] == 0 and u_cn[-1,-1] ==1)

def test_cn_4():
    x_min = 0
    x_max = 1

    dx = 0.01    #Spatial step
    dt = 0.5*(dx**2)#1e-3   #Time-step

    t = 1  #Max time

    num_x_values = (x_max - x_min)/dx
    num_t_values = float(t)/dt

    x_values = np.linspace(x_min, x_max, num_x_values +1)
    t_values = np.linspace(0, t, num_t_values+1)

    g = (lambda x : 1-x)
    a = (lambda t : 0)
    b = (lambda t : 1)

    u_cn = crank_nicolson(x_min, x_max, dx, dt, t, g, a, b)

    assert(u_cn[0,0] == 0 and u_cn[0,-1] == 1)
    assert(u_cn[-1,0] == 0 and u_cn[-1,-1] ==1)
    assert(np.mean(u_cn[-1,:] - g(x_values)) < 0.0001)
