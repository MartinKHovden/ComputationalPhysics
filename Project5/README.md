# Computational physics project 5
## Numerical methods for solving the diffusion equation in one and two dimensions.
This repository contains the material for solving project 5 in FYS4150. In this project
we use three different numerical methods for solving the dimensionless diffusion equation. The three methods
are the forward euler, the backward euler and the Crank-Nicolson. The forward euler is also extended
to solve the diffusion equation in two dimensions.
After that we solve a more practical problem where we look at heat-distribution
in the litosphere as a result of radioactive matter.
## Structure of the repository
In the plots folder you will find plots produced with the various scripts in the repository. THE
plots are named according to the problem they solve and which parameters are used. These can be used as benchmarks together with the tables in the report. In the juyter notebook 5g.ipynb
we use the function for solving the diffusion equation with heat source for solving part
g of the project. This is to simulate the temperature in the lithosphere. In the
file diffusion_solvers.py various methods are found for solving the diffusion equation.
The remaining scripts contains code for solving the part of the project corresponding to the name of the file. 
## Testing
To run test on the library, run the following command in the terminal:
```python
C:... > pytest diffusion_solvers.py
```
Make sure to be in the correct folder.
