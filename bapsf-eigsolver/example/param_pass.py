# Example: Passing parameters to a profile:
# -------------------------------


import eigsolver_4var
from misctools import attrdict

metric = 'cyl'   # choose cylindrical ('cyl') or slab ('cart') geometry
equation = eigsolver_4var.SymbolicEq(metric) # Derive the eigenvalue equation in symbolic form

p = eigsolver_4var.PhysParams(Nr=100, np=3, pp=4, phi0v=20, pparam=attrdict(ra=0.15,rb=0.45))  # define a set of physical
                      # parameters for the problem (size, profiles, etc)

esolver = eigsolver_4var.EigSolve(equation, p)  # Solve the eigenvalue problem
eigsolver_4var.plot_omega(esolver,interactive=True)  # Plot profiles and eigenmodes

# Result:
#Fastest growing mode: omega= (-0.0817573812354+0.00523481491376j)