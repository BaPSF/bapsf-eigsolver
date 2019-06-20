# Example: Drift wave with temperature fluctuations:
# -------------------------------

import eigsolver_4var

metric = 'cyl'   # choose cylindrical ('cyl') or slab ('cart') geometry
equation = eigsolver_4var.SymbolicEq(metric) # Derive the eigenvalue equation in symbolic form

p = eigsolver_4var.PhysParams(Nr=100, np=3, tp='LAPD_nonrotating', mtheta=20.)  # define a set of physical
                      # parameters for the problem (size, profiles, etc)

esolver = eigsolver_4var.EigSolve(equation, p)  # Solve the eigenvalue problem
eigsolver_4var.plot_omega(esolver,interactive=True)  # Plot profiles and eigenmodes

# Result:
# Fastest growing mode: omega= (-0.0335949297514+0.00990164880834j)