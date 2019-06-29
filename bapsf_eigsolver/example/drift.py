

import eigsolver
import imp
imp.reload(eigsolver)

metric = 'cyl'   # choose cylindrical ('cyl') or slab ('cart') geometry
equation = eigsolver.SymbolicEq(metric) # Derive the eigenvalue equation in symbolic form

p = eigsolver.PhysParams(Nr=100, np=3, mtheta=27.625)  # define a set of physical
                      # parameters for the problem (size, profiles, etc)

esolver = eigsolver.EigSolve(equation, p)  # Solve the eigenvalue problem

eigsolver.plot_omega(esolver)  # Plot profiles and eigenmodes




