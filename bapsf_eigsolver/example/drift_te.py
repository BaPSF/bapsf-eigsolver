
import importlib

from bapsf_eigsolver import eigsolver_te

importlib.reload(eigsolver_te)

metric = 'cyl'   # choose cylindrical ('cyl') or slab ('cart') geometry
equation = eigsolver_te.SymbolicEq(metric) # Derive the eigenvalue equation in symbolic form

p = eigsolver_te.PhysParams(Nr=100, np=3, tp='LAPD_nonrotating', mtheta=20.)  # define a set of physical
                      # parameters for the problem (size, profiles, etc)

esolver = eigsolver_te.EigSolve(equation, p)  # Solve the eigenvalue problem
eigsolver_te.plot_omega(esolver)  # Plot profiles and eigenmodes

# Result:
# Fastest growing mode: omega= (-0.0656901535535+0.0318964836453j)
