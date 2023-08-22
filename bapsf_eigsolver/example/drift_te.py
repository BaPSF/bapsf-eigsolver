#!/usr/bin/env python3

import importlib
from bapsf_eigsolver import eigsolver
from bapsf_eigsolver import sym_eqns

importlib.reload(eigsolver)

metric = 'cyl'   # choose cylindrical ('cyl') or slab ('cart') geometry

# Derive the eigenvalue equation in symbolic form
equation = sym_eqns.brag_temperature.BragTemp(metric)

# define a set of physical parameters for the problem (size, profiles, etc)
p = eigsolver.PhysParams(Nr=100, np=3, tp='LAPD_nonrotating', m_theta=20.)

esolver = eigsolver.EigSolve(equation, p)  # Solve the eigenvalue problem
eigsolver.plot_omega(esolver)  # Plot profiles and eigenmodes

# Result:
# Fastest growing mode: omega= (-0.0656901535535+0.0318964836453j)
