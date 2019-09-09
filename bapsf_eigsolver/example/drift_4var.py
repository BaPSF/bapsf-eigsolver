#!/usr/bin/env python3

import importlib
from bapsf_eigsolver import eigsolver
from bapsf_eigsolver import sym_eqns

importlib.reload(eigsolver)

metric = 'cyl'   # choose cylindrical ('cyl') or slab ('cart') geometry
# here you can pass te_fluc = False, otherwise the
# default value of True is used
equation = sym_eqns.brag_4var.BragFourVar(metric)

p = eigsolver.PhysParams(Nr=100, np=3, tp='LAPD_nonrotating', m_theta=20.)
# define a set of physical parameters for the problem (size, profiles, etc)

esolver = eigsolver.EigSolve(equation, p)  # Solve the eigenvalue problem
eigsolver.plot_omega(esolver)  # Plot profiles and eigenmodes

# Result:
# Fastest growing mode: omega= (-0.0656901535535+0.0318964836453j)
