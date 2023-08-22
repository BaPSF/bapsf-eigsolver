#!/usr/bin/env python3

import importlib

from bapsf_eigsolver import eigsolver
from bapsf_eigsolver import sym_eqns


importlib.reload(eigsolver)

# choose cylindrical ('cyl') or slab ('cart') geometry
metric = 'cyl'

# Derive the eigenvalue equation in symbolic form
equation = sym_eqns.braginskii.Brag(metric)
# equation = eigsolver.SymbolicEq(metric)

# define a set of physical
# * parameters for the problem (size, profiles, etc)
p = eigsolver.PhysParams(Nr=100, np=3, m_theta=27.625)

# Solve the eigenvalue problem
esolver = eigsolver.EigSolve(equation, p)

# Plot profiles and eigenmodes
eigsolver.plot_omega(esolver)
