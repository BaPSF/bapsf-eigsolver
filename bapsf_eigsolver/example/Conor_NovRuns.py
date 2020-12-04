#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: conorperks
"""

#!/usr/bin/env python3

import importlib

from bapsf_eigsolver import eigsolver

import numpy as np

importlib.reload(eigsolver)

# choose cylindrical ('cyl') or slab ('cart') geometry
metric = 'cyl'

# Derive the eigenvalue equation in symbolic form
equation = eigsolver.SymbolicEq(metric)

# Number of mode number to consider
N = 100

# Poloidal mode numbers
m_theta_set=np.linspace(0,80, N)
#m_theta_set = 20

# Fastest growing eigenvalues for each m_theta
eig_imag = np.zeros(N)
eig_real = np.zeros(N)

# Fastest growing eigenvectors for each m_theta
#eig_phi_real = np.zeros([100,N])
#eig_phi_imag = np.zeros([100,N])
#eig_ni_real = np.zeros([100,N])
#eig_ni_imag = np.zeros([100,N])

for i in np.arange(1):
    print(m_theta_set[i])
    
    # Defines physical parameters for the problem
    p = eigsolver.PhysParams(Nr=100, b0 = 0.1, rmin_m=0.22, rmax_m=0.25,
                          np='Conor1', tp='Conor1', pp='Conor1',
                          n0 = 1.5126e18, te0 = 4.2778, phi0v=21.1094,
                          m_theta=m_theta_set[i], nz = 0)

    # Solve the eigenvalue problem
    esolver = eigsolver.EigSolve(equation, p)
    
    # Stores the fastest growing eigenvalue
    eig_imag[i] = esolver.alleigval[-1].imag
    eig_real[i] = esolver.alleigval[-1].real
    
    # Stores the fastest growing eigenvector
    #eig_phi_real[:,i] = esolver.eigPhi[:,-1].real
    #eig_phi_imag[:,i] = esolver.eigPhi[:,-1].imag
    #eig_ni_real[:,i] = esolver.eigN[:,-1].real
    #eig_ni_imag[:,i] = esolver.eigN[:,-1].imag

# Plot profiles and eigenmodes
eigsolver.plot_omega(esolver)










