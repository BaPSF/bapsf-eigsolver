from bapsf_eigsolver import eigsolver
from bapsf_eigsolver.misctools import attrdict

metric = 'cyl'
equation = eigsolver.SymbolicEq(metric) # Derive the eigenvalue equation in symbolic form

w = 20.
mtheta=1.
p = eigsolver.PhysParams(Nr=100, rmin_m=0.001, rmax_m=0.45,
                          np=5, tp=0, pp=4,
                          param=attrdict(w=w, x0=0.5, n2=0.9, ra=0.001, rb=0.45),
                          phi0v=50., nz=0., mtheta=mtheta)

esolver = eigsolver.EigSolve(equation, p)  # Solve the eigenvalue problem
eigsolver.plot_omega(esolver)  # Plot profiles and eigenmodes

# Fastest growing mode: omega= (0.0123935824817+0.00236358432211j)