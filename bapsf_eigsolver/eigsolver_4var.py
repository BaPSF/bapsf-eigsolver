#!/usr/bin/env python
#
# Derive the analytic equation for radial eigenmodes from BOUT equations
# and solve it for eigenvalues/eigenvectors
#
#

"""
Example: Drift wave with temperature fluctuations:
-------------------------------


import eigsolver_4var

metric = 'cyl'   # choose cylindrical ('cyl') or slab ('cart') geometry
equation = eigsolver_4var.SymbolicEq(metric) # Derive the eigenvalue equation in symbolic form

p = eigsolver_4var.PhysParams(Nr=100, np=3, tp='LAPD_nonrotating', mtheta=20.)  # define a set of physical
                      # parameters for the problem (size, profiles, etc)

esolver = eigsolver_4var.EigSolve(equation, p)  # Solve the eigenvalue problem
eigsolver_4var.plot_omega(esolver)  # Plot profiles and eigenmodes

# Result:
# Fastest growing mode: omega= (-0.0656901535535+0.0318964836453j)


Example: Drift wave without temperature fluctuations:
-------------------------------


import eigsolver_4var

metric = 'cyl'   # choose cylindrical ('cyl') or slab ('cart') geometry
equation = eigsolver_4var.SymbolicEq(metric, te_fluc=False) # Derive the eigenvalue equation in symbolic form

p = eigsolver_4var.PhysParams(Nr=100, np=3, tp='LAPD_nonrotating', mtheta=20.)  # define a set of physical
                      # parameters for the problem (size, profiles, etc)

esolver = eigsolver_4var.EigSolve(equation, p)  # Solve the eigenvalue problem
eigsolver_4var.plot_omega(esolver)  # Plot profiles and eigenmodes

# Result:
# Fastest growing mode: omega= (-0.0352585012003+0.0104498199244j)


"""



import matplotlib.pyplot as plt
import numpy
import sympy
import sys

from . import profiles
from . import BOUTppmath as bout
from . import misctools as tools


# ==== Grid and physical parameters ==============================

class PhysParams(object):
    """Store physical parameters for the eigenvalue problem"""
    def __init__(self, device="LAPD", **keywords):

        # Default parameters for LAPD device, also the list of all acceptable 
        # arguments for __init__ method
        self.LAPDset = {
            "Nr"     : 100,    # Radial grid size (# of intervals)
            "aa"     : 4.,     # Helium plasma
            "b0"     : 0.04,   # T, magnetic field
            "zeff"   : 1.,     # zeff
            "nu_e"   : 0.,     # electron-ion/neutral collisions, *Omega_CI
            "nu_in"  : 0.,     # ion-neutral collisions, *Omega_CI
            "mtheta" : 10.,    # azimuthal mode number
            "nz"     : 1.,     # axial mode number
            "nr"     : 1.,     # radial wave number, only used for diagnostics
            "rmin_m" : 0.15,   # m
            "rmax_m" : 0.45,   # m
            "Ln_m"   : 0.1,    # m
            "Lz_m"   : 17.,    # m
            "np"     : 0,      # Constant Ln density profile
            "tp"     : 0,      # Constant Te
            "pp"     : 0,      # Constant Er
            "n0"     : 2.5e18, # m^-3
            "te0"    : 5.,     # eV, electron temperature
            "ti0"    : 1.,     # eV, ion temperature (constant), used for mu_ii calculation
            "phi0v"  : 0.,     # V, No background potential
            "mu_fac" : 0.,     # viscosity term factor: 1/0 -- include/neglect ion viscosity
            "k"      : 1.,
            "param"  : {},
            "nparam" : {},
            "tparam" : {},
            "pparam" : {}
            }

        if device.upper() == "LAPD":
            for (key, val) in list(self.LAPDset.items()):
                setattr(self, key, val)
        # TMP
        self.sw = numpy.ones(15)  # switches for different terms in the dispersion relation
        self.param = {}  # additional parameters for profile control 
        self.nparam = {}
        self.tparam = {}
        self.pparam = {}

        for (key, val) in list(keywords.items()):
            # Check that the argument is valid (one from the LAPDset list)
            if key in self.LAPDset:
                # valid key
                setattr(self, key, val)
            else:
                print("Wrong argument name: %s !" % key)

        # Create the grid object, calculate the profiles
        self.grid = profiles.EqGrid(Nr=self.Nr, np=self.np, tp=self.tp, pp=self.pp,
                           param=self.param, 
                           nparam=self.nparam, tparam=self.tparam, pparam=self.pparam)

        self.update_params()
        self.omega0 = 1.e-2+1.e-2j
        

    # Setup omega0 as a property to automatically update the dependent variables
    def get_omega0(self):
        return self._omega0

    def set_omega0(self, omega0):
        self._omega0 = omega0
        self.update_omega0()

    omega0 = property(get_omega0, set_omega0) 


    def update_params(self):
        """Calculate the dependent parameters when the independent values change"""

        ii = 1.j

        p = self  # just to avoid replacing p. everywhere
        p.I = ii  # needed for lambdify function
        p.mu   = p.aa*1836.  # Mi/me
        Cs   = 9.79e3*numpy.sqrt(p.te0/p.aa)   # ion sound velocity, m/s
        p.Om_CI = 9.58e3/p.aa*p.b0*1.e4  # ion gyrofrequency, 1/s
        p.rho_s = Cs/p.Om_CI # [m]

        p.rho_s /= 1.0018830 # slightly adjust rho_s to be exactly the same as the value in
              # BOUT, calculated as rho_s = 1.02*sqrt(AA*Te_x)/ZZ/bmag;
              # Difference: 9.79e3(from Cs)/9.58e3(from Om_CI) / 1.02 = 1.0018830

        p.phi0 = p.phi0v/p.te0 #  normalized potential

        # Normalize the lengths:
        p.rmin = p.rmin_m/p.rho_s
        p.rmax = p.rmax_m/p.rho_s
        p.Ln = p.Ln_m/p.rho_s
        p.Lz = p.Lz_m/p.rho_s
        p.kpar = 2.*numpy.pi/p.Lz  # parallel wave number, rho_s
        p.k = p.kpar*p.nz

        kx = p.nr*2.*numpy.pi/(p.rmax-p.rmin)  # normalized to 1/rho_s
        ky = p.mtheta/(p.rmin+p.rmax)*2. # estimate, normilized to 1/rho_s

        p.omstar = ky/p.Ln   # normalized to Om_CI, dimensionless

        p.omExB0 = 2*p.phi0/p.rmax**2   # normalized to Om_CI, dimensionless. 
                                        # omExB=const in r for phi~r^2


        # radial grid, grid.x=[0,1] expanded to r=[rmin,rmax]
        p.r=p.rmin + p.grid.x*(p.rmax-p.rmin)  
        p.dxdr=1./(p.rmax-p.rmin) # all derivatives in .grid object (p.grid.ni[1] = dni/dx) need to
                                  # be multiplied by p.dxdr to obtain dni/dr
        p.dr=p.grid.h/p.dxdr  # grid step in r


        # Set the nu_ei radial profile (normalized to OmCI)
        #logLam=24.D - alog(sqrt(p.n0*p.ni_arr*1.D-6)/p.Te)
        logLam=24. - numpy.log(numpy.sqrt(p.n0*1.e-6)/p.te0)  # constant logLam - to agree with BOUT formulation
        p.logLam = logLam

        p.nuei_arr=(p.zeff*2.91e-6*(p.n0*p.grid.ni[0]*1.e-6)*logLam 
                    /(p.te0*p.grid.te[0])**1.5 * 0.51 / p.Om_CI)

        # If nu_e set to some value (>0) then use constant collisionality
        # (like in the NLD case)
        # if (p.nu_e gt 0.D) then p.nuei_arr[*] = p.nu_e

        p.spar = p.mu/p.nuei_arr[0]*p.nz**2*p.kpar**2/(kx**2+ky**2)  # normalized to Om_CI, dimensionless

        # Calculate the radial profiles with the correct normalizations
        p.ni  = numpy.zeros(p.grid.ni.shape)
        p.te  = numpy.zeros(p.grid.te.shape)
        p.phi = numpy.zeros(p.grid.phi.shape)

        # transform the derivatives from d/dx to d/dr
        for i in range(p.ni.shape[0]):
            p.ni[i] = p.grid.ni[i]*p.dxdr**i  
        for i in range(p.te.shape[0]):
            p.te[i] = p.grid.te[i]*p.dxdr**i
        for i in range(p.phi.shape[0]):
            p.phi[i] = p.grid.phi[i]*p.dxdr**i
        p.iLn = -p.ni[1]/p.ni[0]

        # Calculate ion-ion viscosity (Braginskii expression eta_1)
        ZZ=1.
        lambda_ii = 23.-numpy.log(ZZ**3*numpy.sqrt(2.*p.n0*1.e-6)/p.ti0**1.5)

        nu_ii = 4.78e-8 * ZZ**4 / numpy.sqrt(p.aa) * p.n0*1.e-6*p.ni[0] * p.logLam / p.ti0**1.5
        nu_ii = 4.78e-8 * ZZ**4 / numpy.sqrt(p.aa) * p.n0*1.e-6*p.ni[0] * lambda_ii / p.ti0**1.5
                # nu_ii in 1/s
        p.nu_ii = nu_ii
 
#  nuiix     = 4.78e-8*pow(ZZ,4.)*Ni_x*lambda_ii/pow(Ti_x, 1.5)/sqrt(AA); // 1/s

        # eta_1: magnetized ion-ion viscosity, normalized using BOUT convention
        p.mu_ii = p.mu_fac * (
                      0.3 * p.ti0/p.te0 * (nu_ii / p.Om_CI)
                             )
        p.mu1_ii = p.mu_ii


        # eta_0: unmagnetized ion-ion viscosity, normalized using BOUT convention
        p.mu_ii = p.mu_fac * (
                      0.96 * p.ti0/p.te0 * p.Om_CI / nu_ii
                             )
        p.mu0_ii = p.mu_ii
        
#(x,y,z)=(10,1,1), mu0_hat=3.107466e+01, mu0=6.254384e-03, Ti=3.000000e-02, Ni=7.745064e-01 
#nuiix=5.695718e+06, mui0_hat(before Te/Ti)=1.614687e-01, l_ei=1.133540e+01, l_ii=5.533709e+00 


    def update_omega0(self):
        """Update the values that depend on omega0"""
        pass


    def pprint(self):
        """Print the values stored"""
        
        # First the independent values
        keys = sorted(self.__dict__.keys())
        print("\nIndependent parameters:")
        for key in keys:
            if key in self.LAPDset:
                print("%10s = %s" % (key, getattr(self, key)))

        print("\nCalculated parameters:")
        excludelist = ["LAPDset", "grid"]
        for key in keys:
            if not ((key in self.LAPDset)or(key in excludelist)):
                val = getattr(self, key)
                if isinstance(val, numpy.ndarray): # array
                    print("%10s = %s, size %s" % (key, "array", val.shape))
                else:
                    print("%10s = %s" % (key, val))

# -----------------------------------------------------------

def blendparams(p1, p2, frac, flog=False):
    """ 
    Calculate an intermediate set of values based on p1,p2 sets and a fraction "frac" 
    Gradual change p1 -> p2
    Returns: p = p1*(1.-frac) + p2*frac
                        or
             p = p1**(1.-frac)*p2**frac
    """

    p = PhysParams(Nr=p1.Nr)  # create the new set of parameters, assume p1.Nr==p2.Nr
    p.frac = frac # save the fraction value used for blending (for plots)

    def linmix(v1,v2,frac):
        return v1*(1.-frac) + v2*frac

    def logmix(v1,v2,frac):
        if not(isinstance(v1, numpy.ndarray)): # called with scalar arguments -- check signs
            if v1*v2<0:
                raise Exception("Can't mix positive and negative values as 'log' fraction!")
        # Assume array arguments have the right sizes
        return numpy.sign(v1)*abs(v1)**(1.-frac)*abs(v2)**frac

    if flog:
        f = logmix
    else:
        f = linmix

    # List of parameters to blend
    ilist = ["aa", "b0", "zeff", "nu_e", "nu_in", "mtheta", "nz", "nr", "rmin_m", "rmax_m",
             "Ln_m", "Lz_m", "n0", "te0", "phi0v", "k"]

    # Calculate the independent parameters as p = f(p1,p2),
    for iparam in ilist: 
        setattr(p, iparam, f(getattr(p1, iparam), getattr(p2, iparam), frac))


    if (p1.np==p2.np)and(p1.tp==p2.tp)and(p1.pp==p2.pp):
        # Profiles don't change, but rmin/rmax might change
        p.grid = profiles.EqGrid(Nr=p1.Nr, np=p1.np, tp=p1.tp, pp=p1.pp, 
                        param=p1.param,
                        nparam=p1.nparam, tparam=p1.tparam, pparam=p1.pparam)  
        #TODO: implement param argument correctly

    else:
        # Assume rmin/rmax don't change, but np/tp/pp do. Mix the ni/te/phi0 profiles
        p.grid  = profiles.EqGrid(Nr=p1.Nr)  # create new grid object  #TMP: add param argument

        for i in range(p.ni.shape[0]):
            p.grid.ni[i]  = f(p1.grid.ni[i], p2.grid.ni[i], frac)
        for i in range(p.te.shape[0]):
            p.grid.te[i]  = f(p1.grid.te[i], p2.grid.te[i], frac)
        for i in range(p.phi.shape[0]):
            p.grid.phi[i] = f(p1.grid.phi[i], p2.grid.phi[i], frac)


    # Update the dependent parameters in p
    p.update_params()
    p.omega0 = 1.e-2+1.e-2j

    return p

#;====================================================================================




# ==== Symbolic derivation of the equation =======================

class SymbolicEq(object):
    """Derive the eigenvalue equation in symbolic form"""

    def __init__(self, metric="cyl", te_fluc=True):
        self.metric = metric
        self.varpack = self._create_symbols(self.metric) # all symbolic variables

        self.symb_eq = self.build_symb_eq(self.varpack, te_fluc)  # linear equations for all variables: N, vpar, phi, (Te if te_fluc==True)
                                                         # stored as list of sympy objects

        self.symb_RHS = [-eq.coeff(self.varpack.omega0) for eq in self.symb_eq] # all terms containing omega0, they appear with a "-" on the RHS
        self.symb_LHS = [sympy.simplify(eq + rhs*self.varpack.omega0) for (eq,rhs) in zip(self.symb_eq, self.symb_RHS)] # all terms without omega0

        self.NVAR = 4  # Number of variables/equations
        self.vars = [self.varpack.N, self.varpack.vpar, self.varpack.phi, self.varpack.Te]


    def _create_symbols(self, metric):
        """Create all symbols/functions"""

        I = complex(0., 1)
        b0 = [0,0,-1]  # unit vector in B direction
                       # Note: in LAPD geometry, the axial field has to be in the negative
                       # direction to be consistent with BOUT
        print("b0=", b0)               

        # Coordinates and time
        r, th, z, t  = sympy.symbols('r theta z t') # Coordinates and time
        x  = [r, th, z]

        # Parameters
        epsilon = sympy.Symbol('epsilon')  # linearization (small) parameter
        k       = sympy.Symbol('k')        # parallel wave vector
        mtheta  = sympy.Symbol('mtheta')   # azimuthal mode number
        mu      = sympy.Symbol('mu')       # mass ratio mi_me
                                           # omega_D = omega0 - m/r dphi0/dr
        omega0  = sympy.Symbol('omega0')   # actual frequency, not Doppler-shifted
        nu_e    = sympy.Symbol('nu_e')     # e-i + e-n collision rate 
        nu_in   = sympy.Symbol('nu_in')    # i-n collision rate
        mu_ii   = sympy.Symbol('mu_ii')    # i-i viscosity

        # Functions (x,t)
        FExp  = sympy.exp(I*mtheta*th + I*k*z - I*omega0*t)  
        
        N0  = sympy.Function('N0')(r)  # equilibrium density
        N   = sympy.Function('N')(r)   # radial part of density perturbation
        fN  = N0 + epsilon*N*N0*FExp   # full density. Note: N is normalized to N0!!! 
                                       # (to avoid large numbers in the FD matrix) 

        phi0  = sympy.Function('phi0')(r)  # equilibrium potential
        phi   = sympy.Function('phi')(r)   # radial part of perturbed potential
        fphi  = phi0 + epsilon*phi*FExp        # full potential
        
        vpar   = sympy.Function('v')(r)  # radial dependence of the parallel electron velocity
        fvpar  = epsilon*vpar*FExp  # parallel electron velocity, perturbed component only

        Te0  = sympy.Function('Te0')(r)  # equilibrium electron temperature
        Te   = sympy.Function('Te')(r)   # radial part of perturbed temperature
        fTe  = Te0 + epsilon*Te*FExp        # full temperature


        # More functions used in several places
        gphi = bout.Grad(fphi, x, metric)  # Grad of full potential
        gperpphi = bout.CrossProd(b0, bout.CrossProd(gphi, b0))  # perpendicular (to b0) part of Grad Phi
        vort = bout.DivPerp(fN*bout.GradPerp(fphi, x, metric), x, metric) # BOUT definition of vorticity
        bxGradN = bout.CrossProd(b0, bout.Grad(fN, x, metric)) # temp variable, used in the vorticity eq.
        vE = bout.CrossProd(b0, gphi)  # ExB drift velocity, equilibrium + perturbation


        # Pack everything in one variable
        fpack = self._pack_symbols(r,th,z,t, epsilon,k,mtheta,mu,omega0,nu_e,nu_in,mu_ii,
                   {'x':x, 'FExp':FExp, 'N0':N0, 'N':N, 'fN':fN, 
                    'phi0':phi0, 'phi':phi, 'fphi':fphi,
                    'vpar':vpar, 'fvpar':fvpar,
                    'Te0':Te0, 'Te':Te, 'fTe':fTe, 'gphi':gphi, 'gperpphi':gperpphi, 'vort':vort, 
                    'vE':vE, 'bxGradN':bxGradN, 
                    'metric':metric})
    
        return fpack


    def _pack_symbols(self, *args):
        """Combine all symbols/function into a dictionary
        Input: first all symbols, then (last) a dictionary of all functions (name:function)
        """
        d = tools.attrdict()
        for arg in args[:-1]:
            d[arg.name] = arg
        for key, item in list(args[-1].items()):
            d[key] = item

        return d


    def build_symb_eq(self, p, te_fluc):
        """Construct the linear equations for all variables (N,vpar,phi,Te)"""
    
        print("Constructing the dispersion relation in symbolic form...")
        print("Phi-equation is modified to exactly reproduce BOUT vorticity equation.")



        # Density equation
        Ni_eq  = sympy.expand(sympy.simplify((
                        p.fN.diff(p.t)                            # dN/dt
                      + bout.DotProd(p.vE, bout.Grad(p.fN, p.x, p.metric))  # vE.Grad(N)
                      + (p.fN*p.fvpar).diff(p.z)                  # div_par Jpar
                                ) / p.FExp)) #/ p.N0 / p.FExp)
        

        # Vparallel equation: parallel electron momentum
        Vpar_eq = sympy.expand(sympy.simplify((
                        p.fvpar.diff(p.t)                            # d vpar/dt
                      + bout.DotProd(p.vE, bout.Grad(p.fvpar, p.x, p.metric))  # vE.grad(vpar)
                      + p.mu*(p.fN*p.Te0).diff(p.z)/p.N0             # mu Grad_par(N Te0) / Ni0
                      + p.mu*1.71*(p.fTe).diff(p.z)                  # 1.71 mu Grad_par(Te)
                      - p.mu*p.fphi.diff(p.z)                        # mu Grad_par(phi)
                      + p.nu_e*p.fvpar                               # nu_e vpar
                                 ) / p.FExp))
    


        # BOUT vorticity equation: Alternative formulation
        print("Using BOUT vorticity equation.")
        Phi_eq = sympy.expand((
               p.vort.diff(p.t) 
             + (p.fN*p.fvpar).diff(p.z)
             + bout.DotProd(p.vE, bout.Grad(p.vort, p.x, p.metric))
             - 0.5*bout.DotProd( p.bxGradN, bout.Grad(bout.DotProd(p.vE,p.vE), p.x, p.metric))
             + p.nu_in*p.vort
             - p.mu_ii*bout.Delp2Perp(p.vort, p.x, p.metric)
                 ) / p.FExp) # / p.N0


        if te_fluc:
            Te_eq = sympy.expand((
                p.fTe.diff(p.t)
                + bout.DotProd(p.vE, bout.Grad(p.fTe, p.x, p.metric))         # vE.grad(Te)
                + 1.71*2./3.* p.fTe * p.fvpar.diff(p.z)             # 1.71 2/3 Te Grad_par(vpar)
                + 2./p.mu * p.nu_e * p.fTe                          # 2 mu nu_e Te
                ) / p.FExp)
        else: # \frac{\partial T_e}{\partial t} = 0
            Te_eq = sympy.expand(p.fTe.diff(p.t) / p.FExp)




        Nonlin_eq = [Ni_eq, Vpar_eq, Phi_eq, Te_eq] # All equations, full form, not linearized
        Lin_eq = [eq.coeff(p.epsilon) for eq in Nonlin_eq] # List of linearized equations


        print("Done")

        return Lin_eq


    def get_Dvar_coeffs(self, eq, f, r):
        """Extract the coefficients (symbolic) at D[f, {r, i=0..2}] in the expression of the form
        eq = c2(r)*f''(r) + c1(r)*f'(r) + c0(r)*f(r)"""
    

        # Coefficient at f''
        c2 = eq.coeff(f.diff(r,2))
        if not c2: c2 = 0  # transform NoneType value to 0 
        eq = sympy.expand(sympy.simplify((eq - c2*f.diff(r,2))))

        # Coefficient at f'
        c1 = eq.coeff(f.diff(r))
        if not c1: c1 = 0
        eq = sympy.expand(sympy.simplify((eq - c1*f.diff(r))))

        # Coefficient at f
        c0 = eq.coeff(f)
        if not c0: c0 = 0


        return (c0,c1,c2)


    def compile_function(self, sf, p, pvalues):
        """Compile a symbolic function (sf) info a callable function
        using the values given. Result: f(r,ni,te,phi0)
        Input:
              sf -- symbolic function (representing the coefficient of the equations)
              p  -- pack of all symbols
              pvalues -- numerical values
        """

        
        # Replace symbols by values given
        ni    = sympy.Symbol("ni")
        te    = sympy.Symbol("te")
        phi   = sympy.Symbol("phi")
        nu_e  = sympy.Symbol("nu_e")

        if sf:
            # Substitute the scalar constants by numerical values
            sf = sf.subs(sympy.I, complex(0.,1))
            sf = sf.subs(p.k, pvalues.k)
            sf = sf.subs(p.nu_in, pvalues.nu_in)
            sf = sf.subs(p.mu, pvalues.mu)
            sf = sf.subs(p.mtheta, pvalues.mtheta)

            # Substitute functions(r) and their derivatives by simple names suitable for 
            # further substitution by a vector

            sf = sf.subs(p.N0.diff(p.r,2),   pvalues.n0*sympy.Symbol("ni[2]"))
            sf = sf.subs(p.N0.diff(p.r),     pvalues.n0*sympy.Symbol("ni[1]"))
            sf = sf.subs(p.N0,               pvalues.n0*sympy.Symbol("ni[0]"))
            sf = sf.subs(p.phi0.diff(p.r,3), pvalues.phi0*sympy.Symbol("phi[3]"))
            sf = sf.subs(p.phi0.diff(p.r,2), pvalues.phi0*sympy.Symbol("phi[2]"))
            sf = sf.subs(p.phi0.diff(p.r),   pvalues.phi0*sympy.Symbol("phi[1]"))
            sf = sf.subs(p.phi0,             pvalues.phi0*sympy.Symbol("phi[0]"))
            sf = sf.subs(p.Te0.diff(p.r,2),  sympy.Symbol("te[2]"))
            sf = sf.subs(p.Te0.diff(p.r,1),  sympy.Symbol("te[1]"))
            sf = sf.subs(p.Te0,              sympy.Symbol("te[0]"))


        # Ugly hack: adding a dummy variable that will ensure that the result of the function is
        # a vector (not scalar!), for any expression sf (for example, for sf=0). 
        # The compiled function should be called with an extra argument "dummyvec" with 0 values. 
        dv = sympy.Symbol("dummyvec") 
        sf = sf + dv
        

        f = sympy.lambdify((p.r, 
                            ni, te, phi, nu_e, p.mu_ii, dv), sf, pvalues)
        return f


    def apply_params(self, pvalues):
        """Compile the symbolic functions (coefficients) into callable functions
        using the values from "pvalues" object
        Construct the arrays LHS/RHS of the form [i_eq, i_var, i_order], with
        indices i_eq -- equation index, i_var -- variable index (N,vpar,phi,Te), i_order -- derivative index (0,1,2)
        Elements of array: callable functions f(r,ni,te,phi,nu_e,mu_ii,dummyvec)
        """

        # outermost index goes first in LHS_fcoeff[i][j][k]
        LHS_fcoeff = [[[0 for i in range(3)]
                              for j in range(self.NVAR)] 
                                  for k in range(self.NVAR)]
                                  
        RHS_fcoeff = [[[0 for i in range(3)]
                              for j in range(self.NVAR)] 
                                  for k in range(self.NVAR)]
                                  

        for i_eq in range(self.NVAR):
            for i_var in range(self.NVAR):
                # Get all coefficients at f'', f', f

                coeffs = self.get_Dvar_coeffs(self.symb_LHS[i_eq], # linear equation
                                              self.vars[i_var],    # variable (N,vpar,phi,Te)
                                              self.varpack.r)      # radial variable -- symbolic

                for i_order in range(3):
                    LHS_fcoeff[i_eq][i_var][i_order] = \
                           self.compile_function(coeffs[i_order], self.varpack, pvalues)


                coeffs = self.get_Dvar_coeffs(self.symb_RHS[i_eq], # linear equation
                                              self.vars[i_var],    # variable (N,vpar,phi,Te)
                                              self.varpack.r)      # radial variable -- symbolic
                    
                for i_order in range(3):
                    RHS_fcoeff[i_eq][i_var][i_order] = \
                           self.compile_function(coeffs[i_order], self.varpack, pvalues)



        return (LHS_fcoeff,RHS_fcoeff)


# ==== Find the eigenvalue of the equation  =======================

class EigSolve(object):
    """Find the eigenvalue of the equation LHS = omega0 RHS"""

    def __init__(self, equation, pvalues, sortby="gamma_asc"):

        self.Nr  = pvalues.Nr
        self.NVAR = equation.NVAR
        self.NTOT = self.Nr*self.NVAR

        self.pvalues   = pvalues
        self.equation  = equation

        self.LHS_fcoeff, self.RHS_fcoeff = self.equation.apply_params(self.pvalues) 
        # set the physical parameters and compile functions f(r,ni,te,phi)
        # Construct the arrays LHS/RHS of the form [i_eq, i_var, i_order], with
        # indices i_eq -- equation index, i_var -- variable index (N,vpar,phi,Te), i_order -- derivative index (0,1,2)

        self.fdiff_matrix(sortby)  # Discretize and solve the eigenvalue problem


    def i_lkp(self, ir, iv):
        """Lookup index in the FD matrix: ir is the radial index, iv is the variable/equation index"""
        return ir*self.NVAR + iv   # 0..Nr*NVAR-1


    def fdiff_matrix(self, sortby):
        """Construct the finite difference matrix of the equations"""

        r    = self.pvalues.r  # radial grid, including end points
        ni   = self.pvalues.ni
        te   = self.pvalues.te
        phi  = self.pvalues.phi
        nu_e = self.pvalues.nuei_arr
        mu_ii= self.pvalues.mu_ii
        dv = numpy.zeros(self.Nr)  # the dummyvec argument for compiled functions

        self.MLHS = numpy.zeros((self.NTOT,self.NTOT), complex)
        self.MRHS = numpy.zeros((self.NTOT,self.NTOT), complex)

        print("Constructing the finite differences matrix...")
        for i_eq in range(self.NVAR):
            for i_var in range(self.NVAR):
                for ir in range(1,self.Nr-1):

                    self.MLHS[self.i_lkp(ir,i_eq), self.i_lkp(ir,i_var)] = (
                        -2*self.LHS_fcoeff[i_eq][i_var][2](r,ni,te,phi,nu_e,mu_ii,dv) 
                         + self.LHS_fcoeff[i_eq][i_var][0](r,ni,te,phi,nu_e,mu_ii,dv)*self.pvalues.dr**2 
                                                                         )[ir]
                    self.MRHS[self.i_lkp(ir,i_eq), self.i_lkp(ir,i_var)] = (
                        -2*self.RHS_fcoeff[i_eq][i_var][2](r,ni,te,phi,nu_e,mu_ii,dv) 
                         + self.RHS_fcoeff[i_eq][i_var][0](r,ni,te,phi,nu_e,mu_ii,dv)*self.pvalues.dr**2 
                                                                         )[ir]

                    self.MLHS[self.i_lkp(ir,i_eq), self.i_lkp(ir+1,i_var)] = (
                           self.LHS_fcoeff[i_eq][i_var][2](r,ni,te,phi,nu_e,mu_ii,dv) 
                         + self.LHS_fcoeff[i_eq][i_var][1](r,ni,te,phi,nu_e,mu_ii,dv)*self.pvalues.dr*0.5 
                                                                           )[ir]
                    self.MRHS[self.i_lkp(ir,i_eq), self.i_lkp(ir+1,i_var)] = (
                           self.RHS_fcoeff[i_eq][i_var][2](r,ni,te,phi,nu_e,mu_ii,dv) 
                         + self.RHS_fcoeff[i_eq][i_var][1](r,ni,te,phi,nu_e,mu_ii,dv)*self.pvalues.dr*0.5 
                                                                           )[ir]

                    self.MLHS[self.i_lkp(ir,i_eq), self.i_lkp(ir-1,i_var)] = (
                           self.LHS_fcoeff[i_eq][i_var][2](r,ni,te,phi,nu_e,mu_ii,dv) 
                         - self.LHS_fcoeff[i_eq][i_var][1](r,ni,te,phi,nu_e,mu_ii,dv)*self.pvalues.dr*0.5 
                                                                           )[ir]
                    self.MRHS[self.i_lkp(ir,i_eq), self.i_lkp(ir-1,i_var)] = (
                           self.RHS_fcoeff[i_eq][i_var][2](r,ni,te,phi,nu_e,mu_ii,dv) 
                         - self.RHS_fcoeff[i_eq][i_var][1](r,ni,te,phi,nu_e,mu_ii,dv)*self.pvalues.dr*0.5 
                                                                           )[ir]


        # Boundary conditions: zero values at r=rmin,rmax for all functions
        for i_var in range(self.NVAR):
            ir = 0
#            self.MLHS[:,self.i_lkp(ir,i_var)] = 0
#            self.MRHS[:,self.i_lkp(ir,i_var)] = 0
            self.MLHS[self.i_lkp(ir,i_var), self.i_lkp(ir,i_var)] = 1
            self.MRHS[self.i_lkp(ir,i_var), self.i_lkp(ir,i_var)] = 1
            ir = self.Nr-1
#            self.MLHS[:,self.i_lkp(ir,i_var)] = 0
#            self.MRHS[:,self.i_lkp(ir,i_var)] = 0
            self.MLHS[self.i_lkp(ir,i_var), self.i_lkp(ir,i_var)] = 1
            self.MRHS[self.i_lkp(ir,i_var), self.i_lkp(ir,i_var)] = 1

                
        print("Solving the linear system...")
        self.MTOT = numpy.dot(numpy.linalg.inv(self.MRHS), self.MLHS)
        print("Done")

#        from misctools import ppmatrix
#        print "Re(MTOT):"
#        ppmatrix(self.MTOT[3:-3,3:-3].real,digits=2)
#        print "Im(MTOT):"
#        ppmatrix(self.MTOT[3:-3,3:-3].imag,digits=2)


        from numpy.linalg.linalg import eig
        self.alleigval, self.alleigvec = eig(self.MTOT)

        # Sort all eigenvalues/vectors by the growth rate
        s_index = list(range(self.Nr*self.NVAR))
        vv = list(zip(self.alleigval, s_index))


        # Sorting with lambda:
        #vv_sorted = sorted(vv, lambda x, y: int(sign(x[0].imag-y[0].imag))) # sort by imag


        # Set of functions for sorting the eigenvalues -- more flexible than lambda sorting
        def fc_gamma_asc(x,y):
            # sort by gamma, ascending
            return int(numpy.sign(x[0].imag-y[0].imag))
        def fc_abs_des(x,y):
            # sort by abs of the eigenvalue, descending
            return int(numpy.sign(abs(y[0])-abs(x[0])))
        def fc_absomega_asc(x,y):
            # sort by abs(omega), ascending, only growing modes
            # exclude omega=1 (values due to BC at matrix corners)
            if (x[0].imag < 0):
                return -1
            if (y[0].imag < 0):
                return 1

            if abs((x[0].real - 1))<1.e-10:
                return -1
            elif abs((y[0].real - 1))<1.e-10:
                return 1
            else:
                return int(numpy.sign(abs(x[0].real)-abs(y[0].real)))
                
        def cmp_to_key(mycmp):
            # 'Convert a cmp= function into a key= function'
            class K:
                def __init__(self, obj, *args):
                    self.obj = obj

                def __lt__(self, other):
                    return mycmp(self.obj, other.obj) < 0

                def __gt__(self, other):
                    return mycmp(self.obj, other.obj) > 0

                def __eq__(self, other):
                    return mycmp(self.obj, other.obj) == 0

                def __le__(self, other):
                    return mycmp(self.obj, other.obj) <= 0

                def __ge__(self, other):
                    return mycmp(self.obj, other.obj) >= 0

                def __ne__(self, other):
                    return mycmp(self.obj, other.obj) != 0
            return K

        try:
            fsort = {"gamma_asc"    : fc_gamma_asc,
                     "abs_des"      : fc_abs_des,
                     "absomega_asc" : fc_absomega_asc}[sortby]
            vv_sorted = sorted(vv, key=cmp_to_key(fsort))  # sort the eigenvalues according to sortby parameter
        except KeyError:
            print("Error: Wrong sort parameter!")
            vv_sorted = vv  # don't sort if sortby is wrong


        s_index = [item[1] for item in vv_sorted]  # s_index is the index in the sorted list
    

        self.alleigval, self.alleigvec = self.alleigval[s_index], self.alleigvec[:,s_index] 
        self.pvalues.omega0 = self.alleigval[-1]
        print("Fastest growing mode: omega=", self.alleigval[-1])


        self.eigN    = self.alleigvec[list(range(0,self.Nr*self.NVAR,  self.NVAR)),:] # eigN[ir, eigpos]
        self.eigVpar = self.alleigvec[list(range(1,self.Nr*self.NVAR+1,self.NVAR)),:]
        self.eigPhi  = self.alleigvec[list(range(2,self.Nr*self.NVAR+2,self.NVAR)),:]
        self.eigTe   = self.alleigvec[list(range(3,self.Nr*self.NVAR+3,self.NVAR)),:]

        self.crossphase_ni = self.ni_phi_phase()
        self.crossphase_te = self.te_phi_phase()
        self.avgphase_ni = numpy.sqrt((self.crossphase_ni[1:-1,-1]**2).mean())
        self.avgphase_te = numpy.sqrt((self.crossphase_te[1:-1,-1]**2).mean())


        return self.alleigval[-1]


    def ni_phi_phase(self):
        # Calculate cross phase between ni and phi

        def atan2vec(yv,xv):
            return numpy.array([numpy.arctan2(y,x) for (x,y) in zip(xv,yv)])

        val = self.eigN/self.eigPhi

        phase = numpy.zeros(val.shape)
        for eig in range(self.Nr*self.NVAR):
            phase[:,eig] = atan2vec(val[:,eig].imag, val[:,eig].real)
        
        return phase

    def te_phi_phase(self):
        # Calculate cross phase between ni and phi

        def atan2vec(yv,xv):
            return numpy.array([numpy.arctan2(y,x) for (x,y) in zip(xv,yv)])

        val = self.eigTe/self.eigPhi

        phase = numpy.zeros(val.shape)
        for eig in range(self.Nr*self.NVAR):
            phase[:,eig] = atan2vec(val[:,eig].imag, val[:,eig].real)
        
        return phase


# -----------------------------------------------------------
def plot_omega(esolver, ommin=None, ommax=None, interactive=False, pos=-1):
    """Plot the eigenvalues (omega0) on the complex plane, 
    and eigenfunctions for max(Im(omega))"""


    if (not ommin) or (not ommax):
        # Plot the region around the fastest growing modes
        nmodes = 5
        ommin = esolver.alleigval[-nmodes]
        ommax = esolver.alleigval[-1]
        dom = ommax - ommin
        ommin -= dom*0.1
        ommax += dom*0.1

    omre = [min(ommin.real, ommax.real), max(ommin.real, ommax.real)] 
    omim = [min(ommin.imag, ommax.imag), max(ommin.imag, ommax.imag)] 

    import matplotlib.font_manager

    plt.ioff()  # turn off matplotlib interactive regime for faster plotting

    fig = plt.figure(1, figsize=(10,10))
    fig.clf()

    # subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95) # reduce subplot margins
    prop = matplotlib.font_manager.FontProperties(size=10) # reduce legend font size

    # -----------------------------------------
    # Plot the equilibrium profiles
    a=plt.subplot(2,2,1)

    def plot_scale(a, x, y, scale, label):
        # Force the profile to 0 if the corresponding scale is 0
        if abs(scale)>1.e-10:
            a.plot(x, y, label=label)
        else:
            a.plot(x, y*0, label=label)
            
    plot_scale(a, esolver.pvalues.grid.x, esolver.pvalues.grid.ni[0],
                        esolver.pvalues.n0, r'$N_i/$%.2e$m^{-3}$' % esolver.pvalues.n0)
    plot_scale(a, esolver.pvalues.grid.x, esolver.pvalues.grid.te[0],
                        esolver.pvalues.te0, r'$T_e/$%.2feV' % esolver.pvalues.te0)
    plot_scale(a, esolver.pvalues.grid.x, esolver.pvalues.grid.phi[0], 
                        esolver.pvalues.phi0v, r'$\phi_0/$%.2fV' % esolver.pvalues.phi0v)
    a.legend(prop=prop)
    plt.xlabel('x')


    # -----------------------------------------
    def plot_eigenvalues(pos):
        # Plot the eigenvalues, highlight the value at position=pos
        a=plt.subplot(2,2,2)
        a.clear()
        a.plot(esolver.alleigval.real, esolver.alleigval.imag, 'o')
        a.plot([esolver.alleigval[pos].real], [esolver.alleigval[pos].imag], 'ro', markersize=10)
        a.axis([omre[0], omre[1], omim[0], omim[1]])
        plt.xlabel(r'$Re(\omega/\Omega_{ci})$')
        plt.ylabel(r'$Im(\omega/\Omega_{ci})$')


    # -----------------------------------------
    def plot_eigenvectors(pos, type='cart'):
        # Plot the solution eigenvectors: phi and ni
        # type=cart:  plot Re and Im
        # type=polar: plot amplitude and phase

        def atan2vec(yv,xv):
            return numpy.array([numpy.arctan2(y,x) for (x,y) in zip(xv,yv)])

        if type == 'cart':
            vp1 = esolver.eigPhi[:,pos].real
            vp1label = r'$Re(\phi)$'
            vp2 = esolver.eigPhi[:,pos].imag
            vp2label = r'$Im(\phi)$'
            vn1 = esolver.eigN[:,pos].real
            vn1label = r'$Re(Ni)$'
            vn2 = esolver.eigN[:,pos].imag
            vn2label = r'$Im(Ni)$'

        if type == 'polar':
            vp1 = abs(esolver.eigPhi[:,pos])
            vp1label = r'$Abs(\phi)$'
            vp2 = atan2vec(esolver.eigPhi[:,pos].imag, esolver.eigPhi[:,pos].real)
            vp2label = r'$Phase(\phi)$'
            vn1 = abs(esolver.eigN[:,pos])
            vn1label = r'$Abs(Ni)$'
            vn2 = atan2vec(esolver.eigN[:,pos].imag, esolver.eigN[:,pos].real)
            vn2label = r'$Phase(Ni)$'

        # Phi
        a=plt.subplot(2,2,3)
        a.clear()
        a.plot(esolver.pvalues.grid.x, vp1, label=vp1label)
        a.plot(esolver.pvalues.grid.x, vp2, label=vp2label)
        a.legend(prop=prop)
        plt.xlabel('x')
        # Ni
        a=plt.subplot(2,2,4)
        a.clear()
        a.plot(esolver.pvalues.grid.x, vn1, label=vn1label)
        a.plot(esolver.pvalues.grid.x, vn2, label=vn2label)
        a.legend(prop=prop)
        plt.xlabel('x')


    plot_eigenvalues(pos)
    plot_eigenvectors(pos, type='cart')

    plt.draw(); fig.canvas.flush_events()
    plt.show()
    plt.ion()


    if interactive:
        # Choose the eigenvalue and plot the corresponding eigenfunction
        # Repeat the loop until clicked on the same value twice

        print("Click on the eigenvalue to plot the solution. To exit: choose the same value twice.")
        pos_prev = len(esolver.alleigval)-1
        pos_click = 0
        while pos_prev != pos_click:

            pos_prev = pos_click
            bmi = BlockingMouseInput()
            clicks = bmi(fig, 1, verbose=True)
            om0 = complex(clicks[0][0],clicks[0][1])

            # Find which eigenvalue was clicked
            pos_click = abs(esolver.alleigval - om0).argmin() # position of the closest to om0 value
            print("Clicked: omega=", esolver.alleigval[pos_click])
    
            plot_eigenvalues(pos_click)
            plot_eigenvectors(pos_click)

            plt.draw(); fig.canvas.flush_events()




# -----------------------------------------------------------
def trace_root(equation, p1, p2, nfrac=100, accumulate=1., 
               flog=False, max=False, 
               plotparam=None, execfunc=None, noplots=False):
    """Trace one root from p1 set of paramters to p2"""
    import time

    scanres = [] # store the parameters and the solution at each step

    om = p1.omega0
    omega_last = om  # used for max growth rate check

    start_trace = time.clock()

    for i in range(nfrac):
        
        frac=(i/(nfrac-1.))**accumulate

        print("Tracing: frac=%10.3e   (%d/%d)" % (frac, i, nfrac))
        
        p = blendparams(p1,p2,frac, flog=flog)
        esolver = EigSolve(equation, p)
        if not noplots:
            plot_omega(esolver)

        scanres.append(p) # save the parameters and the solution at each step
        if execfunc: execfunc(p)

        # stop if looking for the max growth rate
        if (max and (p.omega0.imag < omega_last.imag)):
            print('Prev. step growth rate (gamma/OmCI): ' % omega_last.imag)
            print('Last  step growth rate (gamma/OmCI): ' % p.omega0.imag)
            print('Max growth rate found, exiting...')
            break 
        omega_last = p.omega0
        om = p.omega0

        p.alleigval  = esolver.alleigval
        p.eigN    = esolver.eigN[:, -6:]
        p.eigVpar = esolver.eigVpar[:, -6:]
        p.eigPhi  = esolver.eigPhi[:, -6:]
        p.eigTe   = esolver.eigTe[:, -6:]
        p.crossphase_ni = esolver.crossphase_ni[:, -6:]
        p.crossphase_te = esolver.crossphase_te[:, -6:]
        p.avgphase_ni = esolver.avgphase_ni
        p.avgphase_te = esolver.avgphase_te

        if (not noplots) and plotparam and (i % 5 == 4):
            # Update parameter scan plot
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')

            fig = plt.figure(2, figsize=(7,7))
            fig.clf()
            p_param = numpy.array([getattr(p,plotparam) for p in scanres])
            p_omega = numpy.array([p.omega0.real for p in scanres])
            p_gamma = numpy.array([p.omega0.imag for p in scanres])
            a=plt.subplot(1,2,1)
            a.plot(p_param, p_omega, 'o-', label=r'\omega')
            plt.xlabel(plotparam)
            a.legend()
            a=plt.subplot(1,2,2)
            a.plot(p_param, p_gamma, 'o-', label=r'\gamma')
            plt.xlabel(plotparam)
            a.legend()
            plt.draw()
            fig.show()
            

    p2.omega0 = p.omega0
    p2.update_params()

    end_trace = time.clock()
    print("Total trace time: %.2gs" % (end_trace-start_trace))

    if plotparam:
        # Save the scan data in ascii file
        p_param = numpy.array([getattr(p,plotparam) for p in scanres])
        p_omega = numpy.array([p.omega0.real for p in scanres])
        p_gamma = numpy.array([p.omega0.imag for p in scanres])
        numpy.savetxt('scan_%s.txt' % plotparam, numpy.transpose(numpy.array([p_param, p_omega, p_gamma])))

        # Find the fastest growing mode
        a=p_gamma.argmax()
        print("Fastest mode: %s=%.3e,  omega=%.3e,  gamma=%.3e" % (plotparam, 
                                                  p_param[a], p_omega[a], p_gamma[a]))

    return scanres

# -----------------------------------------------------------
def save_scan(scanres, fname='trace_scan.dat'):
    """ Save the results of a scan (from trace_root function) """
    import pickle

    f = open(fname, 'w')
    keys = list(scanres[0].LAPDset.keys()) + ['omega0','ni','te', 
                 'phi','rho_s','kpar','omExB0','rmin','rmax',
                 'spar','Om_CI','omstar', 'phi0',
                 'alleigval', 'eigN', 'eigVpar', 'eigPhi', 'eigTe',
                 'crossphase_ni', 'crossphase_te', 'avgphase_ni', 'avgphase_te'] # choose the data to save
    data = {}  # dictionary of vector values, data only (no functions) -- pickle-able object

    for key in keys:
        val = numpy.array([getattr(p,key) for p in scanres])  # extract the data from scanres
        data[key] = val

    pickle.dump(data, f)
        
    f.close()

# -----------------------------------------------------------
def load_scan(fname='trace_scan.dat'):
    """ Load the results of a scan """
    import pickle

    f = open(fname, 'r')
    data = pickle.load(f)
    f.close()
    return data # list of dictionaries, data only (no functions)

# -----------------------------------------------------------
def combine_scans(*d, **darg):
    """ Combine several scans and sort them according to the sortparam value 
        Syntax: combine_scans(d1,d2,..., sortparam='mtheta', finclude=lambda x: x>10.)
        If finclude is specified, only finclude(x)=True elements will be included (only
        used when sortparam is specified)
    """
    
    sortparam = None
    finclude = None
    for key, val in list(darg.items()):
        if key == 'sortparam':
            sortparam = val
        if key == 'finclude':
            finclude = val

    dnew = d[0]
    for key in list(d[0].keys()):
        for darg in d[1:]:
            dnew[key] = numpy.append(dnew[key], darg[key])

    # Sort if asked
    if sortparam:
        idx = numpy.argsort(dnew[sortparam])
        for key in list(dnew.keys()):
            dnew[key] = dnew[key][idx]

    # Filter elements if asked
    if finclude and sortparam:
        idx = finclude(dnew[sortparam])  # get the indices of all elements that 
                                         # satisfy the finclude condition
        for key in list(dnew.keys()):
            dnew[key] = dnew[key][idx]

    return dnew

# -----------------------------------------------------------
 
class BlockingMouseInput(object):
    """Class that stops the program execution until mouse click(s)"""

    callback = None
    verbose = False
    def __call__(self, fig, n=1, verbose=False):
        """Blocking call to retrieve n coordinates through mouse clicks."""
        import time
        
        assert isinstance(n, int), "Requires an integer argument"

        # Ensure that the current figure is shown
        fig.show()
        # connect the click events to the on_click function call
        self.callback = fig.canvas.mpl_connect('button_press_event',
                                               self.on_click)

        # initialize the list of click coordinates
        self.clicks = []

        self.verbose = verbose

        # wait for n clicks
        print("Waiting for mouse click...", end=' ')
        sys.stdout.flush()
        counter = 0
        while len(self.clicks) < n:
            fig.canvas.flush_events()
            # rest for a moment
            time.sleep(0.01)
#        print "\r"+" "*40+"\n"


        # All done! Disconnect the event and return what we have
        fig.canvas.mpl_disconnect(self.callback)
        self.callback = None
        return self.clicks

    def on_click(self, event):
        """Event handler to process mouse click"""

        # if it's a valid click, append the coordinates to the list
        if event.inaxes:
            self.clicks.append((event.xdata, event.ydata))
            if self.verbose:
                print("\rInput %i: %f, %f" % (len(self.clicks),
                                    event.xdata, event.ydata))

 
# -----------------------------------------------------------



# -----------------------------------------------------------



if __name__ == '__main__':
    pass

#    import profile
#    print "Running the profiler..."
#    profile.run('profrun_simple()', 'prof')
#
#    import pstats
#    p = pstats.Stats('prof')
#    p.sort_stats('time').print_stats(50)
