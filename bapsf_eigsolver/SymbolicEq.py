#!/usr/bin/env /u/local/apps/python/2.6.1/bin/python
#
# Class that defines symbolic equations

class SymbolicEq(object):
    """Summary of the class SymbolicEq


    Defines the differential equations in symbolic form.
    The class creates the three equations for density (N), parallel electron velocity (v_par) and potential (phi).
    The symbolic variables are initially defined and used to create symbolic equations for these three quanities
    via the Braginskii fluid equations.
    The equations are then linearized, i.e. only terms containing the perturbation factor (epsilon) are
    considered to create the final equations.
    Finally, terms with the frequency (omega0) are saved as LHS terms while all remaining terms are saved as RHS
    terms. The RHS and LHS notation exists to setup the eigenvalue matrix calculation later on.

    
    """

    def __init__(self, metric="cyl", temp_boolean):

        """
        Parameters:
        metric: str ("cyl" or "cart")
            the coordinate system to be used. For LAPD the default is
            cylindrical coordinates or "cyl".  The other option is
            "cart" for Cartesian coordinates.
        """
        # store coordinate system
        self.metric = metric

        # create symbolic variables and functions
        # * metric is passed to BOUT
        self.varpack = self._create_symbols(self.metric)

        # build symbolic equations (linearized eqns)
        # * symb_eq is a list of linearized eqns where
        #   ** symb_eq[0] = N = density
        #   ** symb_eq[1] = v_par = parallel electron velocity
        #   ** symb_eq[2] = phi = potential
        #
        if (temp_boolean):
            self.symb_eq = self.build_symb_eq_temp(self.varpack)
        else:
            self.symb_eq = self.build_symb_eq_temp(self.varpack)

        # build right-hand-side (RHS) and left-hand-side (LHS) symbolic
        # equations
        #
        # self.symb_RHS = [-eq.coeff(self.varpack.omega0)
        #                  for eq in self.symb_eq]
        # self.symb_LHS = [eq + rhs*self.varpack.omega0
        #                  for (eq, rhs) in zip(self.symb_eq, self.symb_RHS)]
        #
        self.symb_RHS = []
        self.symb_LHS = []
        for eq in self.symb_eq:
            rhs = -eq.coeff(self.varpack.omega0)
            self.symb_RHS.append(rhs)
            self.symb_LHS.append(sympy.simplify(eq + (rhs * self.varpack.omega0)))

        # Number of variables/equations
        self.NVAR = 3
        self.vars = [self.varpack.N, self.varpack.v_par, self.varpack.phi]

    def _create_symbols(self, metric):
        """Create all symbols/functions

        Parameters
        ----------
        metric: the coordinate system

        Returns
        -------
        fpack
            a pack of all the variables, symbols and functions defined

        """

         # define imaginary unit "i"
        I = complex(0., 1.)  # this is the imaginary unit "i"

        # define magnetic field vector
        # Note: in LAPD geometry the axial field points in -z (towards main
        #       cathode)
        #
        b0 = [0, 0, -1]
        print("b0=", b0)

        # Coordinates and time
        r, th, z, t = sympy.symbols('r theta z t') # Coordinates and time
        x = [r, th, z]  # position vector

        # -- Parameters --
        # epsilon:  linearization (small) parameter
        # k      :  parallel wave vector
        # m_theta:  azimuthal mode number
        # mu     :  ion to electron mass ratio
        # omega0 :  actual frequency, not Doppler-shifted
        # nu_e   :  electron-ion + electron-neutral collision rate
        # nu_in  :  ion-neutral collision rate
        # mu_ii  :  ion-ion viscosity
        #
        epsilon = sympy.Symbol('epsilon')
        k = sympy.Symbol('k')
        m_theta = sympy.Symbol('m_theta')
        mu = sympy.Symbol('mu')
        omega0 = sympy.Symbol('omega0')
        # omega_D = omega0 - m/r dphi0/dr
        nu_e = sympy.Symbol('nu_e')
        nu_in = sympy.Symbol('nu_in')
        mu_ii = sympy.Symbol('mu_ii')

        # -- Functions (x,t) --
        #
        # we want solution in terms of this exponential --- why?
        Eig_func = sympy.exp(I * (m_theta * th + k * z - omega0 * t))
        
        # setup up density profiles
        # N0     :  equilibrium density
        # N      :  radial part of density perturbation
        # N_total:  full density
        #           * N is normalized to N0!!!
        #           * normalization is to avoid large numbers in
        #             the Finite Differences matrix
        #
        N0 = sympy.Function('N0')(r)
        N = sympy.Function('N')(r)
        N_total = N0 + (epsilon * N * N0 * Eig_func)

        # setup potential profiles
        # phi0     :  equilibrium potential
        # phi      :  radial part of perturbed potential
        # phi_total:  full potential
        #
        phi0 = sympy.Function('phi0')(r)
        phi = sympy.Function('phi')(r)
        phi_total = phi0 + (epsilon * phi * Eig_func)
        
        # setup parallel velocity profiles
        # v_par  :  radial dependence of the parallel electron velocity
        # fv_par :  parallel electron velocity, perturbed component only
        #
        v_par = sympy.Function('v')(r)
        fv_par = epsilon * v_par * Eig_func

        # setup electron temperature profiles
        # Te0:  equilibrium electron temperature
        # Te: radial part of perturbed temperature
        # fTe: full temperature
        Te0 = sympy.Function('Te0')(r)
        Te   = sympy.Function('Te')(r)
        fTe  = Te0 + epsilon*Te*FExp        

        # additional misc functions
        # gphi    :  gradient of full potential
        # gperpphi:  perpendicular (to b0) part of Grad Phi
        # vort    :  BOUT definition of vorticity
        # bxGradN :  temp variable, used in the vorticity eq.
        # vE      :  ExB drift velocity, equilibrium + perturbation
        #
        gphi = bout.Grad(phi_total, x, metric)
        gperpphi = bout.CrossProd(b0, bout.CrossProd(gphi, b0))
        vort = bout.DivPerp(N_total * bout.GradPerp(phi_total, x, metric),
                            x,
                            metric)
        bxGradN = bout.CrossProd(b0, bout.Grad(N_total, x, metric))
        vE = bout.CrossProd(b0, gphi)

        # Pack everything in one variable
        # fpack = self._pack_symbols(r, th, z, t, epsilon, k, m_theta, mu,
        #                            omega0,nu_e,nu_in,mu_ii,
        #            {'x':x, 'Eig_func':Eig_func, 'N0':N0, 'N':N, 'N_total':N_total,
        #             'phi0':phi0, 'phi':phi, 'phi_total':phi_total,
        #             'v_par':v_par, 'fv_par':fv_par,
        #             'Te0':Te0, 'Te':Te, 'fTe':fTe, 'gphi':gphi, 'gperpphi':gperpphi, 'vort':vort,
        #             'vE':vE, 'bxGradN':bxGradN,
        #             'metric':metric})
        fpack = self._pack_symbols(**{
            'r': r,
            'th': th,
            'z': z,
            't': t,
            'epsilon': epsilon,
            'k': k,
            'm_theta': m_theta,
            'mu': mu,
            'omega0': omega0,
            'nu_e': nu_e,
            'nu_in': nu_in,
            'mu_ii': mu_ii,
            'x': x,
            'Eig_func': Eig_func,
            'N0': N0,
            'N': N,
            'N_total': N_total,
            'phi0': phi0,
            'phi': phi,
            'phi_total': phi_total,
            'v_par': v_par,
            'fv_par': fv_par,
            'Te0': Te0,
            'Te': Te,
            'fte': fTe,
            'gphi': gphi,
            'gperpphi': gperpphi,
            'vort': vort,
            'vE': vE,
            'bxGradN': bxGradN,
            'metric': metric,
        })
    
        return fpack


    @staticmethod
    def _pack_symbols(**symbols):
        """Combine all symbols/function into a dictionary
        
        Input: first all symbols, then (last) a dictionary of all functions
            (name:function)
        """
        d = tools.attrdict()
        d.update(symbols)

        return d


    def build_symb_eq(self, p):
        """Construct the linear equations for all variables (N,v_par,phi)

        Parameters
        ----------
        p: the package of all variable and symbols for the equations

        Returns
        ----------
        Lin_eq
            Array of the linearized version of each of the three equations
        """
    
        print("Constructing the dispersion relation in symbolic form...\n"
              "Phi-equation is modified to exactly reproduce BOUT vorticity equation.")

        # Density equation
        # * eqn 4.1 in B. Friedman 2013 dissertation
        #   (https://escholarship.org/uc/item/4799v1k0)
        # * eqn 5.11 in D. Schaffner 2013 dissertation
        #   (https://escholarship.org/uc/item/7hz553m0)
        #
        #  [ d(N_total)/dt
        #    + dot(vE, Grad(N_total))
        #    + d(N_total * fv_par)/dz ] / Eig_func
        #
        # where N_total = density
        #       vE      = ExB drift velocity
        #       fv_par  = perturbed parallel electron velocity
        #       dz      = parallel direction
        #       Eig_func    = solution function
        #
        Ni_eq = sympy.expand(sympy.simplify(
            (p.N_total.diff(p.t)  # dN/dt
             + bout.DotProd(p.vE, bout.Grad(p.N_total, p.x, p.metric))
             + (p.N_total * p.fv_par).diff(p.z))
            / p.Eig_func
        ))  # / p.N0 / p.Eig_func)
        
        # Vparallel equation: parallel electron momentum
        # * eqn 4.2 in B. Friedman 2013 dissertation
        #   (https://escholarship.org/uc/item/4799v1k0)
        # * eqn 5.12 in D. Schaffner 2013 dissertation
        #   (https://escholarship.org/uc/item/7hz553m0)
        #
        #  [ d(fv_par)/dt
        #    + dot(vE, Grad(fv_par))
        #    + mu * d(N_total * Te0)/dz / N0
        #    - mu * d(phi_total)/dz
        #    + nu_e * fv_par ] / Eig_func
        #
        # where N_total   = full density
        #       N0        = equilibrium density
        #       vE        = ExB drift velocity
        #       fv_par    = perturbed parallel electron velocity
        #       Te0       = equilibrium electron temperature
        #       phi_total = full potential
        #       mu        = ion-to-electron mass ratio
        #       nu_e      = electron-ion + electron_neutral collision rate
        #       dz        = parallel direction
        #       Eig_func      = solution function
        #
        V_par_eq = sympy.expand(sympy.simplify(
            (p.fv_par.diff(p.t)
             + bout.DotProd(p.vE, bout.Grad(p.fv_par, p.x, p.metric))
             + p.mu * (p.N_total * p.Te0).diff(p.z) / p.N0
             - p.mu * p.phi_total.diff(p.z)
             + p.nu_e * p.fv_par)
            / p.Eig_func
        ))
    

        # BOUT vorticity equation: Alternative formulation
        # 
        print("Using BOUT vorticity equation.")
        Phi_eq = sympy.expand((
            p.vort.diff(p.t)  # d vort/dt
            + (p.N_total * p.fv_par).diff(p.z)  # d (N.v_par)/dt
            + bout.DotProd(p.vE, bout.Grad(p.vort, p.x, p.metric))  # vE. grad(vort)

            # 0.5* b x grad(N). grad^2(vE)
            - 0.5 * bout.DotProd(p.bxGradN, bout.Grad(bout.DotProd(p.vE, p.vE), p.x, p.metric))

            + p.nu_in*p.vort  # nu_in.vort
            - p.mu_ii*bout.Delp2Perp(p.vort, p.x, p.metric) # mu_ii. grad_perp^2(vort)
        ) / p.Eig_func
        )                              

        # Non-lineqr (full) symbolic equations
        Nonlin_eq = [Ni_eq, V_par_eq, Phi_eq]

        # build linearized symbolic equns
        # * only consider terms with epsilon factor
        # * eqns for N, v_par, and phi are stored in Lin_eq[0], Lin_eq[1], and
        #   Lin_eq[2] respectively; if Te is calculated, stored in Lin_eq[4]
        #
        Lin_eq = [eq.coeff(p.epsilon) for eq in Nonlin_eq]
        print ("P: \n", Lin_eq[2])

        print("Done")

        return Lin_eq



    def build_symb_eq_temp(self, p):
        """Construct the linear equations for all variables (N,v_par,phi)

        Parameters
        ----------
        p: the package of all variable and symbols for the equations

        Returns
        ----------
        Lin_eq
            Array of the linearized version of each of the three equations
        """
    
        print("Constructing the dispersion relation in symbolic form...\n"
              "Phi-equation is modified to exactly reproduce BOUT vorticity equation.")

        # Density equation
        # * eqn 4.1 in B. Friedman 2013 dissertation
        #   (https://escholarship.org/uc/item/4799v1k0)
        # * eqn 5.11 in D. Schaffner 2013 dissertation
        #   (https://escholarship.org/uc/item/7hz553m0)
        #
        #  [ d(N_total)/dt
        #    + dot(vE, Grad(N_total))
        #    + d(N_total * fv_par)/dz ] / Eig_func
        #
        # where N_total = density
        #       vE      = ExB drift velocity
        #       fv_par  = perturbed parallel electron velocity
        #       dz      = parallel direction
        #       Eig_func    = solution function
        #
        Ni_eq = sympy.expand(sympy.simplify(
            (p.N_total.diff(p.t)  # dN/dt
             + bout.DotProd(p.vE, bout.Grad(p.N_total, p.x, p.metric))
             + (p.N_total * p.fv_par).diff(p.z))
            / p.Eig_func
        ))  # / p.N0 / p.Eig_func)
        
        # Vparallel equation: parallel electron momentum
        # * eqn 4.2 in B. Friedman 2013 dissertation
        #   (https://escholarship.org/uc/item/4799v1k0)
        # * eqn 5.12 in D. Schaffner 2013 dissertation
        #   (https://escholarship.org/uc/item/7hz553m0)
        #
        #  [ d(fv_par)/dt
        #    + dot(vE, Grad(fv_par))
        #    + mu * d(N_total * Te0)/dz / N0
        #    + mu * 1.71 * d(fTe)/dz 
        #    - mu * d(phi_total)/dz
        #    + nu_e * fv_par ] / Eig_func
        #
        # where N_total   = full density
        #       N0        = equilibrium density
        #       vE        = ExB drift velocity
        #       fv_par    = perturbed parallel electron velocity
        #       Te0       = equilibrium electron temperature
        #       fTe       = full temperature
        #       phi_total = full potential
        #       mu        = ion-to-electron mass ratio
        #       nu_e      = electron-ion + electron_neutral collision rate
        #       dz        = parallel direction
        #       Eig_func      = solution function
        #
        V_par_eq = sympy.expand(sympy.simplify(
            (p.fv_par.diff(p.t)
             + bout.DotProd(p.vE, bout.Grad(p.fv_par, p.x, p.metric))
             + p.mu * (p.N_total * p.Te0).diff(p.z) / p.N0
             + p.mu*1.71*(p.fTe).diff(p.z)
             - p.mu * p.phi_total.diff(p.z)
             + p.nu_e * p.fv_par)
            / p.Eig_func
        ))
    

        # BOUT vorticity equation: Alternative formulation
        # 
        print("Using BOUT vorticity equation.")
        Phi_eq = sympy.expand((
            p.vort.diff(p.t)  # d vort/dt
            + (p.N_total * p.fv_par).diff(p.z)  # d (N.v_par)/dt
            + bout.DotProd(p.vE, bout.Grad(p.vort, p.x, p.metric))  # vE. grad(vort)

            # 0.5* b x grad(N). grad^2(vE)
            - 0.5 * bout.DotProd(p.bxGradN, bout.Grad(bout.DotProd(p.vE, p.vE), p.x, p.metric))

            + p.nu_in*p.vort  # nu_in.vort
            - p.mu_ii*bout.Delp2Perp(p.vort, p.x, p.metric) # mu_ii. grad_perp^2(vort)
        ) / p.Eig_func
        )

        Te_eq = sympy.expand((
            p.fTe.diff(p.t)
            + bout.DotProd(p.vE, bout.Grad(p.fTe, p.x, p.metric))
            ) / p.Eig_func
        )


        # Non-lineqr (full) symbolic equations
        Nonlin_eq = [Ni_eq, V_par_eq, Phi_eq, Te_eq]

        # build linearized symbolic equns
        # * only consider terms with epsilon factor
        # * eqns for N, v_par, and phi are stored in Lin_eq[0], Lin_eq[1], and
        #   Lin_eq[2] respectively; if Te is calculated, stored in Lin_eq[4]
        #
        Lin_eq = [eq.coeff(p.epsilon) for eq in Nonlin_eq]

        print("Done")

        return Lin_eq
