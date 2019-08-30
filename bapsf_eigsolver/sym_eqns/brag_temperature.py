
import sympy
from bapsf_eigsolver import BOUTppmath as Bout
from bapsf_eigsolver import misctools as tools
from abc import abstractmethod
from .base import SymEqBase


class BragTemp(SymEqBase):

    def __init__(self, metric):
        super().__init__(metric)

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
        #    + d(N_total * fv_par)/dz ] / eig_func
        #
        # where N_total = density
        #       vE      = ExB drift velocity
        #       fv_par  = perturbed parallel electron velocity
        #       dz      = parallel direction
        #       eig_func    = solution function
        #
        Ni_eq = sympy.expand(sympy.simplify(
            (p.N_total.diff(p.t)  # dN/dt
             + Bout.DotProd(p.vE, Bout.Grad(p.N_total, p.x, p.metric))
             + (p.N_total * p.fv_par).diff(p.z))
            / p.eig_func
        ))  # / p.N0 / p.eig_func)

        # Vparallel equation: parallel electron momentum
        # * eqn 4.2 in B. Friedman 2013 dissertation
        #   (https://escholarship.org/uc/item/4799v1k0)
        # * eqn 5.12 in D. Schaffner 2013 dissertation
        #   (https://escholarship.org/uc/item/7hz553m0)
        #
        #  [ d(fv_par)/dt
        #    + dot(vE, Grad(fv_par))
        #    + mu * d(N_total * Te0)/dz / N0
        #    + 1.71 * mu * Grad_par(Te)
        #    - mu * d(phi_total)/dz
        #    + nu_e * fv_par ] / eig_func
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
        #       eig_func      = solution function
        #
        V_par_eq = sympy.expand(sympy.simplify(
            (p.fv_par.diff(p.t)
             + Bout.DotProd(p.vE, Bout.Grad(p.fv_par, p.x, p.metric))
             + p.mu * (p.N_total * p.Te0).diff(p.z) / p.N0
             + p.mu * 1.71 * (p.fTe).diff(p.z)
             - p.mu * p.phi_total.diff(p.z)
             + p.nu_e * p.fv_par)
            / p.eig_func
        ))

        # BOUT vorticity equation: Alternative formulation
        #
        print("Using BOUT vorticity equation.")
        Phi_eq = sympy.expand((
                                      p.vort.diff(p.t)  # d vort/dt
                                      + (p.N_total * p.fv_par).diff(p.z)  # d (N.v_par)/dt
                                      + Bout.DotProd(p.vE, Bout.Grad(p.vort, p.x, p.metric))  # vE. grad(vort)

                                      # 0.5* b x grad(N). grad^2(vE)
                                      - 0.5 * Bout.DotProd(p.bxGradN,
                                                           Bout.Grad(Bout.DotProd(p.vE, p.vE), p.x, p.metric))

                                      + p.nu_in * p.vort  # nu_in.vort
                                      - p.mu_ii * Bout.Delp2Perp(p.vort, p.x, p.metric)  # mu_ii. grad_perp^2(vort)
                              ) / p.eig_func
                              )

        Te_eq = sympy.expand((p.fTe.diff(p.t)
                              + Bout.DotProd(p.vE, Bout.Grad(p.fTe, p.x, p.metric))
                             ) / p.eig_func)

        # Non-linear (full) symbolic equations
        Nonlin_eq = [Ni_eq, V_par_eq, Phi_eq, Te_eq]

        # build linearized symbolic eqns
        # * only consider terms with epsilon factor
        # * eqns for N, v_par, and phi are stored in Lin_eq[0], Lin_eq[1], and
        #   Lin_eq[2] respectively
        #
        Lin_eq = [eq.coeff(p.epsilon) for eq in Nonlin_eq]

        print("Done")

        return Lin_eq

    def get_Dvar_coeffs(self, eq, f, r):
        """Extract the coefficients (symbolic) at D[f, {r, i=0..2}] in the expression of the form
        eq = c2(r)*f''(r) + c1(r)*f'(r) + c0(r)*f(r)

        Parameters
        ----------
        eq: The lin"""

        # Coefficient at f''
        c2 = eq.coeff(f.diff(r, 2))
        if not c2:
            # transform NoneType value to 0
            c2 = 0
        eq = sympy.expand(sympy.simplify((eq - c2 * f.diff(r, 2))))

        # Coefficient at f'
        c1 = eq.coeff(f.diff(r))
        if not c1:
            c1 = 0
        eq = sympy.expand(sympy.simplify((eq - c1 * f.diff(r))))

        # Coefficient at f
        c0 = eq.coeff(f)
        if not c0:
            c0 = 0

        # Normalize all coefficient so that c2=1 to improve matrix properties
        # (make the matrix determinant closer to 1)
        # norm = c2
        # c2 /= norm
        # c1 /= norm
        # c0 /= norm

        return c0, c1, c2

    def compile_function(self, sf, p, pvalues):
        """Compile a symbolic function (sf) info a callable function
        using the values given. Result: f(r,ni,te,phi0)

        Arguments:
        sf -- symbolic function (representing the coefficient of the equations)
        p  -- pack of all symbols
        pvalues -- numerical values
        """

        # Replace symbols by values given
        ni = sympy.Symbol("ni")
        te = sympy.Symbol("te")
        phi = sympy.Symbol("phi")
        nu_e = sympy.Symbol("nu_e")

        if sf:
            # Substitute the scalar constants by numerical values
            sf = sf.subs(sympy.I, complex(0., 1))
            sf = sf.subs(p.k, pvalues.k)
            sf = sf.subs(p.nu_in, pvalues.nu_in)
            sf = sf.subs(p.mu, pvalues.mu)
            sf = sf.subs(p.m_theta, pvalues.m_theta)

            # Substitute functions(r) and their derivatives by
            # simple names suitable for further substitution by a vector

            sf = sf.subs(p.N0.diff(p.r, 2), pvalues.n0 * sympy.Symbol("ni[2]"))
            sf = sf.subs(p.N0.diff(p.r), pvalues.n0 * sympy.Symbol("ni[1]"))
            sf = sf.subs(p.N0, pvalues.n0 * sympy.Symbol("ni[0]"))
            sf = sf.subs(p.phi0.diff(p.r, 3), pvalues.phi0 * sympy.Symbol("phi[3]"))
            sf = sf.subs(p.phi0.diff(p.r, 2), pvalues.phi0 * sympy.Symbol("phi[2]"))
            sf = sf.subs(p.phi0.diff(p.r), pvalues.phi0 * sympy.Symbol("phi[1]"))
            sf = sf.subs(p.phi0, pvalues.phi0 * sympy.Symbol("phi[0]"))
            sf = sf.subs(p.Te0.diff(p.r, 2), sympy.Symbol("te[2]"))
            sf = sf.subs(p.Te0.diff(p.r, 1), sympy.Symbol("te[1]"))
            sf = sf.subs(p.Te0, sympy.Symbol("te[0]"))

        # Ugly hack:
        #   adding a dummy variable that will ensure that the result of
        #   the function is a vector (not scalar!), for any expression sf
        #   (for example, for sf=0).
        #
        # The compiled function should be called with an extra
        # argument "dummyvec" with 0 values.
        #
        dummyvec = sympy.Symbol("dummyvec")
        sf = sf + dummyvec

        f = sympy.lambdify((p.r, ni, te, phi, nu_e, p.mu_ii, dummyvec),
                           sf)
        return f

    def apply_params(self, pvalues):
        """Compile the symbolic functions (coefficients) into callable functions
        using the values from "pvalues" object
        Construct the arrays LHS/RHS of the form [i_eq, i_var, i_order], with
        indices i_eq -- equation index, i_var -- variable index (N,v_par,phi), i_order -- derivative index (0,1,2)
        Elements of array: callable functions f(r,ni,te,phi,nu_e,mu_ii,dummyvec)
        """

        # outermost index goes first in LHS_fcoeff[i][j][k]
        LHS_fcoeff = [[[0 for i in range(3)]
                       for j in range(self.NVAR)]
                      for k in range(self.NVAR)]

        RHS_fcoeff = [[[0 for i in range(3)]
                       for j in range(self.NVAR)]
                      for k in range(self.NVAR)]
        #
        # LHS_fcoeff = [[[0] * 3] * self.NVAR] * self.NVAR
        # RHS_fcoeff = [[[0] * 3] * self.NVAR] * self.NVAR

        for i_eq in range(self.NVAR):
            for i_var in range(self.NVAR):
                # Get all coefficients at f'', f', f

                coeffs = self.get_Dvar_coeffs(self.symb_LHS[i_eq],  # linear equation
                                              self.vars[i_var],  # variable (N,v_par,phi)
                                              self.varpack.r)  # radial variable -- symbolic

                for i_order in range(3):
                    LHS_fcoeff[i_eq][i_var][i_order] = \
                        self.compile_function(coeffs[i_order], self.varpack, pvalues)

                coeffs = self.get_Dvar_coeffs(self.symb_RHS[i_eq],  # linear equation
                                              self.vars[i_var],  # variable (N,v_par,phi)
                                              self.varpack.r)  # radial variable -- symbolic

                for i_order in range(3):
                    RHS_fcoeff[i_eq][i_var][i_order] = \
                        self.compile_function(coeffs[i_order], self.varpack, pvalues)

        return LHS_fcoeff, RHS_fcoeff

    @property
    def arguments(self):
        return 'r', 'ni', 'te', 'phi', 'nu_e', 'mu_ii'
