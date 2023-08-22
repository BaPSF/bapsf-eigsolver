
import sympy
from bapsf_eigsolver import profiles
from bapsf_eigsolver import BOUTppmath as Bout
from bapsf_eigsolver import misctools as tools
from abc import abstractmethod
from abc import ABC


class SymEqBase(ABC):

    def __init__(self, metric="cyl"):

        self.metric = metric

        # create symbolic variables and functions
        # * metric is passed to BOUT
        self.varpack = self._create_symbols(self.metric)

        self.symb_eq = self.build_symb_eq(self.varpack)

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
        r, th, z, t = sympy.symbols('r theta z t')  # Coordinates and time
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
        eig_func = sympy.exp(I * (m_theta * th + k * z - omega0 * t))

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
        N_total = N0 + (epsilon * N * N0 * eig_func)

        # setup potential profiles
        # phi0     :  equilibrium potential
        # phi      :  radial part of perturbed potential
        # phi_total:  full potential
        #
        phi0 = sympy.Function('phi0')(r)
        phi = sympy.Function('phi')(r)
        phi_total = phi0 + (epsilon * phi * eig_func)

        # setup parallel velocity profiles
        # v_par  :  radial dependence of the parallel electron velocity
        # fv_par :  parallel electron velocity, perturbed component only
        #
        v_par = sympy.Function('v')(r)
        fv_par = epsilon * v_par * eig_func

        # setup electron temperature profiles
        # Te0:  equilibrium electron temperature
        # Te: radial part of perturbed temperature
        # fTe: full temperature
        #
        Te0 = sympy.Function('Te0')(r)
        Te = sympy.Function('Te')(r)
        fTe = Te0 + epsilon * Te * eig_func

        # additional misc functions
        # gphi    :  gradient of full potential
        # gperpphi:  perpendicular (to b0) part of Grad Phi
        # vort    :  BOUT definition of vorticity
        # bxGradN :  temp variable, used in the vorticity eq.
        # vE      :  ExB drift velocity, equilibrium + perturbation
        #
        gphi = Bout.Grad(phi_total, x, metric)
        gperpphi = Bout.CrossProd(b0, Bout.CrossProd(gphi, b0))
        vort = Bout.DivPerp(N_total * Bout.GradPerp(phi_total, x, metric),
                            x,
                            metric)
        bxGradN = Bout.CrossProd(b0, Bout.Grad(N_total, x, metric))
        vE = Bout.CrossProd(b0, gphi)

        # Pack everything in one variable
        # fpack = self._pack_symbols(r, th, z, t, epsilon, k, m_theta, mu,
        #                            omega0,nu_e,nu_in,mu_ii,
        #            {'x':x, 'eig_func':eig_func, 'N0':N0, 'N':N, 'N_total':N_total,
        #             'phi0':phi0, 'phi':phi, 'phi_total':phi_total,
        #             'v_par':v_par, 'fv_par':fv_par,
        #             'Te0':Te0, 'Te': Te, 'fTe': fTe,
        #             'gphi':gphi, 'gperpphi':gperpphi, 'vort':vort,
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
            'eig_func': eig_func,
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
            'fTe': fTe,
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
        # for arg in args[:-1]:
        #     d[arg.name] = arg
        # for key, item in list(args[-1].items()):
        #     d[key] = item

        return d

    @property
    @abstractmethod
    def arguments(self):
        raise NotImplementedError

    @abstractmethod
    def build_symb_eq(self, p):
        raise NotImplementedError
