#!/usr/bin/env /u/local/apps/python/2.6.1/bin/python
#
# Class that defines profiles

import numpy
from scipy.interpolate.fitpack2 import UnivariateSpline


class ProfileFit(UnivariateSpline):
    """
    Calculate a fit (as piecewise cubic polynomials) for a profile
    If ra,rb are specified, the spline is calculated in the interval (ra,rb) 
    (meaning that the interval x=0..1 is mapped to ra..rb). Otherwise,
    the full interval min(r)..max(r) is used
    """
    def __init__(self, r, y, ra=None, rb=None, norm=None):
        
        # Normalize r:
        self.rmin = min(r)
        self.rmax = max(r)
        self.r    = r.copy()
        self.y    = y.copy()
        if not ra: 
            self.ra = self.rmin 
        else: 
            self.ra=ra
        if not rb: 
            self.rb = self.rmax 
        else: 
            self.rb=rb
        x = (r-self.ra)/(self.rb-self.ra)  # if ra/rb are specified, x can be <0 or >1

        # Normalize the function values
        if not norm:
            norm = y[0] # normalize to y[0] if no norm value is explicitly provided
        self.norm = norm
        self.f0 = self.norm
        y /= self.f0

        # Calculate the splines
        UnivariateSpline.__init__(self, x,y,s=0)
        
    def derivatives(self, x):
        """Redefine derivatives method to use vector x, not only scalar"""
        return numpy.array([UnivariateSpline.derivatives(self, ix) for ix in x])
            
    def cumintegral(self, x):
        """Calculate integral of the spline between 0 and x"""
        return numpy.array([UnivariateSpline.integral(self, 0., ix) for ix in x])
       


class EqGrid(object):
    """Initialize equilibrium profiles (density, temperature, phi). 
    All profiles are defined on 0<=x<=1 interval"""

    def __init__(self, Nr=100, np=0, tp=0, pp=0, param=None, 
                       nparam=None, tparam=None, pparam=None): 
        # param is still here for legacy reasons, to keep old configs working)

        object.__init__(self)

        self.Nr = Nr
        self.x = numpy.linspace(0.,1.,Nr)
        self.h = 1./(Nr-1)  # Nr -- number radial *intervals*, not points

        # Initialize the profiles
        # Some of the profile options require additional information about radial interval ra,rb -- passed as param['ra'] etc
        if param:
            nparam = tparam = pparam = param
        self.ni  = self.get_nprofile(self.x, np, param=nparam)
        self.te  = self.get_tprofile(self.x, tp, param=tparam)
        self.phi = self.get_pprofile(self.x, pp, param=pparam)


    def get_nprofile(self, x, p, param=None):
        """Normalized density profile (and radial derivative)"""
        ni = numpy.zeros((3,len(x)),float)  # ni[0]-density, ni[1]-derivative dni/dx

        if p == 0:
            ni[0,:]  = 1.
            ni[1,:]  = 0.
            ni[2,:]  = 0.

        elif p == 1:
            ni[0,:]  = 1. - 0.5*x
            ni[1,:]  = -0.5
            ni[2,:]  = 0.

        elif p == "linear_test":
            x = numpy.linspace(0.,1.,self.Nr+2) # To make it like BOUT++
            ni[0,:]  = 1. - 0.6*x[1:-1]
            ni[1,:]  = -0.6
            ni[2,:]  = 0.


        elif p == 2:
            Ln = param.w
            ni[0,:]  =  numpy.exp(-x/Ln)
            ni[1,:]  = -numpy.exp(-x/Ln)/Ln
            ni[2,:]  =  numpy.exp(-x/Ln)/(Ln**2)

        elif p == 3:
            # Similar to LAPD profile in biasing expriment -- phase before biasing
            # Carter and Maggs, PoP, biasing paper, Fig. 3a
            pc=numpy.array([-13.6595, 311.560, -2205.60, 7159.09, -11090.9, 6666.66])/2.5 # coefficients from POLY_FIT
            rmin_m = 0.15
            rmax_m = 0.45
            x = numpy.linspace(0.,1.,self.Nr+2) # To make it like BOUT++
            rfull = x*(rmax_m-rmin_m) + rmin_m   # r in meters - this is how the fit was done
            rm = rfull[1:-1]

            ni[0,:] = pc[0] + pc[1]*rm + pc[2]*rm**2 + pc[3]*rm**3 + pc[4]*rm**4 + pc[5]*rm**5
            ni[1,:] = ((pc[1] + 2*pc[2]*rm + 3*pc[3]*rm**2 + 4*pc[4]*rm**3 + 5*pc[5]*rm**4)
                       *(rmax_m-rmin_m))
            ni[2,:] = ((2*pc[2] + 6*pc[3]*rm + 12*pc[4]*rm**2 + 20*pc[5]*rm**3)
                       *(rmax_m-rmin_m)**2)

            # C4 derivatives like in BOUT++
            #from BOUTppmath import DDX_C4

            #ni[1,2:-2] = DDX_C4(ni[0,2:-2],ni[0,1:-3],ni[0,3:-1],ni[0,0:-4],ni[0,4:])

        elif p == '3_with_buffer':
            # Similar to LAPD profile in biasing expriment -- phase before biasing
            # Carter and Maggs, PoP, biasing paper, Fig. 3a
            pc=numpy.array([-13.6595, 311.560, -2205.60, 7159.09, -11090.9, 6666.66])/2.5 # coefficients from POLY_FIT
            rmin_m = 0.15
            rmax_m = 0.45
            rm = x*(rmax_m-rmin_m) + rmin_m   # r in meters - this is how the fit was done

            ni[0,:] = 0.5 + pc[0] + pc[1]*rm + pc[2]*rm**2 + pc[3]*rm**3 + pc[4]*rm**4 + pc[5]*rm**5
            ni[1,:] = ((pc[1] + 2*pc[2]*rm + 3*pc[3]*rm**2 + 4*pc[4]*rm**3 + 5*pc[5]*rm**4)
                       *(rmax_m-rmin_m))
            ni[2,:] = ((2*pc[2] + 6*pc[3]*rm + 12*pc[4]*rm**2 + 20*pc[5]*rm**3)
                       *(rmax_m-rmin_m)**2)



#            import BOUTmath
#            ni[1,:] = BOUTmath.deriv_full(ni[0,:], self.h)
#            ni[2,:] = BOUTmath.deriv_full(ni[1,:], self.h)

        elif p == 4:
            ni[0,:]  = 1. - 0.5*x*x
            ni[1,:]  = -x
            ni[2,:]  = -1.

        elif p == 5:  
            # d tanh(x) /dx = 1 - tanh(x)**2
            w  = param.w
            x0 = param.x0
            n2 = param.n2
            n0 = 1.

            # ni = n0 + (f(x)-f(0))/(f(1)-f(0))*(n2-n0)
            # ni(0) = n0,   ni(1) = n2
            # f(x) = (1-tanh((x-x0)*w))
            f0 = (1-numpy.tanh((0.-x0)*w))
            f1 = (1-numpy.tanh((1.-x0)*w))
            alpha = 1./(f1-f0)*(n2-n0)

            ni[0,:]  =((1-numpy.tanh((x-x0)*w)) - f0)*alpha + n0
            ni[1,:]  = (-(1. - (numpy.tanh((x-x0)*w))**2)*w) * alpha
            ni[2,:]  = (2.*numpy.tanh((x-x0)*w)*( 1. - (numpy.tanh((x-x0)*w))**2 )*w*w) * alpha


        elif p == 'tanh':
            ##-option for comparison with 2DX
            rmin_m = 0.15
            rmax_m = 0.45
            rm = x*(rmax_m-rmin_m) + rmin_m 
            
            f_rmin=1.0
            f_rmax=0.5

            rav=0.30
            wid=0.05

            rarg=(rav-rm)/wid

            ni[0,:]  = f_rmax + (f_rmin-f_rmax)*0.5*(1.0+numpy.tanh(rarg))
            ni[1,:]  = ((f_rmax-f_rmin)/(2*wid))/(numpy.cosh(rarg))**2
            ni[2,:]  = ((f_rmax-f_rmin)/wid**2)*(numpy.tanh(rarg))/(numpy.cosh(rarg))**2

            
        elif p == 'LAPD_nonrotating':
            # Non-rotating plasma, data from paper J. Maggs et al, PoP 14 (2007), Fig.9
            # Use with params: ra=0.15, rb=0.45m,  ni0=3.8e18, te0=9 eV
            r_m  = numpy.array([0.,   10., 20.,   30.,  40., 45.,  50.])*0.01 # in m
            Ne   = numpy.array([1.9, 1.8, 1.4,  0.75,  0.25, 0.15, 0.1])*2.e12  # in cm^-3
            s = ProfileFit(r_m, Ne)
            ni[0:2,:] = numpy.transpose(s.derivatives(x)[:,0:2])

        elif p == 'LAPD_rotating':
            # Rotating plasma, data from paper J. Maggs et al, PoP 14 (2007), Fig.9
            # Use with params: ra=0.15, rb=0.45m,  ni0=3.6e18, te0=7.8 eV
            ra, rb = param.ra, param.rb
            r_m  = numpy.array([0.,  10., 15.,  20., 25.,  30., 35.,  45., 50.])*0.01 # in m
            Ne   = numpy.array([1.8, 1.8, 2.0, 2.0, 0.8,  0.2, 0.13, 0.1, 0.09])*2.e12 # in cm^-3
            s = ProfileFit(r_m, Ne, ra, rb) # x=0..1 interval is mapped to r=ra..rb
            ni[0:3,:] = numpy.transpose(s.derivatives(x)[:,0:3])

        elif p == 'gauss':
            # Gaussian with max at r=ra, f(ra=1), f(rb=vb)
            w  = param.w
            vb = param.vb
            
            v  = numpy.exp(-(1./w)**2)  # value at r=rb
            alpha = (1.-vb)/(1.-v)   # f = (g(x)-v) * alpha + vb

            ni[0,:]  = (numpy.exp(-(x/w)**2)-v) *alpha + vb
            ni[1,:]  = -numpy.exp(-(x/w)**2)*2*x/(w*w)*alpha
            ni[2,:]  = numpy.exp(-(x/w)**2)*( (2*x/(w*w))**2 - 2/(w*w))*alpha

        elif p == 'invgauss':
            # Inversed gaussian with max at r=rb, f(ra=va), f(rb=1)
            w  = param.w
            va = param.va

            v  = numpy.exp(-(1./w)**2)
            alpha = -(1.-va)/(1.-v)   # f = (g(x)-v) * alpha + 1

            ni[0,:]  = (numpy.exp(-(x/w)**2)-v) *alpha + 1.
            ni[1,:]  = -numpy.exp(-(x/w)**2)*2*x/(w*w)*alpha
            ni[2,:]  = numpy.exp(-(x/w)**2)*( (2*x/(w*w))**2 - 2/(w*w))*alpha


        elif p == 'DaveA_set1':
            # Profile from swept probe data. Troy, 2010/03/30.
            # Data: ethanol, /data/ppopovich/DAVE_DWsolver/PROFILES_20100330/profiles.sav
            # Use with params: ra=0.001, rb=0.042m, n0=1.9843e18
            ra, rb = param.ra, param.rb
            r_m = numpy.array([0.0000,  0.2500,  0.5000,  0.7500,  1.0000,  1.2500,  1.5000,  1.7500,  2.0000,  2.2500,  2.5000,  2.7500,  3.0000,  3.2500,  3.5000,  3.7500,  4.0000,  4.2500,  4.5000,  4.7500,  5.0000]) *0.01 # in m
            Ne = numpy.array([1.3292,  1.3298,  1.3350,  1.3402,  1.3439,  1.3452,  1.3412,  1.3271,  1.3058,  1.2966,  1.3208,  1.3980,  1.5454,  1.7402,  1.9067,  1.9860,  1.9994,  1.9870,  1.9727,  1.9681,  1.9843])*1.e12 # in cm^-3

            nnorm = 1.9843*1.e12  # normalization for ni
            s = ProfileFit(r_m, Ne, ra, rb, norm=nnorm) # x=0..1 interval is mapped to r=ra..rb
            ni[0:3,:] = numpy.transpose(s.derivatives(x)[:,0:3])

        elif p == 'DaveA_set2_avg':
            # Profile from swept probe data, averaged. Troy, 2010/03/30.
            # Data: ethanol, /data/ppopovich/DAVE_DWsolver/PROFILES_20100330/profiles.sav
            # Use with params: ra=0.001, rb=0.042m, n0=1.9906e18
            ra, rb = param.ra, param.rb
            r_m = numpy.array([0.0000,  0.2500,  0.5000,  0.7500,  1.0000,  1.2500,  1.5000,  1.7500,  2.0000,  2.2500,  2.5000,  2.7500,  3.0000,  3.2500,  3.5000,  3.7500,  4.0000,  4.2500,  4.5000,  4.7500,  5.0000]) *0.01 # in m
            Ne = numpy.array([1.4474,  1.4443,  1.4414,  1.4499,  1.4671,  1.4773,  1.4711,  1.4588,  1.4537,  1.4637,  1.4935,  1.5437,  1.6088,  1.6823,  1.7568,  1.8249,  1.8836,  1.9345,  1.9763,  1.9988,  1.9906])*1.e12 # in cm^-3

            nnorm = 1.9906*1.e12  # normalization for ni
            s = ProfileFit(r_m, Ne, ra, rb, norm=nnorm) # x=0..1 interval is mapped to r=ra..rb
            ni[0:3,:] = numpy.transpose(s.derivatives(x)[:,0:3])


        elif p == 'Shu_annulus':
            # Profile from Shu's run with annulus of inner radius 6 cm
            # Annulus bias of 75V
            # Neon plasma A=20
            # ra=0.03 m, rb=0.0775 m
            rmin = 3.0
            rmax = 7.75
            A1 = 0.381
            A2 = 0.0554
            x0 = 5.639
            dx = 0.269


            rcm = rmin + (rmax-rmin)*x

            from . import BOUTppmath

            ni[0,:] = ((A1-A2)/(1+numpy.exp((rcm-x0)/dx))+A2)*2.6
            ni[1,:] = -2.6*(A1-A2)*numpy.exp((rcm-x0)/dx)/dx/(1+numpy.exp((rcm-x0)/dx))**2
            ni[2,:] = 5.2*(A1-A2)*numpy.exp(2.*(rcm-x0)/dx)/dx**2/(1+numpy.exp((rcm-x0)/dx))**3 - 2.6*(A1-A2)*numpy.exp((rcm-x0)/dx)/dx**2/(1+numpy.exp((rcm-x0)/dx))**2

        return ni


    def get_tprofile(self, x, p, param=None):
        """Normalized temperature profile"""
        te = numpy.zeros((2,len(x)),float)
        te[1][:]  = 0.  # d te/dr, not used

        if p == 0:
            te[0,:] = 1.

        elif p == 1:
            te[0,:] = 1. - x
            te[1,:] = -1.

        elif p == 'LAPD_BOUTpp':
            # Non-rotating plasma, data from paper J. Maggs et al, PoP 14 (2007), Fig.4b
            pc=numpy.array([0.01000000000000,  -0.00217278378278,  -0.03380882156298,   0.51773982924486, -2.03366007734383,   3.17529533513940,  -2.19693112025107,   0.56522609391293])*1.e2 # coefficients from POLY_FIT
            rmin_m = 0.15
            rmax_m = 0.45
            rm = x

            te[0,:] = pc[0] + pc[1]*rm + pc[2]*rm**2 + pc[3]*rm**3 + pc[4]*rm**4 + pc[5]*rm**5 + pc[6]*rm**6 + pc[7]*rm**7
            te[1,:] = (pc[1] + 2*pc[2]*rm + 3*pc[3]*rm**2 + 4*pc[4]*rm**3 + 5*pc[5]*rm**4 + 6*pc[6]*rm**5 + 7*pc[7]*rm**6)
                       


        elif p == 'LAPD_nonrotating':
            # Non-rotating plasma, data from paper J. Maggs et al, PoP 14 (2007), Fig.4b
            r_m  = numpy.array([0.,  10., 20., 25.,  30., 35.,  43.,  50.])*0.01 # in m
            Te   = numpy.array([9.,  9.,  8.2, 7.,  4.5,  2.5,  1.75, 1.7]) # in eV
            s = ProfileFit(r_m, Te)
            te[0,:] = s(x)
            from . import BOUTppmath
            te[1,:] = BOUTppmath.deriv_full(te[0,:], self.h)

        elif p == 'LAPD_rotating':
            # Rotating plasma, data from paper J. Maggs et al, PoP 14 (2007), Fig.11
            # Temperature normalization: te0=7.8 eV
            ra, rb = param.ra, param.rb

            r_m  = numpy.array([0.,  5.,  10.,  20.,  25.,  28.,  32.,  36.,  43.,  50.])*0.01 # in m
            Te   = numpy.array([7.8, 7.8, 7.8,  8.1,  7.,   5.3,  5.1,  5.2,  5.3,  5.4]) # in eV
            s = ProfileFit(r_m, Te, ra, rb, norm=7.8) # x=0..1 interval is mapped to r=ra..rb
            te[0,:] = s(x)
            from . import BOUTppmath
            te[1,:] = BOUTppmath.deriv_full(te[0,:], self.h)

        elif p == 'gauss':
            # Gaussian with max at r=ra, f(ra=1), f(rb=vb)
            w  = param.w
            vb = param.vb
            
            v  = numpy.exp(-(1./w)**2)  # value at r=rb
            alpha = (1.-vb)/(1.-v)   # f = (g(x)-v) * alpha + vb

            te[0,:]  = (numpy.exp(-(x/w)**2)-v) *alpha + vb
            te[1,:]  = -numpy.exp(-(x/w)**2)*2*x/(w*w)*alpha

        elif p == 'invgauss':
            # Inversed gaussian with max at r=rb, f(ra=va), f(rb=1)
            w  = param.w
            va = param.va

            v  = numpy.exp(-(1./w)**2)
            alpha = -(1.-va)/(1.-v)   # f = (g(x)-v) * alpha + 1

            te[0,:]  = (numpy.exp(-(x/w)**2)-v) *alpha + 1.
            te[1,:]  = -numpy.exp(-(x/w)**2)*2*x/(w*w)*alpha

        elif p == 'DaveA_set1':
            # Profile from swept probe data. Troy, 2010/03/30.
            # Data: ethanol, /data/ppopovich/DAVE_DWsolver/PROFILES_20100330/profiles.sav
            # Use with params: ra=0.001, rb=0.042m, te0=7.1 eV
            ra, rb = param.ra, param.rb
            r_m = numpy.array([0.0000,  0.2500,  0.5000,  0.7500,  1.0000,  1.2500,  1.5000,  1.7500,  2.0000,  2.2500,  2.5000,  2.7500,  3.0000,  3.2500,  3.5000,  3.7500,  4.0000,  4.2500,  4.5000,  4.7500,  5.0000]) *0.01 # in m
            Te = numpy.array([3.7190,  3.7180,  3.7163,  3.7645,  3.8312,  3.8718,  3.9551,  4.2364,  4.7784,  5.4325,  6.0325,  6.4967,  6.7823,  6.9044,  6.9536,  7.0053,  7.0564,  7.0855,  7.0920,  7.0916,  7.1003]) # in eV

            tnorm = 7.1 # normalization for Te
            s = ProfileFit(r_m, Te, ra, rb, norm=tnorm) # x=0..1 interval is mapped to r=ra..rb
            te[0,:] = s(x)
            from . import BOUTppmath
            te[1,:] = BOUTppmath.deriv_full(te[0,:], self.h)

        elif p == 'DaveA_set2_avg':
            # Profile from swept probe data, averaged. Troy, 2010/03/30.
            # Data: ethanol, /data/ppopovich/DAVE_DWsolver/PROFILES_20100330/profiles.sav
            # Use with params: ra=0.001, rb=0.042m, te0=6.7543 eV
            ra, rb = param.ra, param.rb
            r_m = numpy.array([0.0000,  0.2500,  0.5000,  0.7500,  1.0000,  1.2500,  1.5000,  1.7500,  2.0000,  2.2500,  2.5000,  2.7500,  3.0000,  3.2500,  3.5000,  3.7500,  4.0000,  4.2500,  4.5000,  4.7500,  5.0000]) *0.01 # in m
            Te = numpy.array([3.6832,  3.7112,  3.7741,  3.8230,  3.8791,  4.0302,  4.3186,  4.6521,  4.9270,  5.1556,  5.4143,  5.7319,  6.0590,  6.3443,  6.5614,  6.6930,  6.7478,  6.7593,  6.7563,  6.7513,  6.7543]) # in eV

            tnorm = 6.7543 # normalization for Te
            s = ProfileFit(r_m, Te, ra, rb, norm=tnorm) # x=0..1 interval is mapped to r=ra..rb
            te[0,:] = s(x)
            from . import BOUTppmath
            te[1,:] = BOUTppmath.deriv_full(te[0,:], self.h)


        return te

    def get_pprofile(self, x, p, param=None):
        """Normalized phi0 profile (and radial derivatives up to 3rd order)"""
        phi = numpy.zeros((4,len(x)),float)  # phi[0]-density, phi[1]-first derivative dphi/dx etc

        if p == 0:
            # Const
            phi[0,:]  = 1.
            phi[1:,:] = 0.

        elif (p == 4) or (p == 'uniform'):
            # Quadratic -- solid body rotation, phi0 ~ (r/rmax)**2, rmax=param.rb
            # Normalization: phi0(rmax)=1
            dr = param.rb-param.ra
            r = param.ra + x*dr
            rmax = param.rb

            phi[0,:] = (r/rmax)**2
            phi[1,:] = 2*r/rmax**2*dr
            phi[2,:] = 2./rmax**2*dr**2
            phi[3,:] = 0.

        elif p == 13:
            # Profile for KH test (slab).
            from BOUTmath import int0_N,int1_N,int2_N,int3_N
            w = param.width
            x4 = x*4-2.  # project x=[0,1] to [-2,2] interval
            phi[0,:]  =     int3_N(x4-1.,w) + int3_N(x4+1.,w) - 2*int3_N(x4,w) + 0.5
                           # add 0.5 to make max(phi0)=1, for easy normalization
            phi[1,:]  =  4*(int2_N(x4-1.,w) + int2_N(x4+1.,w) - 2*int2_N(x4,w))
            phi[2,:]  = 16*(int1_N(x4-1.,w) + int1_N(x4+1.,w) - 2*int1_N(x4,w))
            phi[3,:]  = 64*(int0_N(x4-1.,w) + int0_N(x4+1.,w) - 2*int0_N(x4,w))
            # 4,16,64 -- factors to transform the interval from [0,1] to [-2,2] (derivatives!)

        elif p == 'LAPD_rotating':
            # Rotating plasma, data from paper J. Maggs et al, PoP 14 (2007), Fig.6a
            # Use with params: ra=0.15, rb=0.45m,  phi0v=85.45,  te0=7.8

            r_m  = numpy.array([10., 15.,  20.,  25.,  30.,  35.,  40.,  45.,  50.])*0.01 # in m
            M    = numpy.array([ 0., 0. ,  0. , 0.07, 0.25,  0.8,  1.3,  1.4,  1.4]) # Mach number
#            Cs   = 11300. # m/s;  as an approx. take Te=5.34 eV (value at 0.45m)
#            vth  = M*Cs
            
            ra, rb = param.ra, param.rb
            dxdr   = 0.0474784165232100 # dxdr=1./(rmax-rmin): divide the spline by dxdr
                     # to compensate for the multiplication by dxdr in update_params function
            s = ProfileFit(r_m, M/dxdr, ra=ra, rb=rb, norm=1.) # x=0..1 interval is mapped to r=ra..rb
            phi0max  = 10.95470779302
            phi[0,:]   = s.cumintegral(x)[:]/phi0max
            phi[1:4,:] = numpy.transpose(s.derivatives(x)[:,0:3])/phi0max
            # with this normalization, we have 
            # phi[0] = 1 at r=0.45m
            # phi0*phi[1]*dxdr = M = vtheta/Cs  (*dxrd is done in update_params)

        elif p == 'gauss':
            # Gaussian with max at r=ra, f(ra=1), f(rb=vb)
            w  = param.w
            vb = param.vb
            
            v  = numpy.exp(-(1./w)**2)  # value at r=rb
            alpha = (1.-vb)/(1.-v)   # f = (g(x)-v) * alpha + vb

            phi[0,:]  = (numpy.exp(-(x/w)**2)-v) *alpha + vb
            phi[1,:]  = -numpy.exp(-(x/w)**2)*2*x/(w*w)*alpha
            phi[2,:]  = numpy.exp(-(x/w)**2)*( (2*x/(w*w))**2 - 2/(w*w))*alpha
            phi[3,:]  = numpy.exp(-(x/w)**2)*( -8*x**3/w**6 + 12*x/w**4)*alpha

        elif p == 'invgauss':
            # Inversed gaussian with max at r=rb, f(ra=va), f(rb=1)
            w  = param.w
            va = param.va

            v  = numpy.exp(-(1./w)**2)
            alpha = -(1.-va)/(1.-v)   # f = (g(x)-v) * alpha + 1

            phi[0,:]  = (numpy.exp(-(x/w)**2)-v) *alpha + 1.
            phi[1,:]  = -numpy.exp(-(x/w)**2)*2*x/(w*w)*alpha
            phi[2,:]  = numpy.exp(-(x/w)**2)*( (2*x/(w*w))**2 - 2/(w*w))*alpha
            phi[3,:]  = numpy.exp(-(x/w)**2)*( -8*x**3/w**6 + 12*x/w**4)*alpha

        elif p == 'DaveA_set1':
            # Profile from swept probe data. Troy, 2010/03/30.
            # Data: ethanol, /data/ppopovich/DAVE_DWsolver/PROFILES_20100330/profiles.sav
            # Use with params: ra=0.001, rb=0.042m, phi0=2.8565 V
            ra, rb = param.ra, param.rb
            r_m = numpy.array([0.0000,  0.2500,  0.5000,  0.7500,  1.0000,  1.2500,  1.5000,  1.7500,  2.0000,  2.2500,  2.5000,  2.7500,  3.0000,  3.2500,  3.5000,  3.7500,  4.0000,  4.2500,  4.5000,  4.7500,  5.0000]) *0.01 # in m
            Phi = numpy.array([3.6935,  3.6921,  3.6759,  3.6428,  3.6031,  3.5692,  3.5439,  3.5228,  3.5014,  3.4751,  3.4372,  3.3664,  3.2357,  3.0617,  2.9171,  2.8574,  2.8524,  2.8531,  2.8454,  2.8420,  2.8565]) # in V

            pnorm = 2.8565 # normalization for Phi0
            s = ProfileFit(r_m, Phi, ra, rb, norm=pnorm) # x=0..1 interval is mapped to r=ra..rb
            phi[0:4,:] = numpy.transpose(s.derivatives(x)[:,0:4])

        elif p == 'DaveA_set2_avg':
            # Profile from swept probe data, averaged. Troy, 2010/03/30.
            # Data: ethanol, /data/ppopovich/DAVE_DWsolver/PROFILES_20100330/profiles.sav
            # Use with params: ra=0.001, rb=0.042m, phi0=3.0127 V
            ra, rb = param.ra, param.rb
            r_m = numpy.array([0.0000,  0.2500,  0.5000,  0.7500,  1.0000,  1.2500,  1.5000,  1.7500,  2.0000,  2.2500,  2.5000,  2.7500,  3.0000,  3.2500,  3.5000,  3.7500,  4.0000,  4.2500,  4.5000,  4.7500,  5.0000]) *0.01 # in m
            Phi = numpy.array([3.6972,  3.6950,  3.6803,  3.6417,  3.5846,  3.5305,  3.4920,  3.4546,  3.4019,  3.3414,  3.2937,  3.2626,  3.2243,  3.1601,  3.0906,  3.0476,  3.0348,  3.0299,  3.0178,  3.0077,  3.0127]) # in V

            pnorm = 3.0127 # normalization for Phi0
            s = ProfileFit(r_m, Phi, ra, rb, norm=pnorm) # x=0..1 interval is mapped to r=ra..rb
            phi[0:4,:] = numpy.transpose(s.derivatives(x)[:,0:4])

        return phi
 
