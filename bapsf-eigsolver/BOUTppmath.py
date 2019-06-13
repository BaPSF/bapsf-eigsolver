#!/usr/bin/env /u/local/apps/python/2.6.1/bin/python
"""
Define finite differencing schemes (same as in BOUT)
and useful vector/differential operations for 4D arrays
-----------------------------------------------------------


def grad_cyl_4D(fval, du):
def Div_perp_cyl_4D(vval, du):
def advec_cyl_4D(vval, fval, du):
def crossProd_4D(v1,v2): 
def dotProd_4D(v1,v2):
def ddt_4D(fval, du):
def ddx_4D(fval, du):
def ddy_4D(fval, du):

"""




import numpy as np 

# -----------------------------------------------------------
# Finite difference schemes as defined in BOUT


#/*----1st derivative----*/

#/*right-sided, 1st order*/
def DDX_R1(fc,fm,fp,fmm,fpp):
    return ((fp)-(fc))
 
#/*left-sided, 1st order*/
def DDX_L1(fc,fm,fp,fmm,fpp):
    return ((fc)-(fm))

#/*central, 4th order*/
def DDX_C4(fc,fm,fp,fmm,fpp):
    return ( ((8.*(fp) - 8.*(fm) + (fmm) - (fpp))/12.) ) 


#central, 2nd order
def DDX_C2(fc,fm,fp,fmm,fpp):
    return ( 0.5*((fp)-(fm)) )


#/*---2nd derivative----*/

#central, 4th order
def D2DX2_C4(fc,fm,fp,fmm,fpp):
    return  ( (-(fpp) + 16.*(fp) - 30.*(fc) + 16.*(fm) - (fmm))/12. )

def D2L1DR1(fc,fm,fp,fmm,fpp):
    return (fp - 2.*fc +fm)


# Advection schemes as defined in BOUT

#/*central, 2nd order*/
def VDDX_C2(vc,fc,fm,fp,fmm,fpp):
    return  ( (vc)*0.5*((fp)-(fm)) )

#/*upwind, 1st order*/
def VDDX_U1(vc,fc,fm,fp,fmm,fpp):
    return np.where(vc > 0., (vc)*((fc)-(fm)), (vc)*((fp)-(fc)) )

#/*central, 4th order*/
def VDDX_C4(vc,fc,fm,fp,fmm,fpp):
    return ( ((vc)*(8.*(fp) - 8.*(fm) + (fmm) - (fpp))/12.) ) 

#/*upwind, 4th order*/
def VDDX_U4(vc,fc,fm,fp,fmm,fpp):
    np.where (vc > 0.0, (vc)*(4.*(fp) - 12.*(fm) + 2.*(fmm) + 6.*(fc))/12., 
                     (vc)*(-4.*(fm) + 12.*(fp) - 2.*(fpp) - 6.*(fc))/12.)

# -----------------------------------------------------------

def addzguard(fval,du):
    # Add 2 guard points in azimuthal direction. Assign values to ensure periodicity
    # Inputs:
    #        fval[*,*,*,*] - 4D array of function values
    #        du            - BOUTreader object with all grid information
    #
    # Outputs:
    #        4D array with extra 4 points in azimuthal direction
    #       
    
    # Add extra 4 points in theta direction
    flarge = np.zeros(fval.shape + np.array([0,0,4,0]), fval.dtype)
    flarge[:,:,2:-2,:] = fval[:,:,:,:].copy()
    # Assign periodic values. BOUT convention: open periodic: iz=0 <-> nz (last)
    #    flarge[:,:,2,:]  <-> fval[:,:,-1,:]
    #    flarge[:,:,-3,:] <-> fval[:,:,0,:]
    flarge[:,:,0,:] = fval[:,:,-2,:]
    flarge[:,:,1,:] = fval[:,:,-1,:]
    flarge[:,:,-2,:] = fval[:,:,0,:]
    flarge[:,:,-1,:] = fval[:,:,1,:]

    return flarge

# -----------------------------------------------------------

def grad_cyl_4D(fval, du):
#
# Evaluate gradient in cylindrical coordinates (dimensionless)
#
# Inputs:
#        fval[*,*,*,*] - 4D array of function values
#        du            - GRIDreader object with all grid information
#
# Outputs:
#        gradient cyl. components in 4D (margin points excluded)
#

    # Initialize grad4D as a list (r,theta,z) of 4D arrays of the same size as fval
    grad4D = [np.zeros(fval.shape, fval.dtype),  # r 
              np.zeros(fval.shape, fval.dtype),  # theta
              np.zeros(fval.shape, fval.dtype)]  # z


    # Call syntax:  DDX_C4(f[ix,iy,iz,it],f[ix-1,iy,iz,it],f[ix+1,iy,iz,it], $
    #                                     f[ix-2,iy,iz,it],f[ix+2,iy,iz,it])/dr_n

    # Radial/parallel derivatives are not calculated in guard points (2 on each size)

    # Cribsheet: slice indices  i:nx   <--> i:    (include last element)
    #                           i:nx-1 <--> i:-1  (exclude last element)
    #                           i:nx-2 <--> i:-2  (exclude last 2 elements)

    # Radial derivative of fval
    grad4D[0][2:-2,:,:,:] = (
            DDX_C4(fval[2:-2,:,:,:], fval[1:-3,:,:,:], fval[3:-1,:,:,:], 
                                     fval[0:-4,:,:,:], fval[4:  ,:,:,:])
                                               / du.dr_n)
                           

    # Theta derivative of fval
    # Add theta guard cells to the 4D array
    fval_wg = addzguard(fval, du)
     # slice indices in z:   [:] (original array) <--> [2:-2] (array with guard cells)
    grad4D[1][:,:,:,:] = (
            DDX_C4(fval_wg[:,:,2:-2,:], fval_wg[:,:,1:-3,:], fval_wg[:,:,3:-1,:], 
                                        fval_wg[:,:,0:-4,:], fval_wg[:,:,4:  ,:])
                                               / (du.dtheta*du.rxy[:,:,np.newaxis,np.newaxis]))

    return grad4D

#; TODO
#grad4D.z[*,*,*,*] = 0

# -----------------------------------------------------------
def grad_perp_cyl_4D(fval, du):
#
# Evaluate gradient in cylindrical coordinates (dimensionless)
#
# Inputs:
#        fval[*,*,*,*] - 4D array of function values
#        du            - GRIDreader object with all grid information
#
# Outputs:
#        gradient cyl. components in 4D (margin points excluded)
#

    # Initialize grad4D as a list (r,theta,z) of 4D arrays of the same size as fval
    grad4D = [np.zeros(fval.shape, fval.dtype),  # r 
              np.zeros(fval.shape, fval.dtype)]  # theta
              


    # Call syntax:  DDX_C4(f[ix,iy,iz,it],f[ix-1,iy,iz,it],f[ix+1,iy,iz,it], $
    #                                     f[ix-2,iy,iz,it],f[ix+2,iy,iz,it])/dr_n

    # Radial/parallel derivatives are not calculated in guard points (2 on each size)

    # Cribsheet: slice indices  i:nx   <--> i:    (include last element)
    #                           i:nx-1 <--> i:-1  (exclude last element)
    #                           i:nx-2 <--> i:-2  (exclude last 2 elements)

    # Radial derivative of fval
    grad4D[0][2:-2,:,:,:] = (
            DDX_C4(fval[2:-2,:,:,:], fval[1:-3,:,:,:], fval[3:-1,:,:,:], 
                                     fval[0:-4,:,:,:], fval[4:  ,:,:,:])
                                               / du.dr_n)
                           

    # Theta derivative of fval
    # Add theta guard cells to the 4D array
    fval_wg = addzguard(fval, du)
     # slice indices in z:   [:] (original array) <--> [2:-2] (array with guard cells)
    grad4D[1][:,:,:,:] = (
            DDX_C4(fval_wg[:,:,2:-2,:], fval_wg[:,:,1:-3,:], fval_wg[:,:,3:-1,:], 
                                        fval_wg[:,:,0:-4,:], fval_wg[:,:,4:  ,:])
                                               / (du.dtheta*du.rxy[:,:,np.newaxis,np.newaxis]))

    return grad4D


# -----------------------------------------------------------
def Div_perp_cyl_4D(vval, du):
#
# Evaluate the perp. part of the divergence (dimensionless)
#
# Inputs:
#        vval[*,*,*,*] - 4D array of vector components
#        du            - GRIDreader object with all grid information
#
# Outputs:
#        perpendicular divergence (cyl coordinates) as 4D array, radial guard points excluded
#

    # Initialize res4D as an array of the same size as vval[0]
    res4D = np.zeros(vval[0].shape, vval[0].dtype)

    # Radial/parallel derivatives are not calculated in guard points (2 on each size)

    # Cribsheet: slice indices  i:nx   <--> i:    (include last element)
    #                           i:nx-1 <--> i:-1  (exclude last element)
    #                           i:nx-2 <--> i:-2  (exclude last 2 elements)


    # Radial derivative of vval[0]
    res4D[2:-2,:,:,:] = (
            DDX_C4(vval[0][2:-2,:,:,:], vval[0][1:-3,:,:,:], vval[0][3:-1,:,:,:], 
                                        vval[0][0:-4,:,:,:], vval[0][4:  ,:,:,:])
                                               / du.dr_n
         +
            vval[0][2:-2,:,:,:] / du.rxy[2:-2,:,np.newaxis,np.newaxis])
                           

    # Theta derivative of vval[1]
    # Add theta guard cells to the 4D array
    fval_wg = addzguard(vval[1], du)
     # slice indices in z:   [:] (original array) <--> [2:-2] (array with guard cells)
    res4D[:,:,:,:] += (
            DDX_C4(fval_wg[:,:,2:-2,:], fval_wg[:,:,1:-3,:], fval_wg[:,:,3:-1,:], 
                                        fval_wg[:,:,0:-4,:], fval_wg[:,:,4:  ,:])
                                               / (du.dtheta*du.rxy[:,:,np.newaxis,np.newaxis]))

    return res4D


# -----------------------------------------------------------


def advec_cyl_4D(vval, fval, du, order = 'VDDX_U1'):
#
# Evaluate advection term in cylindrical coordinates
#
# Inputs:
#        vval[*,*,*,*], fval[*,*,*,*] - 4D array of velocity (structure) and function values
#        du                           - GRIDreader object with all grid information
#
# Outputs:
#        advection (cyl coordinates) as 4D array, radial guard points excluded
#


    advec4D = np.zeros(fval.shape, fval.dtype)

    # For theta derivatives: add theta guard cells to the 4D array
    fval_wg = addzguard(fval, du)


    if order == 'VDDX_U1':
        
        advec4D[2:-2,:,:,:] = (
            # Radial component
            VDDX_U1(vval[0][2:-2,:,:,:],
                    fval[2:-2,:,:,:], fval[1:-3,:,:,:], fval[3:-1,:,:,:], 
                    fval[0:-4,:,:,:], fval[4:  ,:,:,:])
                    / du.dr_n
            +
            # Azimuthal component
            VDDX_U1(vval[1][2:-2,:,:,:],
                    fval_wg[2:-2,:,2:-2,:], fval_wg[2:-2,:,1:-3,:], fval_wg[2:-2,:,3:-1,:], 
                    fval_wg[2:-2,:,0:-4,:], fval_wg[2:-2,:,4:  ,:])
                    / (du.dtheta*du.rxy[2:-2,:,np.newaxis,np.newaxis])
            )

    if order == 'VDDX_C4':
        
        advec4D[2:-2,:,:,:] = (
            # Radial component
            VDDX_C4(vval[0][2:-2,:,:,:],
                    fval[2:-2,:,:,:], fval[1:-3,:,:,:], fval[3:-1,:,:,:], 
                    fval[0:-4,:,:,:], fval[4:  ,:,:,:])
                    / du.dr_n
            +
            # Azimuthal component
            VDDX_C4(vval[1][2:-2,:,:,:],
                    fval_wg[2:-2,:,2:-2,:], fval_wg[2:-2,:,1:-3,:], fval_wg[2:-2,:,3:-1,:], 
                    fval_wg[2:-2,:,0:-4,:], fval_wg[2:-2,:,4:  ,:])
                    / (du.dtheta*du.rxy[2:-2,:,np.newaxis,np.newaxis])
            )



    rvrfval = vval[0]*fval*du.rxy[:,:,np.newaxis,np.newaxis]
    vthfval = vval[1]*fval
    vthfval_wg = addzguard(vthfval, du)

    if order == 'VDDX_L1':

        advec4D[2:-2,:,:,:] = (
            #Radial derivative
            DDX_L1(rvrfval[2:-2,:,:,:], rvrfval[1:-3,:,:,:], rvrfval[3:-1,:,:,:], 
                    rvrfval[0:-4,:,:,:], rvrfval[4:  ,:,:,:])
                    / (du.dr_n*du.rxy[2:-2,:,np.newaxis,np.newaxis])

            +
            #Azimuthal derivative
            DDX_L1(vthfval_wg[2:-2,:,2:-2,:], vthfval_wg[2:-2,:,1:-3,:], vthfval_wg[2:-2,:,3:-1,:], 
                    vthfval_wg[2:-2,:,0:-4,:], vthfval_wg[2:-2,:,4:  ,:])
                    / (du.dtheta*du.rxy[2:-2,:,np.newaxis,np.newaxis])
            ) 
        

# TODO:  vpar
                           
    return advec4D

# -----------------------------------------------------------

def crossProd_4D(v1,v2):
#
# Evaluate cross product of two vectors
#
    import copy

#    # Initialize res4D as a list (r,theta,z) of 4D arrays of the same size as v1
#    res4D = copy.deepcopy(v1)

    res4D  = [ v1[1]*v2[2] - v1[2]*v2[1],  # r
               v1[2]*v2[0] - v1[0]*v2[2],  # theta
               v1[0]*v2[1] - v1[1]*v2[0] ] # z

    return res4D

# -----------------------------------------------------------

def dotProd_4D(v1,v2):
#
# Evaluate dot product of two vectors
#

#    # Initialize res4D
#    res4D = zeros(v1[0].shape, v1[0].dtype)

    res4D = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

    return res4D

# -----------------------------------------------------------
def dotProd2(v1,v2):
#
# Evaluate dot product of two vectors
#

#    # Initialize res4D
#    res4D = zeros(v1[0].shape, v1[0].dtype)

    res4D = v1[0]*v2[0] + v1[1]*v2[1]

    return res4D

# -----------------------------------------------------------

def ddt_4D(fval, du):
#
# Evaluate normalized time derivative for a 4D array
#
# Inputs:
#        fval[*,*,*,*] - 4D array
#        du            - GRIDreader object with all grid information
#
# Outputs:
#        time derivative of fval
#

    res4D = np.zeros(fval.shape, fval.dtype)

    res4D[:,:,:,2:-2] = (
        DDX_C4(fval[:,:,:,2:-2], fval[:,:,:,1:-3], fval[:,:,:,3:-1], 
                                 fval[:,:,:,0:-4], fval[:,:,:,4:  ])
                                               / du.dt_n)
                           
    return res4D

# -----------------------------------------------------------

def ddx_4D(fval, du, order='DDX_C4'):
#
# Evaluate d/dx (dimensionless)
#
# Inputs:
#        fval[*,*,*,*] - 4D array of function values
#        du            - GRIDreader object with all grid information
#
# Outputs:
#        d/dx fval in 4D (margin points excluded)
#

    # Initialize rad4D as an array of the same size as fval
    res4D = np.zeros(fval.shape, fval.dtype)

    # Radial/parallel derivatives are not calculated in guard points (2 on each size)

    # Cribsheet: slice indices  i:nx   <--> i:    (include last element)
    #                           i:nx-1 <--> i:-1  (exclude last element)
    #                           i:nx-2 <--> i:-2  (exclude last 2 elements)

    # Radial derivative of fval

    if order == 'DDX_C4':
        res4D[2:-2,:,:,:] = (
            DDX_C4(fval[2:-2,:,:,:], fval[1:-3,:,:,:], fval[3:-1,:,:,:], 
                                     fval[0:-4,:,:,:], fval[4:  ,:,:,:])
                                               / du.dr_n)
#        res4D[:2,:,:,:] = (
#            DDX_R1(fval[:2,:,:,:], fval[1:-3,:,:,:], fval[1:3,:,:,:], 
#                                     fval[0:-4,:,:,:], fval[4:  ,:,:,:])
#                                               / du.dr_n[2:4,:,newaxis,newaxis])
#        res4D[-2:,:,:,:] = (
#            DDX_L1(fval[-2:,:,:,:], fval[-3:-1,:,:,:], fval[1:3,:,:,:], 
#                                     fval[0:-4,:,:,:], fval[4:  ,:,:,:])
#                                               / du.dr_n[2:4,:,newaxis,newaxis])

    if order == 'DDX_C2':
        print('doing DDX_C2')
        res4D[2:-2,:,:,:] = (
            DDX_C2(fval[2:-2,:,:,:], fval[1:-3,:,:,:], fval[3:-1,:,:,:], 
                                     fval[0:-4,:,:,:], fval[4:  ,:,:,:])
                                               / du.dr_n)


    if order == 'DDX_R1':
        print('doing DDX_R1')
        res4D[2:-2,:,:,:] = (
            DDX_R1(fval[2:-2,:,:,:], fval[1:-3,:,:,:], fval[3:-1,:,:,:], 
                                     fval[:-4,:,:,:], fval[4:  ,:,:,:])
                                               / du.dr_n)

    return res4D

# -----------------------------------------------------------


def d2dx2_4D(fval, du, order='D2DX2_C4'):
#
# Evaluate d/dx (dimensionless)
#
# Inputs:
#        fval[*,*,*,*] - 4D array of function values
#        du            - GRIDreader object with all grid information
#
# Outputs:
#        d/dx fval in 4D (margin points excluded)
#

    # Initialize rad4D as an array of the same size as fval
    res4D = np.zeros(fval.shape, fval.dtype)

    # Radial/parallel derivatives are not calculated in guard points (2 on each size)

    # Cribsheet: slice indices  i:nx   <--> i:    (include last element)
    #                           i:nx-1 <--> i:-1  (exclude last element)
    #                           i:nx-2 <--> i:-2  (exclude last 2 elements)

    # Radial derivative of fval

    if order == 'D2DX2_C4':
        res4D[2:-2,:,:,:] = (
            D2DX2_C4(fval[2:-2,:,:,:], fval[1:-3,:,:,:], fval[3:-1,:,:,:], 
                                     fval[0:-4,:,:,:], fval[4:  ,:,:,:])
                                               / (du.dr_n)**2)

    if order == 'D2L1DR1':
        print('doing D2L1DR1')
        res4D[2:-2,:,:,:] = (
            D2L1DR1(fval[2:-2,:,:,:], fval[1:-3,:,:,:], fval[3:-1,:,:,:], 
                                     fval[0:-4,:,:,:], fval[4:  ,:,:,:])
                                               / (du.dr_n)**2)


    return res4D

# -----------------------------------------------------------

def Delp2perp_4D(fval, du, order='D2DX2_C4'):
#
# Evaluate d/dx (dimensionless)
#
# Inputs:
#        fval[*,*,*,*] - 4D array of function values
#        du            - GRIDreader object with all grid information
#
# Outputs:
#        d/dx fval in 4D (margin points excluded)
#

    # Initialize rad4D as an array of the same size as fval
    res4D = np.zeros(fval.shape, fval.dtype)


    # Radial/parallel derivatives are not calculated in guard points (2 on each size)

    # Cribsheet: slice indices  i:nx   <--> i:    (include last element)
    #                           i:nx-1 <--> i:-1  (exclude last element)
    #                           i:nx-2 <--> i:-2  (exclude last 2 elements)

    # Radial derivative of fval

     # Theta derivative of fval
    # Add theta guard cells to the 4D array
    fval_wg = addzguard(fval, du)
     # slice indices in z:   [:] (original array) <--> [2:-2] (array with guard cells)


    if order == 'D2DX2_C4':
        res4D[2:-2,:,:,:] = (
            D2DX2_C4(fval[2:-2,:,:,:], fval[1:-3,:,:,:], fval[3:-1,:,:,:], 
                                     fval[0:-4,:,:,:], fval[4:,:,:,:])/ du.dr_n**2
            +1./(du.rxy[2:-2,:,np.newaxis,np.newaxis])*DDX_C4(fval[2:-2,:,:,:], fval[1:-3,:,:,:], fval[3:-1,:,:,:], 
                                     fval[0:-4,:,:,:], fval[4:,:,:,:])/ (du.dr_n)
            + D2DX2_C4(fval_wg[2:-2,:,2:-2,:], fval_wg[2:-2,:,1:-3,:], fval_wg[2:-2,:,3:-1,:], 
                                        fval_wg[2:-2,:,0:-4,:], fval_wg[2:-2,:,4:  ,:])/ ((du.dtheta**2)*(du.rxy[2:-2,:,np.newaxis,np.newaxis]**2)))

                
    return res4D

# -----------------------------------------------------------

def ddy_4D_L1(fval, du):
#
# Evaluate d/dy (dimensionless)
#
# Inputs:
#        fval[*,*,*,*] - 4D array of function values
#        du            - GRIDreader object with all grid information
#
# Outputs:
#        d/dy fval in 4D (margin points excluded)
#

    # Initialize rad4D as an array of the same size as fval
    res4D = np.zeros(fval.shape, fval.dtype)

    # Parallel derivatives are not calculated in guard points (2 on each size)

    # Cribsheet: slice indices  i:nx   <--> i:    (include last element)
    #                           i:nx-1 <--> i:-1  (exclude last element)
    #                           i:nx-2 <--> i:-2  (exclude last 2 elements)

    # Parallel derivative of fval
    res4D[:,2:-2,:,:] = (
            DDX_L1(fval[:,2:-2,:,:], fval[:,1:-3,:,:], fval[:,3:-1,:,:], 
                                     fval[:,0:-4,:,:], fval[:,4:  ,:,:])
                                               / du.dpar_n)

    return res4D
# -----------------------------------------------------------

def ddy_4D_R1(fval, du):
#
# Evaluate d/dy (dimensionless)
#
# Inputs:
#        fval[*,*,*,*] - 4D array of function values
#        du            - GRIDreader object with all grid information
#
# Outputs:
#        d/dy fval in 4D (margin points excluded)
#

    # Initialize rad4D as an array of the same size as fval
    res4D = np.zeros(fval.shape, fval.dtype)

    # Parallel derivatives are not calculated in guard points (2 on each size)

    # Cribsheet: slice indices  i:nx   <--> i:    (include last element)
    #                           i:nx-1 <--> i:-1  (exclude last element)
    #                           i:nx-2 <--> i:-2  (exclude last 2 elements)

    # Parallel derivative of fval
    res4D[:,2:-2,:,:] = (
            DDX_R1(fval[:,2:-2,:,:], fval[:,1:-3,:,:], fval[:,3:-1,:,:], 
                                     fval[:,0:-4,:,:], fval[:,4:  ,:,:])
                                               / du.dpar_n)

    return res4D

# -----------------------------------------------------------

def ddz_4D(fval, du):
#
# Evaluate d/dz (dimensionless)
#
# Inputs:
#        fval[*,*,*,*] - 4D array of function values
#        du            - GRIDreader object with all grid information
#
# Outputs:
#        d/dz fval in 4D (all points)
#

    # Initialize rad4D as an array of the same size as fval
    res4D = np.zeros(fval.shape, fval.dtype)

    # Radial/parallel derivatives are not calculated in guard points (2 on each size)

    # Cribsheet: slice indices  i:nx   <--> i:    (include last element)
    #                           i:nx-1 <--> i:-1  (exclude last element)
    #                           i:nx-2 <--> i:-2  (exclude last 2 elements)


    # Theta derivative of fval
    # Add theta guard cells to the 4D array
    fval_wg = addzguard(fval, du)
    # slice indices in z:   [:] (original array) <--> [2:-2] (array with guard cells)

    res4D[:,:,:,:] = (
            DDX_C4(fval_wg[:,:,2:-2,:], fval_wg[:,:,1:-3,:], fval_wg[:,:,3:-1,:], 
                                        fval_wg[:,:,0:-4,:], fval_wg[:,:,4:  ,:])
                                               / du.dtheta)

    return res4D

# -----------------------------------------------------------

def radial_Average(fval,du,order = 'trapezoid'):
    
    if order == 'trapezoid':
        #Trapezoid Rule
        res2D = np.zeros(fval.shape, fval.dtype)

        for ix in range(1,fval.shape[0]):
            res2D[ix,:] = fval[0,:]+fval[ix,:]
            res2D[ix,:] += 2*fval[1:ix,:].sum(0)
            res2D[ix,:] /= 2.0
            res2D[ix,:] *= du.dr_n


    if order == 'rectangle':
        #Simple rectangle rule
        res2D = np.zeros(fval.shape, fval.dtype)

        for ix in range(1,fval.shape[0]):
            res2D[ix,:] = fval[1:ix+1,:].sum(0)*du.dr_n


    return res2D

# -----------------------------------------------------------

        
def deriv_full(fval, h):
#
# Evaluate dfval/dx using C2 in the internal interval points and forward/backward C1 on the edges 
#
# Inputs:
#        fval[:]  - function values
#        h        - step size
#
    res = fval*0.
    res[1:-1] = (fval[2:] - fval[:-2])*0.5
    res[0]    = (fval[1] - fval[0])
    res[-1]   = (fval[-1] - fval[-2])
    return res/h

# -----------------------------------------------------------
# Vector differential operators for symbolic operations
# -----------------------------------------------------------
import sympy

def Vector(v):
    """ Return sympy matrix of 3x1 elements with values v[0], v[1], v[2]"""
    return sympy.Matrix(3,1, v)

def CrossProd(av, bv):
    """ Return the cross product of two vectors"""
    g1 = av[1]*bv[2] - av[2]*bv[1]
    g2 = av[2]*bv[0] - av[0]*bv[2]
    g3 = av[0]*bv[1] - av[1]*bv[0]
    return sympy.Matrix(3,1, [g1, g2, g3])

def DotProd(av, bv):
    """ Return the cross product of two vectors"""
    return av.dot(bv)

def Grad(f, x, metric='cart'):
    """Return the gradient of f in Cartesian (metric='cart') or 
    Cylindrical (metric='cyl') coordinates, as a Sympy matrix of functions
    Input:  x -- list of coordinates [x,y,z]/[r,theta,z] """

    g = [0,0,0]
    if metric == 'cart':
        for i in range(3):
            g[i] = sympy.diff(f, x[i])
    elif metric == 'cyl':
        g[0] = sympy.diff(f, x[0])       # d/dr
        g[1] = sympy.diff(f, x[1])/x[0]  # 1/r d/dtheta
        g[2] = sympy.diff(f, x[2])       # d/dz
    else:
        print("Wrong agrument of Grad function!")
        return 0
        
    return sympy.Matrix(3,1, g)

def GradPerp(f, x, metric='cart'):
    """Return the perpendicular (to z) part of the gradient of f in
    Cartesian (metric='cart') or Cylindrical (metric='cyl') coordinates, 
    as a Sympy matrix of functions
    Input:  x -- list of coordinates [x,y,z]/[r,theta,z] """

    # GradPerp = Grad(f) - d/dz f,  both in Cartesian and Cylindrical
    return Grad(f, x, metric) - sympy.Matrix(3,1, [0,0,sympy.diff(f, x[2])])



def AGradB(av, bv, x, metric='cart'):
    """ Return the vector (A.Grad)B in Cartesian (metric='cart') or 
Cylindrical (metric='cyl') coordinates, as a Sympy matrix of functions
Input:  x -- list of coordinates [x,y,z]/[r,theta,z] """

    g = [0,0,0]
    if metric == 'cart':
        for j in range(3):
            for i in range(3):
                g[j] = g[j] + av[i]*sympy.diff(bv[j], x[i])

    elif metric == 'cyl':
        g[0] = (av[0]*sympy.diff(bv[0], x[0]) +
                av[1]*sympy.diff(bv[0], x[1])/x[0] + 
                av[2]*sympy.diff(bv[0], x[2]) -
                av[1]*bv[1]/x[0])

        g[1] = (av[0]*sympy.diff(bv[1], x[0]) +
                av[1]*sympy.diff(bv[1], x[1])/x[0] + 
                av[2]*sympy.diff(bv[1], x[2]) +
                av[1]*bv[0]/x[0])

        g[2] = (av[0]*sympy.diff(bv[2], x[0]) +
                av[1]*sympy.diff(bv[2], x[1])/x[0] + 
                av[2]*sympy.diff(bv[2], x[2]))

    else:
        print("Wrong agrument of AGradB function!")
        return 0
        
    return sympy.Matrix(3,1, g)

def Div(v, x, metric='cart'):
    """Return the divergence of v in Cartesian (metric='cart') or 
    Cylindrical (metric='cyl') coordinates
    Input:  x -- list of coordinates [x,y,z]/[r,theta,z] """

    if metric == 'cart':
        g = 0
        for i in range(3):
            g += sympy.diff(v[i], x[i])
    elif metric == 'cyl':
        g = (sympy.diff(x[0]*v[0], x[0])/x[0] +   # 1/r d/dr (r vr)
             sympy.diff(v[1], x[1])/x[0] +        # 1/r d/dtheta vtheta
             sympy.diff(v[2], x[2]))              # d/dz vz
    else:
        print("Wrong agrument of Div function!")
        return 0
        
    return g

def DivPerp(v, x, metric='cart'):
    """Return the perpendicular (to z) part of divergence of v in
    Cartesian (metric='cart') or Cylindrical (metric='cyl')
    coordinates Input: x -- list of coordinates [x,y,z]/[r,theta,z]
    """
    # DivPerp = Div(v) - d/dz vz,  both in Cartesian and Cylindrical
    return Div(v, x, metric) - sympy.diff(v[2], x[2])

def Delp2(f, x, metric='cart'):
    """Return the Laplacian of f in Cartesian (metric='cart') or 
    Cylindrical (metric='cyl') coordinates
    Input:  x -- list of coordinates [x,y,z]/[r,theta,z] """

    return Div(Grad(f, x, metric), x, metric)

def Delp2Perp(f, x, metric='cart'):
    """Return the perpendicular (to z) part of Laplacian of f in Cartesian
    (metric='cart') or Cylindrical (metric='cyl') coordinates 
    Input: x -- list of coordinates [x,y,z]/[r,theta,z] """

    return Delp2(f, x, metric) - sympy.diff(f, x[0], 2)  # = Delp2 - d^2/x^2 for Cart and Cyl

# -----------------------------------------------------------
# Integrals of a Gaussian  -- used for phi0 profile in KH instability cases
from scipy.special import erf

def int0_N(x,w):
    """Gaussian N(x,w)"""
    return np.exp(-(x*x)/(w*w)) / (np.sqrt(np.pi)*w)

def int1_N(x,w):
    """Int N(x,w) dx"""
    return 0.5*erf(x/w)

def int2_N(x,w):
    """Int Int N(x,w) dx"""
    return 0.5* ( np.exp(-(x*x)/(w*w))*w / np.sqrt(np.pi)
                + x*erf(x/w) )

def int3_N(x,w):
    """Int Int Int N(x,w) dx"""
    return 0.125* ( 2.*np.exp(-(x*x)/(w*w))*w*x / np.sqrt(np.pi)
                  + (w*w+2.*x*x)*erf(x/w) )

# -----------------------------------------------------------
def TriDiagDet(a,b,c):
    """
    Calculate the determinant of a tridiagonal complex matrix using the relation

    det M_n = a_n * det M_(n-1) - b_(n-1)*c_(n-1) * det M_(n-2)


#         a_1    b_1    ...
#         c_1    ...    ... 
#    det( ...    ...    ...   ) =
#         ...    ...   b_(n-1)
#         ...  c_(n-1)  a_n
#
#
#         a_n  -b_(n-1) c_(n-1)            a_2  -b_1 c_1       a1  0
#       (                      ) * ... * (               ) * (       )
#          1           0                    1       0           1  0

    Input:
        a,b,c -- diagonals of the matrix. Respective sizes: N, N-1, N-1 elements
#        M -- tridiagonal matrix (2D), full storage (not sparse)

    """

    N = a.shape[0]
    D = np.zeros(N, dtype=complex)

    D[0] = a[0]   # a_1
    D[1] = a[0]*a[1] - b[0]*c[0]    # det(A_{1,2}) = a_1*a_2 - c_1*b_1

    for n in range(2,N):
        D[n] = a[n]*D[n-1] - b[n-1]*c[n-1]*D[n-2]

    return D[N-1]


#    M2_0 = array([[a[0], 0.], [1., 0.]])
#    
#    M_cum = M2_0
#    for i in xrange(1,N):
#        M2_i = array([[a[i], -b[i-1]*c[i-1]], [1., 0.]])
#
#        M_cum = dot(M2_i, M_cum)
#
#    return M_cum[0,0]

# -----------------------------------------------------------


if __name__ == '__main__':
    pass

