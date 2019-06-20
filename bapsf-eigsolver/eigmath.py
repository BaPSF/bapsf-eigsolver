#!/usr/bin/env /u/local/apps/python/2.6.1/bin/python
"""
Define differential operators for use in eigsolver
-----------------------------------------------------------

"""

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
# Integrals of a Gaussian  
from scipy.special import erf

def int0_N(x,w):
    """Gaussian N(x,w)"""
    return exp(-(x*x)/(w*w)) / (sqrt(pi)*w)

def int1_N(x,w):
    """Int N(x,w) dx"""
    return 0.5*erf(x/w)

def int2_N(x,w):
    """Int Int N(x,w) dx"""
    return 0.5* ( exp(-(x*x)/(w*w))*w / sqrt(pi)
                + x*erf(x/w) )

def int3_N(x,w):
    """Int Int Int N(x,w) dx"""
    return 0.125* ( 2.*exp(-(x*x)/(w*w))*w*x / sqrt(pi)
                  + (w*w+2.*x*x)*erf(x/w) )



# -----------------------------------------------------------
if __name__ == '__main__':
    pass

