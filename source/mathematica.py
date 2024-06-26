import numpy as np
import numba
from numba import njit
import sympy as sp
from sympy.physics.wigner import wigner_3j
from param import *


def Al123(ell1, ell2, ell3):
    if ell1==ell2 and ell2==ell3:
        return -1.0
    else:
        return float((wigner_3j(ell1, ell2, ell3, 0,1,-1)+wigner_3j(ell1, ell2, ell3, 0,-1,1))\
            /wigner_3j(ell1, ell2, ell3, 0,0,0))

############################################################################# window fct
@numba.extending.overload(np.gradient)
def np_gradient(f):
    def np_gradient_impl(f):
        out = np.empty_like(f, np.float64)
        out[1:-1] = (f[2:] - f[:-2]) / 2.0
        out[0] = f[1] - f[0]
        out[-1] = f[-1] - f[-2]
        return out

    return np_gradient_impl

@njit
def W(x, z0, Dz, normW=1):
    
    res=(0.5+0.5*np.tanh( (x-(z0-Dz/2))/bb))*(0.5+0.5*np.tanh( (-x+(z0+Dz/2))/bb))
    return res/normW

# sp.diff((0.5+0.5*sp.tanh( (x-(z0-Dz/2))/bb))*(0.5+0.5*sp.tanh( (-x+(z0+Dz/2))/bb)) , x, 1)
@njit
def d1W_(x, z0, Dz, normW=1):
    res = -0.5*(1 - np.tanh((Dz/2 - x + z0)/bb)**2)*(0.5*np.tanh((Dz/2 + x - z0)/bb) + 0.5)/bb +\
            0.5*(1 - np.tanh((Dz/2 + x - z0)/bb)**2)*(0.5*np.tanh((Dz/2 - x + z0)/bb) + 0.5)/bb

    return res/normW

@njit
def d2W_(x, z0, Dz, normW=1):
    res = 0.5*((np.tanh((Dz/2 - x + z0)/bb) + 1)*(np.tanh((Dz/2 + x - z0)/bb)**2 - 1)*np.tanh((Dz/2 + x - z0)/bb) \
            + (np.tanh((Dz/2 - x + z0)/bb)**2 - 1)*(np.tanh((Dz/2 + x - z0)/bb) + 1)*np.tanh((Dz/2 - x + z0)/bb) \
            - (np.tanh((Dz/2 - x + z0)/bb)**2 - 1)*(np.tanh((Dz/2 + x - z0)/bb)**2 - 1))/bb**2
    return res/normW

@njit
def d3W_(x, z0, Dz, normW=1):
    res = (-0.5*(np.tanh((Dz/2 - x + z0)/bb) + 1)*(np.tanh((Dz/2 + x - z0)/bb)**2 - 1)*(3.0*np.tanh((Dz/2 + x - z0)/bb)**2 - 1.0) \
            + 0.5*(np.tanh((Dz/2 - x + z0)/bb)**2 - 1)*(3.0*np.tanh((Dz/2 - x + z0)/bb)**2 - 1.0)*(np.tanh((Dz/2 + x - z0)/bb) + 1) \
            - 1.5*(np.tanh((Dz/2 - x + z0)/bb)**2 - 1)*(np.tanh((Dz/2 + x - z0)/bb)**2 - 1)*np.tanh((Dz/2 - x + z0)/bb) \
            + 1.5*(np.tanh((Dz/2 - x + z0)/bb)**2 - 1)*(np.tanh((Dz/2 + x - z0)/bb)**2 - 1)*np.tanh((Dz/2 + x - z0)/bb))/bb**3
    return res/normW

@njit
def d4W_(x, z0, Dz, normW=1):
    res = (0.5*(np.tanh((Dz/2 - x + z0)/bb) + 1)*(np.tanh((Dz/2 + x - z0)/bb)**2 - 1)*(12.0*np.tanh((Dz/2 + x - z0)/bb)**2 - 8.0)*np.tanh((Dz/2 + x - z0)/bb) \
            - 2.0*(np.tanh((Dz/2 - x + z0)/bb)**2 - 1)*(3.0*np.tanh((Dz/2 - x + z0)/bb)**2 - 1.0)*(np.tanh((Dz/2 + x - z0)/bb)**2 - 1) \
            + 0.5*(np.tanh((Dz/2 - x + z0)/bb)**2 - 1)*(12.0*np.tanh((Dz/2 - x + z0)/bb)**2 - 8.0)*(np.tanh((Dz/2 + x - z0)/bb) + 1)*np.tanh((Dz/2 - x + z0)/bb) \
            - 2.0*(np.tanh((Dz/2 - x + z0)/bb)**2 - 1)*(np.tanh((Dz/2 + x - z0)/bb)**2 - 1)*(3.0*np.tanh((Dz/2 + x - z0)/bb)**2 - 1.0) \
            + 6.0*(np.tanh((Dz/2 - x + z0)/bb)**2 - 1)*(np.tanh((Dz/2 + x - z0)/bb)**2 - 1)*np.tanh((Dz/2 - x + z0)/bb)*np.tanh((Dz/2 + x - z0)/bb))/bb**4
    return res/normW

@njit
def d5W_(x, z0, Dz, normW=1):
    res = (-0.5*(np.tanh((Dz/2 - x + z0)/bb) + 1)*(np.tanh((Dz/2 + x - z0)/bb)**2 - 1)*(8.0*(np.tanh((Dz/2 + x - z0)/bb)**2 - 1)**2 \
            + 44.0*(np.tanh((Dz/2 + x - z0)/bb)**2 - 1)*np.tanh((Dz/2 + x - z0)/bb)**2 + 8.0*np.tanh((Dz/2 + x - z0)/bb)**4) \
            + 10.0*(np.tanh((Dz/2 - x + z0)/bb)**2 - 1)*(3.0*np.tanh((Dz/2 - x + z0)/bb)**2 - 1.0)*(np.tanh((Dz/2 + x - z0)/bb)**2 - 1)*np.tanh((Dz/2 + x - z0)/bb) \
            - 2.5*(np.tanh((Dz/2 - x + z0)/bb)**2 - 1)*(12.0*np.tanh((Dz/2 - x + z0)/bb)**2 - 8.0)*(np.tanh((Dz/2 + x - z0)/bb)**2 - 1)*np.tanh((Dz/2 - x + z0)/bb) \
            + 0.5*(np.tanh((Dz/2 - x + z0)/bb)**2 - 1)*(np.tanh((Dz/2 + x - z0)/bb) + 1)*(8.0*(np.tanh((Dz/2 - x + z0)/bb)**2 - 1)**2 \
            + 44.0*(np.tanh((Dz/2 - x + z0)/bb)**2 - 1)*np.tanh((Dz/2 - x + z0)/bb)**2 + 8.0*np.tanh((Dz/2 - x + z0)/bb)**4) \
            - 10.0*(np.tanh((Dz/2 - x + z0)/bb)**2 - 1)*(np.tanh((Dz/2 + x - z0)/bb)**2 - 1)*(3.0*np.tanh((Dz/2 + x - z0)/bb)**2 - 1.0)*np.tanh((Dz/2 - x + z0)/bb) \
            + 2.5*(np.tanh((Dz/2 - x + z0)/bb)**2 - 1)*(np.tanh((Dz/2 + x - z0)/bb)**2 - 1)*(12.0*np.tanh((Dz/2 + x - z0)/bb)**2 - 8.0)*np.tanh((Dz/2 + x - z0)/bb))/bb**5
    return res/normW

#sp.diff(x**3*sp.diff(D*f*W , x, 2), x, 3)
def mathcalB(which, lterm, qterm, time_dict, z0, Dz, normW):

    sp.init_printing(pretty_print=False)
    def display_no_args(expr):
        functions = expr.atoms(sp.Function)
        reps = {}
    
        for fun in functions:
            reps[fun] = sp.Symbol(fun.name)
    
        return(expr.subs(reps))
    
    def substitute(expr):
        expr = expr.subs(sp.diff(D, x, 5), d5D)
        expr = expr.subs(sp.diff(D, x, 4), d4D)
        expr = expr.subs(sp.diff(D, x, 3), d3D)
        expr = expr.subs(sp.diff(D, x, 2), d2D)
        expr = expr.subs(sp.diff(D, x, 1), d1D)
        
        expr = expr.subs(sp.diff(f, x, 5), d5f)
        expr = expr.subs(sp.diff(f, x, 4), d4f)
        expr = expr.subs(sp.diff(f, x, 3), d3f)
        expr = expr.subs(sp.diff(f, x, 2), d2f)
        expr = expr.subs(sp.diff(f, x, 1), d1f)
        
        expr = expr.subs(sp.diff(W, x, 5), d5W)
        expr = expr.subs(sp.diff(W, x, 4), d4W)
        expr = expr.subs(sp.diff(W, x, 3), d3W)
        expr = expr.subs(sp.diff(W, x, 2), d2W)
        expr = expr.subs(sp.diff(W, x, 1), d1W)

        expr = expr.subs(sp.diff(H, x, 4), d4H)
        expr = expr.subs(sp.diff(H, x, 3), d3H)
        expr = expr.subs(sp.diff(H, x, 2), d2H)
        expr = expr.subs(sp.diff(H, x, 1), d1H)
 
        expr = expr.subs(sp.diff(R, x, 4), d4R)
        expr = expr.subs(sp.diff(R, x, 3), d3R)
        expr = expr.subs(sp.diff(R, x, 2), d2R)
        expr = expr.subs(sp.diff(R, x, 1), d1R)

        expr = expr.subs(sp.diff(a, x, 3), d3a)
        expr = expr.subs(sp.diff(a, x, 2), d2a)
        expr = expr.subs(sp.diff(a, x, 1), d1a)
        
        return display_no_args(expr)

    time_dict['d1f'] = np.gradient(time_dict['fr'], time_dict['r_list'] , edge_order=2)
    time_dict['d2f'] = np.gradient(time_dict['d1f'], time_dict['r_list'], edge_order=2)
    time_dict['d3f'] = np.gradient(time_dict['d2f'], time_dict['r_list'], edge_order=2)
    time_dict['d4f'] = np.gradient(time_dict['d3f'], time_dict['r_list'], edge_order=2)
    time_dict['d5f'] = np.gradient(time_dict['d4f'], time_dict['r_list'], edge_order=2)

    time_dict['d1D'] = np.gradient(time_dict['Dr'], time_dict['r_list'] , edge_order=2)
    time_dict['d2D'] = np.gradient(time_dict['d1D'], time_dict['r_list'], edge_order=2)
    time_dict['d3D'] = np.gradient(time_dict['d2D'], time_dict['r_list'], edge_order=2)
    time_dict['d4D'] = np.gradient(time_dict['d3D'], time_dict['r_list'], edge_order=2)
    time_dict['d5D'] = np.gradient(time_dict['d4D'], time_dict['r_list'], edge_order=2)

    time_dict['d1R'] = np.gradient(time_dict['mathcalR'], time_dict['r_list'] , edge_order=2)
    time_dict['d2R'] = np.gradient(time_dict['d1R'], time_dict['r_list'], edge_order=2)
    time_dict['d3R'] = np.gradient(time_dict['d2R'], time_dict['r_list'], edge_order=2)
    time_dict['d4R'] = np.gradient(time_dict['d3R'], time_dict['r_list'], edge_order=2)

    time_dict['d1H'] = np.gradient(time_dict['Hr'], time_dict['r_list'] , edge_order=2)
    time_dict['d2H'] = np.gradient(time_dict['d1H'], time_dict['r_list'], edge_order=2)
    time_dict['d3H'] = np.gradient(time_dict['d2H'], time_dict['r_list'], edge_order=2)
    time_dict['d4H'] = np.gradient(time_dict['d3H'], time_dict['r_list'], edge_order=2)

    time_dict['d1a'] = np.gradient(time_dict['ar'], time_dict['r_list'] , edge_order=2)
    time_dict['d2a'] = np.gradient(time_dict['d1a'], time_dict['r_list'], edge_order=2)
    time_dict['d3a'] = np.gradient(time_dict['d2a'], time_dict['r_list'], edge_order=2)


    time_dict['d1W'] = d1W_(time_dict['r_list'], z0, Dz, normW)
    time_dict['d2W'] = d2W_(time_dict['r_list'], z0, Dz, normW)
    time_dict['d3W'] = d3W_(time_dict['r_list'], z0, Dz, normW)
    time_dict['d4W'] = d4W_(time_dict['r_list'], z0, Dz, normW)
    time_dict['d5W'] = d5W_(time_dict['r_list'], z0, Dz, normW)

    x = sp.symbols("time_dict['r_list']")
    W = sp.Function("time_dict['Wr']")(x)
    f = sp.Function("time_dict['fr']")(x)
    D = sp.Function("time_dict['Dr']")(x)
    R = sp.Function("time_dict['mathcalR']")(x)
    a = sp.Function("time_dict['ar']")(x)
    H = sp.Function("time_dict['Hr']")(x)
    
    d1D = sp.Function("time_dict['d1D']")(x)
    d2D = sp.Function("time_dict['d2D']")(x)
    d3D = sp.Function("time_dict['d3D']")(x)
    d4D = sp.Function("time_dict['d4D']")(x)
    d5D = sp.Function("time_dict['d5D']")(x)
    
    d1W = sp.Function("time_dict['d1W']")(x)
    d2W = sp.Function("time_dict['d2W']")(x)
    d3W = sp.Function("time_dict['d3W']")(x)
    d4W = sp.Function("time_dict['d4W']")(x)
    d5W = sp.Function("time_dict['d5W']")(x)
    
    d1f = sp.Function("time_dict['d1f']")(x)
    d2f = sp.Function("time_dict['d2f']")(x)
    d3f = sp.Function("time_dict['d3f']")(x)
    d4f = sp.Function("time_dict['d4f']")(x)
    d5f = sp.Function("time_dict['d5f']")(x)

    d1H = sp.Function("time_dict['d1H']")(x)
    d2H = sp.Function("time_dict['d2H']")(x)
    d3H = sp.Function("time_dict['d3H']")(x)
    d4H = sp.Function("time_dict['d4H']")(x)
    
    d1R = sp.Function("time_dict['d1R']")(x)
    d2R = sp.Function("time_dict['d2R']")(x)
    d3R = sp.Function("time_dict['d3R']")(x)
    d4R = sp.Function("time_dict['d4R']")(x)

    d1a = sp.Function("time_dict['d1a']")(x)
    d2a = sp.Function("time_dict['d2a']")(x)
    d3a = sp.Function("time_dict['d3a']")(x)

    if lterm=='density':
        B = W*D
    elif lterm=='rsd':
        B = -sp.diff(D*f*W , x, 2)
    elif lterm=='pot':
        B = W*D*((1.-R)/a+3*f*H**2)
    elif lterm=='doppler':
        B = sp.diff(H*D*W*f*R , x, 1)
    else:
        print('no code for {}'.format(lterm))

    if qterm==4:
        expr = sp.diff(x**3*B, x, 3)
    elif qterm==3:
        expr = sp.diff(x**2*B, x, 2)
    elif qterm==2:
        expr = sp.diff(x**1*B, x, 1)
    else:
        expr = B

    return eval(str(substitute(expr)))
    

@njit
def tmin_fct(ell, nu_p):
    if (ell > 60):
        tmin = ((1. - np.exp(-0.08405923801793776*ell))*ell**1.0388189966482335)/(\
    16.552260860083162 + ell**1.0388189966482335) + 1./72.* (-(((1. -\
           np.exp(-0.08405923801793776*ell))*ell**1.0388189966482335)/(\
        16.552260860083162+ ell**1.0388189966482335)) + ((1. - np.exp(-0.03269491513404876*ell))*ell**1.0606484271153198)/(\
       86.60472131391394+ell**1.0606484271153198))*np.abs(nu_p.imag)
    else:
        tmin = 0.026189266734847335- 0.04199333649354753*ell + \
               0.01813725076906472*ell**2 - 0.0019512662766506912*ell**3 + \
               0.00011476285973931163  *ell**4 - 4.132495310006262e-6 *ell**5 + \
               9.321216021016041e-8    *ell**6 - 1.2840836892275476e-9*ell**7 + \
               9.874351126449866e-12 *ell**8 - 3.247034458438453e-14*ell**9 +\
               1/91512 *(-4223 *(0.026189266734847335 - 0.04199333649354753*ell + \
               0.01813725076906472*ell**2 - 0.0019512662766506912*ell**3 + \
               0.00011476285973931163 *ell**4 - 4.132495310006262e-6*ell**5 + \
               9.321216021016041e-8 *ell**6 - 1.2840836892275476e-9*ell**7 + \
               9.874351126449866e-12*ell**8 - \
               3.247034458438453e-14*ell**9) - \
            961.*(0.0050534423514964006 - 0.004245361441903382*ell + \
               0.0009644735508629553   *ell**2 - \
               0.000029194973960888548 *ell**3 - \
               1.197268126576586e-7   *ell**4 + 3.9232441693781885e-8*ell**5 - \
               1.3806236786152843e-9  *ell**6 + \
               2.380296810916152e-11  *ell**7 - \
               2.105287890873389e-13  *ell**8 + \
               7.627228092016026e-16  *ell**9) + \
            5184.* (0.014502978209351904- 0.01218174975881159*ell + \
               0.002817970220966578*ell**2 - 0.00011942831975390713*ell**3 + \
               1.223432213234367e-6  *ell**4 + 7.921224590247682e-8 *ell**5 - \
               3.5781997384264233e-9 *ell**6 + \
               6.634225862490053e-11 *ell**7 - \
               6.057230587166174e-13 *ell**8 + \
               2.230575708513619e-15 *ell**9)) *np.abs(nu_p.imag) + \
            1./91512.* (41.* (0.026189266734847335 - 0.04199333649354753*ell + \
               0.01813725076906472*ell**2 - 0.0019512662766506912*ell**3 + \
               0.00011476285973931163*ell**4 - 4.132495310006262e-6 *ell**5 + \
               9.321216021016041e-8  *ell**6 - 1.2840836892275476e-9 *ell**7 + \
               9.874351126449866e-12 *ell**8 - \
               3.247034458438453e-14 *ell**9) + \
            31.* (0.0050534423514964006 - 0.004245361441903382*ell + \
               0.0009644735508629553* ell**2 - \
               0.000029194973960888548* ell**3 - \
               1.197268126576586e-7  *ell**4 + 3.9232441693781885e-8 *ell**5 - \
               1.3806236786152843e-9  *ell**6 + \
               2.380296810916152e-11 *ell**7 - \
               2.105287890873389e-13 *ell**8 + \
               7.627228092016026e-16 *ell**9) - \
            72.* (0.014502978209351904 - 0.01218174975881159*ell + \
               0.002817970220966578*ell**2 - 0.00011942831975390713*ell**3 + \
               1.223432213234367e-6  *ell**4 + 7.921224590247682e-8 *ell**5 - \
               3.5781997384264233e-9  *ell**6 + \
               6.634225862490053e-11 *ell**7 - \
               6.057230587166174e-13 *ell**8 + \
               2.230575708513619e-15 *ell**9)) *np.abs(nu_p.imag)**2
    return tmin


@njit
def myhyp21_basic(a1, a2, b1, z):
    '''
    This function is an implementation of the serie representation of the 2F1 function: eq (A.2) of 1705.05022
    '''

    par1=1+0.j
    s=0.j
    i=0.j
    eps=1+0.j
    while(eps.real>1e-10):
        sold=s
        s+=par1
        par1*=(a1+i)*(a2+i)/(b1+i)/(i+1)*z
        eps=np.absolute(sold/s-1.)
        i+=1
    return s

@njit
def mygamma(z):
    q0 = 75122.6331530 + 0.0j 
    q1 = 80916.6278952 + 0.0j 
    q2 = 36308.2951477 + 0.0j 
    q3 = 8687.24529705 + 0.0j 
    q4 = 1168.92649479 + 0.0j 
    q5 = 83.8676043424 + 0.0j 
    q6 = 2.50662827511 + 0.0j
    if (z.real >= 0):  
        p1 = (q0 + q1*z + q2*z**2 + q3*z**3 + q4*z**4 + q5*z**5 +\
           q6*z**6)/(z*(z + 1.) *(z + 2.) *(z + 3.) *(z + 4.) *(z + 5.)* (z + 6.))
        result = p1* (z + 5.5)**(z + 0.5) *np.exp(-z - 5.5)
    else:
        p1 = (q0 + q1* (1. - z) + q2* (1. - z)**2 + q3 *(1. - z)**3 +\
        q4* (1. - z)**4 + q5* (1. - z)**5 +\
        q6 *(1. - z)**6)/((1. - z)* (2. - z)* (3. - z)* (4. - z) *(5. - z) *(6. -\
         z)* (7. - z)) 
        
        p2 = p1 *(1. - z + 5.5)**(1. - z + 0.5)*np.exp(-1. + z - 5.5) 
        result = np.pi/(np.sin(np.pi*z)*p2);
    return result



@njit
def mygammaRatio(z1, z2):

    q0 = 0.0075122633153 + 0.0j
    q1 = 0.0809166278952 + 0.0j
    q2 = 0.363082951477 + 0.0j
    q3 = 0.868724529705 + 0.0j
    q4 = 1.16892649479 + 0.0j
    q5 = 0.838676043424 + 0.0j
    q6 = 0.250662827511 + 0.0j

    result = 1.0 + 0.0j
    for i in range(1, 8):
        result = result*(z2 + i - 1)/(z1 + i - 1);

    result = result*np.exp(z2 - z1)
    p1 = (z1 + 0.5)*np.log(z1 + 5.5)-(z2 + 0.5)*np.log(z2 + 5.5)
    result = result*np.exp(p1);
    z1t = z1/10.0
    z2t = z2/10.0

    p1 = (q0 + q1* z1t + q2 *z1t**2 + q3*z1t**3 + q4*z1t**4 + q5*z1t**5 + q6*z1t**6);
    p2 = (q0 + q1 *z2t + q2 *z2t**2 + q3*z2t**3 + q4*z2t**4 + q5*z2t**5 + q6*z2t**6);

    result = result*p1/p2;
    return result

@njit
def Il(nu_p, z, ell):
    '''
    Simple implementation of I_Assassi_Simonovic_Zaldarriaga defined in eq (2.19) of 1705.05022
    '''

    a1, a2, b1 = (nu_p-1.)/2., ell+nu_p/2., ell+3./2.
    res = np.pi**2*2.**(nu_p-1.)*mygammaRatio(ell+nu_p/2.,ell+3./2.)\
    /mygamma((3.-nu_p)/2.)*z**ell*myhyp21_basic(a1, a2, b1, z**2)
    return res

@njit
def hyp21(nu_p, z, ell):
    '''
    Efficient implementation of I_Assassi_Simonovic_Zaldarriaga, see appendix B. of 1705.05022
    '''

    if z<0.5:
        res=Il(nu_p, z, ell)
    elif z<1:
        #t=-(1.-z**2)**2/4./z**2
        #res=np.pi**(3./2.)*z**(-nu_p/2.)/mygamma((3.-nu_p)/2.)*\
        #        (mygammaRatio(ell+nu_p/2., ell+2.-nu_p/2.)*mygammaRatio(1.-nu_p/2, nu_p/2.-1.)\
        #            *myhyp21_basic(ell/2.+nu_p/4., nu_p/4.-(ell+1.)/2., nu_p/2., t)+\
        #    (-t/4.)**(1.-nu_p/2)*myhyp21_basic(ell/2.-nu_p/4.+1., 1./2.-nu_p/4.-ell/2.,2.-nu_p/2., t))
        #

        t=(1.-z**2)**2/4./z**2
        res=np.pi*z**(-nu_p/2.)*(np.sqrt(np.pi)*mygammaRatio(ell+nu_p/2., ell+2.-nu_p/2.)\
                                 *mygammaRatio(1.-nu_p/2, 3./2.-nu_p/2.)*\
        myhyp21_basic(ell/2.+nu_p/4., nu_p/4.-(ell+1.)/2., nu_p/2., -t)\
        -2.*mygamma(nu_p-2.)*np.cos(np.pi*nu_p/2.)*t**(1.-nu_p/2)\
         *myhyp21_basic(ell/2.-nu_p/4.+1., 1./2.-nu_p/4.-ell/2.,2.-nu_p/2., -t))
    else:
        res=np.pi**2*z**ell*2**(nu_p - 1.)*mygamma(2.-nu_p)\
             /mygamma((3.-nu_p)/2.)**2*mygammaRatio(ell+nu_p/2, 2.+ell-nu_p/2.)
    return res

@njit
def myhyp21(nu_p, t, chi, ell, t1min):
    '''
    returns the result of 4pi * \int dk k**(nu-1) jl(k*chi)jl(k*r) = chi**(-nu_p) * I_Assassi_Simonovic_Zaldarriaga
                                                                   = 2pi^2 / r^2 * I_me
    '''

    if t.real>1:
        fact=t**(-nu_p)
        t=1./t
    else:
        fact=1.+0.j

    if t<t1min:
        return 0.j
    else:
        return chi**(-nu_p) * fact * hyp21(nu_p, t, ell)


