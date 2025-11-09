import numpy as np
import os
from classy import Class
from scipy.integrate import solve_ivp
from numba import njit
import cubature


from param_used import *
from mathematica import *

############################################################################# Basic cosmo fct
@njit
def H_(z):
    return H0*np.sqrt((omega_b+omega_cdm)*(1+z)+omega_l*(1+z)**-2)

@njit
def Om_(z):
    return H0**2*omega_m*(1+z)/H_(z)**2

@njit
def dotH_(z):
    return H_(z)**2 - 3./2.*H0**2*omega_m*(1+z)

@njit
def dist_integrand(z):
    out=1./(H0*np.sqrt((omega_cdm+omega_b)*(1+z)**3+omega_l \
            +omega_r*(1+z)**4+omega_k*(1+z)**2))
    return out[:,0]

def get_distance(z):
    val, err = cubature.cubature(dist_integrand, ndim=1, fdim=1, xmin=[0], xmax=[z],
            relerr=1e-10, maxEval=0, abserr=0, vectorized=True)
    return val

############################################################################ growth factors
def solvr(Y, t):
    a=Y[0]
    H=H_(1./a-1.)
#     print([a**2*H, Y[2], -a*H*Y[2]+3./2.* omega_m*H0**2/a*Y[1]])
    return [a*H, Y[2], -H*Y[2]+3./2.* omega_m*H0**2/a*Y[1], Y[4], -H*Y[4]+3./2.*omega_m*H0**2*(Y[3]+Y[1]**2) / a]


def growth_fct():
    print('computing growth')
    a0=1e-10
    z0=1./a0-1.
    D0=1.
    Dprime0= 2.*D0*H_(z0) / (c*1e5)**2
    
    F0 = 3./7.*a0*a0
    Fprime0 = 12./7.*a0**(3./2.)

    t0 = 1./(H_(z0))
    tmax =1e4
    
    sol = solve_ivp(lambda t, y: solvr(y, t), [t0, tmax], 
                [a0, D0, Dprime0, F0, Fprime0],
                method='DOP853',  # 8th order Runge-Kutta
                rtol=1e-12, atol=1e-14,
                dense_output=True)
    asol = sol.y.T
    
    apy = asol[:,0]
    Dpy = asol[:,1]
    fpy = asol[:,2]/(H_(1./apy-1.)*Dpy)
    vpy = 7./3.*asol[:,3]/Dpy**2
    wpy = 7./6.*asol[:,4]/(H_(1./apy-1.)*Dpy**2)
    
    D0=np.interp(1, asol[:,0], asol[:,1])
    Dpy/=D0

    ra=np.zeros((len(apy)))
    Ha=np.zeros((len(apy)))
    Oma=np.zeros((len(apy)))
    za=1./apy-1.
    for ind, zi in enumerate(za):
        ra[ind]=get_distance(zi)
        Ha[ind]=H_(zi)
        Oma[ind]=Om_(zi)

    mask = np.logical_and(ra[::-1]>100, ra[::-1]<8000) # unphysical small distances, avoid spline error
    dHa = dotH_(1./apy[::-1]-1.)
    return {'a'  : apy[::-1][mask],\
            'ra' : ra[::-1][mask],\
            'Ha' : Ha[::-1][mask],\
            'Oma': Oma[::-1][mask],\
            'Da' : Dpy[::-1][mask],\
            'fa' : fpy[::-1][mask],\
            'va' : vpy[::-1][mask],\
            'wa' : wpy[::-1][mask],\
            'dHa': dHa[mask],\
            'mathcalR': (dHa/Ha[::-1]**2+2./Ha[::-1]/ra[::-1])[mask]}


############################################################################# power spectrum
def trans(z=0):
    if not os.path.isfile(output_dir+'class_transfer.npy') or force:
        print('computing class')
        clss = Class()
        clss.set({'gauge': 'new', 'h': h,'omega_b': omega_b*h**2, 'omega_cdm': omega_cdm*h**2,
                  'output':'dTk,vTk','z_pk': 10, 'A_s': A_s , 'n_s': n_s,
                  'k_per_decade_for_pk' :  100,
                  'k_per_decade_for_bao' : 100,
                  'compute damping scale' : 'yes',
                  'P_k_max_h/Mpc' : 20,
                    'tol_background_integration': 1e-9,
                    'tol_thermo_integration': 1e-9,
                    'tol_perturb_integration': 1e-9,

                 })
        clss.compute()

        tr=clss.get_transfer(z=z)
        tr['k'] = tr.pop('k (h/Mpc)')

        tr['dTdk'] = np.gradient(np.log(tr['phi']), np.log(tr['k']))

        tr['d_m'] =  (omega_cdm*tr['d_cdm'] + omega_b*tr['d_b'])/(omega_b+omega_cdm)

        tr['t_m'] =  (omega_cdm*tr['t_cdm'] + omega_b*tr['t_b'])/(omega_b+omega_cdm)
        tr['v_m'] = -tr['t_m']/tr['k']**2/h

        np.save(output_dir+'class_transfer', tr)
    else:
        print('loading class')
        tr = np.load(output_dir+'class_transfer.npy', allow_pickle=True).tolist()
    
    return tr

def primordial(k):
    return A_s*(k/(k_pivot/h))**(n_s-1)/k**3*2*np.pi**2

def powerspectrum(k,delta_cdm):
    prim = primordial(k)
    T=np.interp(k,delta_cdm[0],delta_cdm[1])
    return prim*T**2

def get_power(z):
    tr = trans(z)
    Pk = powerspectrum(tr['k'],np.array([tr['k'], tr['phi']]))
    return tr, Pk
