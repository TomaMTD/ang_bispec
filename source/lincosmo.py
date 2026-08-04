import numpy as np
import os
from classy import Class
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
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

############################################################################# Euclid bias fits
def b1_euclid(z):
    """Linear bias for the Euclid photometric sample: IST:Fisher fiducial b(z)=sqrt(1+z)
    (arXiv:1910.09273), valid across the full photometric range 0<z<2.5."""
    return np.sqrt(1. + z)
    # Flagship cubic fit (Euclid XIX App. C): more accurate at low z but only valid to z~2:
    # return 0.5125 + 1.377*z + 0.222*z**2 - 0.249*z**3

def b2_euclid(z):
    """Second-order bias via the Lazeyras relation (1511.01096) evaluated on b1_euclid(z)."""
    b1 = b1_euclid(z)
    return 0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**3

def s_euclid(z):
    """1506.01369 Calibrated only to z~2, so extrapolated over the upper half of bin 4 (1.32<z<2.50)."""
    return 0.1194 + 0.2122*z - 0.0671*z**2 + 0.1031*z**3
    # 2110.05435  
    # return 0.0842 + 0.0532*z + 0.298*z**2 - 0.0113*z**3

def volume_element(z):
    """
    Volume element dV/(dOmega dz) in (Mpc/h)^3/steradian
    This is r^2(z) * 1/H(z) where r is comoving distance
    """
    r=np.zeros((len(z)))
    for ind, zi in enumerate(z):
        r[ind]=get_distance(zi)
 
    drdz = 1/H_(z)  # Mpc / h
    return r, r**2 * drdz  # (Mpc/h)^3

def nz_volumetric_to_angular(z, ng_volumetric):
    """
    Convert volumetric n(z) to angular n(z)

    Parameters:
    -----------
    z : array
        Redshift values
    ng_volumetric : array
        Volumetric galaxy density dN/dV in (h/Mpc)^3

    Returns:
    --------
    nz_angular : array
        Angular galaxy distribution dN/(dOmega dz) in h^3/Mpc^3 * (Mpc/h)^3/steradian = 1/steradian
    """
    r, dV_dOmega_dz = volume_element(z)
    nz_angular = ng_volumetric * dV_dOmega_dz
    return r, nz_angular

def construct_window_functions(z_range, bin_edges, nz_total, sigma_z=1e-3):
    """
    Construct smooth window functions for each redshift bin using error functions.
    
    Parameters:
    -----------
    z_range : array
        Fine redshift grid for window functions
    bin_edges : array
        Bin edges in redshift [z_min_1, z_max_1, z_min_2, z_max_2, ..., z_max_N]
    nz_total : array
        Total angular galaxy distribution on z_range grid
    sigma_z : float
        Smoothing scale for error function (default: 1e-3, suitable for spectroscopic surveys)
        
    Returns:
    --------
    windows : array (n_bins, len(z_range))
        Window function for each bin (normalized to unit integral)
    nz_binned : array (n_bins, len(z_range))
        Galaxy distribution for each bin = window * nz_total
    norms : array
        Normalization factors for each bin
    """
    n_bins = len(bin_edges) - 1
    windows = np.zeros((n_bins, len(z_range)))

    # Multiply by the total n(z) distribution
    nz_binned = nz_total[np.newaxis, :]

    # Normalize each bin to unit area
    norms = np.trapz(nz_binned, z_range, axis=1)
    windows_normalized = windows / norms[:, np.newaxis]
    nz_binned_normalized = nz_binned / norms[:, np.newaxis]

    return windows_normalized, nz_binned_normalized, norms

############################################################################ growth factors
def solvr(Y, t):
    a=Y[0]
    H=H_(1./a-1.)
#     print([a**2*H, Y[2], -a*H*Y[2]+3./2.* omega_m*H0**2/a*Y[1]])
    return [a*H, Y[2], -H*Y[2]+3./2.* omega_m*H0**2/a*Y[1], Y[4], -H*Y[4]+3./2.*omega_m*H0**2*(Y[3]+Y[1]**2) / a]


def growth_fct(input_data=0, window_type=None):
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

    # Resample the (exact) dense output on a grid uniform in r: the adaptive stepper leaves only
    # ~26 nodes, none below r=268, which is too coarse for the k=5 splines and too high for lensing.
    r_floor, Nr = 60., 300
    t_map = np.geomspace(t0, tmax, 2000)
    a_map = sol.sol(t_map)[0]
    ok    = a_map < 1.
    r_map = np.array([get_distance(1./A - 1.)[0] for A in a_map[ok]])
    o     = np.argsort(r_map)
    # descending in r == ascending in a, the order the code below expects
    asol  = sol.sol(np.interp(np.linspace(8000., r_floor, Nr), r_map[o], t_map[ok][o])).T

    apy = asol[:,0]
    Dpy = asol[:,1]
    fpy = asol[:,2]/(H_(1./apy-1.)*Dpy)
    vpy = 7./3.*asol[:,3]/Dpy**2
    wpy = 7./6.*asol[:,4]/(H_(1./apy-1.)*Dpy**2)
    
    # exact D(a=1): a linear interp on the node grid made D0 depend on where the stepper stopped
    D0 = sol.sol(brentq(lambda t: sol.sol(t)[0]-1., sol.t[0], sol.t[-1]))[1]
    Dpy/=D0

    ra=np.zeros((len(apy)))
    Ha=np.zeros((len(apy)))
    Oma=np.zeros((len(apy)))
    za=1./apy-1.
    for ind, zi in enumerate(za):
        ra[ind]=get_distance(zi)
        Ha[ind]=H_(zi)
        Oma[ind]=Om_(zi)

    mask = np.logical_and(ra[::-1]>0, ra[::-1]<8000) # unphysical small distances, avoid spline error
    dHa = dotH_(1./apy[::-1]-1.)

    time_dict = {'a'  : apy[::-1][mask],\
            'ra' : ra[::-1][mask],\
            'Ha' : Ha[::-1][mask],\
            'Oma': Oma[::-1][mask],\
            'Da' : Dpy[::-1][mask],\
            'fa' : fpy[::-1][mask],\
            'va' : vpy[::-1][mask],\
            'wa' : wpy[::-1][mask],\
            'dHa': dHa[mask],\
            'mathcalR': (dHa/Ha[::-1]**2+2./Ha[::-1]/ra[::-1])[mask]}

    if window_type=='euclid':
        # Analytical biases on the comoving grid (no n_angular: the euclid window
        # already carries the n_i(z) shape). Enables the existing b1/b2/b_s machinery.
        z_grid = 1./time_dict['a'] - 1.
        time_dict['data'] = {'r' : time_dict['ra'],
                             'b1': b1_euclid(z_grid),
                             'b2': b2_euclid(z_grid),
                             's' : s_euclid(z_grid)}   # magnification slope, for the lensing lterm
        return time_dict
    elif window_type=='ska':
        data = np.loadtxt(input_data)
        z = data[:, 0]
        ng = data[:, 1]
        b1 = data[:, 2]
        b2 = data[:, 3]

        r, n_angular = nz_volumetric_to_angular(z, ng)
        time_dict['data'] = {'r'        :r,
                             'n_angular':n_angular,
                             'b1'       :b1,
                             'b2'       :b2}
        return time_dict
    else:
        return time_dict




############################################################################# power spectrum
def trans(z=0):
    if not os.path.isfile(output_dir+'class_transfer.npy'):# or force:
        print('computing class')
        clss = Class()
        clss.set({'gauge': 'new', 'h': h,'omega_b': omega_b*h**2, 'omega_cdm': omega_cdm*h**2,
                  'output':'dTk,vTk','z_pk': 10, 'A_s': A_s , 'n_s': n_s,
                  'k_per_decade_for_pk' :  50,
                  'k_per_decade_for_bao' : 50,
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
