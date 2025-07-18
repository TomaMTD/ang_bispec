import numpy as np
from numba import njit
import os
from mathematica import *
from lincosmo import *
from param_used import *
from scipy.interpolate import UnivariateSpline

############################################################################# fftlog
@njit
def get_cp_eta_p(k, fct_k, b):
    Nk = len(k)
    kmin, kmax = np.min(k), np.max(k)

    l=np.arange(Nk)
    p_list = np.arange(-Nk//2, Nk//2+1)
    eta_p_list = 2.*np.pi*p_list/np.log(kmax/kmin)
    
    res  =np.zeros((Nk+1), dtype=np.complex128)
    for p in range(-Nk//2, Nk//2+1):
        res[p+Nk//2] = np.sum(fct_k * k**(-b) * kmin**(-1j*eta_p_list[p+Nk//2]) * np.exp(-2.*1j*np.pi*p*l/Nk)) / Nk

    return res, eta_p_list

############################################################################# get fftlog coef
def set_bias(k, fctk_list):
    b_list = np.zeros(len(fctk_list), dtype=np.float64)

    for ind_b, fctk in enumerate(fctk_list):
        ind=0
        for i in range(5, 16):
            b_list[ind_b]+=(np.log(np.abs(fctk[-i])) - np.log(np.abs(fctk[i]))) / (np.log(k[-i]) - np.log(k[i]))
            ind+=1

        b_list[ind_b]/=ind

    print(' biases: {:.2f}'.format(b_list))
    return b_list

def quadratic_terms(k, B, lterm, which):
    """
    Calculate quadratic terms
    
    Parameters:
    -----------
    qterm : int
        Term index (0, 1, 2, or 3)
    k : array_like
        Wave number array
    B : array_like
        Input array B
    lterm : str
        Linear term type ('density' or other)
    which : str
        Derivative type ('d2v', 'd1v', 'd1d', 'd3v', 'd0d')
    
    Returns:
    --------
    array_like
        Computed quadratic terms
    """
    
    B_scaled =B*k**4
    if which == 'd2v':
        if lterm == 'density':
            B_scaled *= k**2

        spline = UnivariateSpline(k, B_scaled, k=5, s=0)
        return [spline.derivative(2)(k),\
                -2./k * spline.derivative(1)(k),\
                B/k**2]

    elif which in ['d1v', 'd1d']:
        if lterm == 'density':
            B_scaled *= k
        else:
            B_scaled /= k
        
        if which == 'd1d':
            B_scaled *= k**2

        spline = UnivariateSpline(k, B_scaled, k=5, s=0)
        return [spline.derivative(1)(k), -B/k]

    elif which == 'd3v':
        if lterm == 'density':
            B_scaled *= k**3
        else:
            B_scaled *= k

        spline = UnivariateSpline(k, B_scaled, k=5, s=0)
        return [spline.derivative(3)(k),
                -3./k * spline.derivative(2)(k),
                3./k**2 * spline.derivative(1)(k),
                -B / k**3]

    elif which == 'd0d':
        if lterm != 'density':
            B_scaled /= k**2
        return [B_scaled * k**2]
    
    else:
        raise ValueError(f"Invalid 'which' parameter: {which}")


def apply_fftlog(k, Pk, lterm, which, qterm, rad):
    if which in ['FG2', 'F2', 'G2', 'dv2', 'local']: 
        if not rad:
            fct_k=[Pk*k**4]
        else:
            fct_k=[Pk]
        np.save(output_dir+'fct_k'.format('FG2_dv2', qterm), np.vstack([k, fct_k]).T)
    else: 
        fct_k = quadratic_terms(qterm, k, Pk, lterm, which) 
        np.save(output_dir+'fct_k_{}_lterm{}_qterm{}'.format(which, lterm, qterm), np.vstack([k, fct_k]).T)

    b=set_bias(k, fct_k)
    cp, eta = get_cp_eta_p(k, fct_k, b)
    return cp, eta, b


def mathcalD(x, y, ell, axis=1):
    dy=np.gradient(y, x, axis=axis)
    return -np.gradient(dy, x, axis=axis)+2./x*dy+(ell*(ell+1)-2.)/x**2*y

def fct_of_r_numeric(ell_list, r_list, window, number_of_derive=2):

    y_list = np.zeros((len(ell_list), len(r_list)), dtype=np.float64)
    if number_of_derive == 0:
        for ind_ell in range(len(ell_list)):
            ell = ell_list[ind_ell]
            y_list[ind_ell] = window

    elif number_of_derive == 1:
        for ind_ell in range(len(ell_list)):
            ell = ell_list[ind_ell]
            y_list[ind_ell] = mathcalD(r_list, window, ell)
    else:
        for ind_ell in range(len(ell_list)):
            ell = ell_list[ind_ell]
            y_list[ind_ell] = mathcalD(r_list, window, ell)
            for _ in range(number_of_derive - 1):
                y_list[ind_ell] = mathcalD(r_list, y_list[ind_ell], ell)

    return y_list


def f_of_r(k, Pk, lterm, which, qterm, Newton=0, time_dict=0, window_args=0):
    fct_r = mathcalB(which, lterm, qterm, Newton, time_dict, r0, ddr, normW)
    np.save(output_dir+'fct_r_{}_lterm{}_qterm{}'.format(which, lterm, qterm), fct_r)
    return fct_r
