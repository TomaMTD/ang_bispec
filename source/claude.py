import numpy as np
from numba import njit
from scipy.integrate import simpson
from dataclasses import dataclass
from typing import Tuple, Callable

@dataclass
class CosmologyParams:
    """Container for cosmological parameters at each radial point"""
    r_list: np.ndarray
    ar: np.ndarray
    Dr: np.ndarray
    fr: np.ndarray
    vr: np.ndarray
    wr: np.ndarray
    Omr: np.ndarray
    Hr: np.ndarray
    Wr: np.ndarray
    mathcalR: np.ndarray = None

# ============================================================================
# Coefficient Functions (α, β, γ)
# ============================================================================
def get_coefficients(p, time_dict):
    """
    Compute α, β, γ coefficients for different orders.
    Returns: (alpha_vals, beta_vals, gamma_vals) each as dict {order: array}
    """
    if p.which == 'F2':
        alpha = {
            0: (7. - 3.*time_dict['vr']) / 14.,
            1: 4.*time_dict['fr'] + 1.5*time_dict['Omr'] - 9./7.*time_dict['wr'],
            2: 18.*time_dict['fr']**2 + 9.*time_dict['fr']**2*time_dict['Omr'] - 4.5*fr*time_dict['Omr']
        }
        beta = {
            0: np.ones_like(time_dict['fr']),
            1: -2*time_dict['fr']**2 + 6*time_dict['fr'] - 4.5*time_dict['Omr'],
            2: 36.*time_dict['fr']**2 + 18.*time_dict['fr']**2*time_dict['Omr']
        }
        gamma = {
            0: np.zeros_like(time_dict['fr']),
            1: 0.5*(-time_dict['fr']**2 + time_dict['fr'] - 3.*time_dict['Omr']),
            2: 0.25*(18*time_dict['fr']**2 + 9.*(time_dict['fr']**2 - time_dict['fr'])*time_dict['Omr'])
        }
    else:  # G2 or dv2
        alpha = {
            0: time_dict['fr'] - 3./7.*time_dict['wr'],
            1: -7.5*time_dict['Omr']*time_dict['fr'],
            2: np.zeros_like(time_dict['fr'])
        }
        beta = {
            0: time_dict['fr'],
            1: -12.*time_dict['Omr']*time_dict['fr'],
            2: np.zeros_like(time_dict['fr'])
        }
        gamma = -2.25*time_dict['Omr']*time_dict['fr']  # Single value for all orders
    
    return alpha, beta, gamma

# ============================================================================
# Coefficients f_nm
# ============================================================================

def compute_f_nm(alpha, beta, time_dict):
    """
    Compute f^(0), f^(2), f^(4) multipoles from coefficients.
    Returns: (f0, f2, f4) arrays
    """
    H2 = time_dict['Hr']**2
    H4 = time_dict['Hr']**4
    
    # f^(0) - 4 components
    f0 = np.zeros((4, len(time_dict['Hr'])))
    f0[0] = (beta[0] - alpha[0]) / 2.
    f0[1] = (alpha[0] - beta[0]) / 4.
    f0[2] = H2 / 2. * (beta[1]/2. - alpha[1])
    f0[3] = H4 / 4. * alpha[2]
    
    # f^(2) - 2 components
    f2 = np.zeros((2, len(time_dict['Hr'])))
    f2[0] = 0.5 * (beta[0]/2. - alpha[0])
    f2[1] = H2 / 4. * alpha[1]
    
    # f^(4) - single component
    f4 = 0.25 * alpha[0]
    
    return f0, f2, f4

def compute_fm_nm(alpha, beta, gamma, time_dict):
    """
    Compute f^(-2), f^(-4)
    Returns: (fm2, fm4) arrays
    """
    H2 = time_dict['Hr']**2
    H4 = time_dict['Hr']**4
    
    # f^(-2) - 3 components
    fm2 = np.zeros((3, len(time_dict['Hr'])))
    if isinstance(gamma, dict):
        g1 = gamma[1]
        g2 = gamma[2]
    else:
        g1 = gamma
        g2 = gamma
    
    fm2[0] = (beta[1] - alpha[1])/2. - 2.*g1
    fm2[1] = (alpha[1] - beta[1])/4. + g1
    fm2[2] = H2 / 2. * (beta[2]/2. - alpha[2])
    fm2 *= H2
    
    # f^(-4) - 2 components
    fm4 = np.zeros((2, len(time_dict['Hr'])))
    fm4[0] = (beta[2] - alpha[2])/2. - 2.*g2
    fm4[1] = (alpha[2] - beta[2])/4. + g2
    fm4 *= H4
    
    return fm2, fm4

# ============================================================================
# Radiation f_nm
# ============================================================================

def compute_radiation_f_nm(p, time_dict):
    """Compute radiation-specific multipoles."""
    H2 = time_dict['Hr']**2
    H4 = time_dict['Hr']**4
    fac = time_dict['Dr'] / time_dict['ar']
    
    # f^(-2)_R - 2 components
    fm2R = np.zeros((2, len(time_dict['fr'])))
    base = time_dict['fr'] + 1.5*time_dict['Omr']
    fm2R[0] = base
    fm2R[1] = -base / 2.
    fm2R *= H2 * fac
    
    # f^(-4)_R - 2 components
    fm4R = np.zeros((2, len(time_dict['fr'])))
    if p.which == 'F2':
        base = 3.*time_dict['fr']*(time_dict['fr'] + 1.5*time_dict['Omr'])
    else:
        base = 3.*(time_dict['fr'] - 1)*(time_dict['fr'] + 1.5*time_dict['Omr'])
    fm4R[0] = base
    fm4R[1] = -base / 2.
    fm4R *= H4 * fac
    
    return fm2R, fm4R

# ============================================================================
# Differential Operators
# ============================================================================

def apply_operator(y, r_list, which, ell=None, mathcalR=None, Hr=None):
    """
    Apply appropriate differential operator based on 'which'.
    - F2: mathcalD operator (requires ell)
    - G2: double gradient
    - dv2: gradient with mathcalR and Hr scaling
    """
    if which == 'F2':
        from mathematica import mathcalD  # Assuming this exists
        return mathcalD(r_list, y, ell, axis=1)
    elif which == 'G2':
        return np.gradient(np.gradient(y, r_list, axis=1), r_list, axis=1)
    else:  # dv2
        y_scaled = y * mathcalR * Hr
        return np.gradient(y_scaled, r_list, axis=1)

# ============================================================================
# Main Integration Functions
# ============================================================================

@njit
def compute_integrand_relativistic(r_eval, chi, ell, fm2_interp, fm4_interp):
    """
    Compute integrand for pure relativistic terms (k^-2, k^-4).
    
    Args:
        r_eval: evaluation points (Nx1 array)
        chi: fixed chi value
        ell: multipole order
        fm2_interp: interpolated f^(-2) values at r_eval
        fm4_interp: interpolated f^(-4) values at r_eval
    
    Returns: (5, N) array of integrands
    """
    from mathematica import Il  # Assuming this exists
    
    t_list = r_eval[:, 0] / chi
    N = len(t_list)
    Am4 = np.zeros(N, dtype=np.complex128)
    
    for ind, t in enumerate(t_list):
        if t > 1:
            fact = t
            t_use = 1. / t
        else:
            fact = 1.
            t_use = t
        Am4[ind] = chi * fact * Il(-1+0.j, t_use+0.j, ell)
    
    # Combine with interpolated multipoles
    out = np.zeros((5, N))
    out[0] = fm2_interp[0] * Am4.real
    out[1] = fm2_interp[1] * Am4.real
    out[2] = fm2_interp[2] * Am4.real
    out[3] = fm4_interp[0] * Am4.real
    out[4] = fm4_interp[1] * Am4.real
    
    return out / (2 * np.pi**2)

@njit
def compute_integrand_radiation(r_eval, chi, ell, f_interp, cp_tr, 
                                 bphi, Nphi, eta, is_F2=True):
    """
    Compute integrand for radiation terms.
    
    Args:
        r_eval: evaluation points
        chi: fixed chi value
        ell: multipole order
        f_interp: interpolated multipole values
        cp_tr, bphi, Nphi, eta: FFTLog parameters
    
    Returns: (4, N) array of integrands
    """
    from mathematica import myhyp21, tmin_fct  # Assuming these exist
    
    t_list = r_eval[:, 0] / chi
    N = len(t_list)
    out = np.zeros((4, N))
    
    # Compute hypergeometric integrals
    Ilm2 = np.zeros(N, dtype=np.complex128)
    Ilm4 = np.zeros(N, dtype=np.complex128)
    
    t1min = tmin_fct(ell, bphi + 1j*0) if ell >= 5 else 0
    
    for p in range(-Nphi//2, Nphi//2 + 1):
        if is_F2:
            nu = 1. + bphi + 1j*p*eta
        else:
            nu = -1. + bphi + 1j*p*eta
        
        cp = cp_tr[p + Nphi//2]
        
        for ind, t in enumerate(t_list):
            if is_F2:
                Ilm4[ind] += cp * myhyp21(nu - 2., t, chi, ell, t1min)
            else:
                Ilm2[ind] += cp * myhyp21(nu, t, chi, ell, t1min)
                Ilm4[ind] += cp * myhyp21(nu - 2., t, chi, ell, t1min)
    
    # Combine with multipoles
    if is_F2:
        out[0] = f_interp[0] * Ilm4.real
        out[1] = f_interp[1] * Ilm4.real
        out[2] = f_interp[2] * Ilm4.real
        out[3] = f_interp[3] * Ilm4.real
    else:
        out[0] = f_interp[0] * Ilm2.real
        out[1] = f_interp[1] * Ilm2.real
        out[2] = f_interp[2] * Ilm4.real
        out[3] = f_interp[3] * Ilm4.real
    
    return out / (2 * np.pi**2)

# ============================================================================
# High-Level Interface
# ============================================================================

def compute_integral(chi_list, ell, which, params: CosmologyParams,
                     r0, ddr, normW, radiation=False, Newton=False,
                     cp_tr=None, bphi=None, Nphi=None, kmax=None, kmin=None):
    """
    Main function to compute integrals over chi.
    
    Args:
        chi_list: array of chi values
        ell: multipole order
        which: 'F2', 'G2', or 'dv2'
        params: CosmologyParams dataclass
        radiation: if True, compute radiation terms; else relativistic
        Newton: if True, use Newtonian limit (Hr=0)
    
    Returns:
        Array of shape (n_components+1, len(chi_list)) where first row is chi_list
    """
    from mathematica import W_tilde  # Assuming this exists
    
    # Prepare cosmology
    Hr = np.zeros_like(params.Hr) if Newton else params.Hr
    alpha, beta, gamma = get_coefficients(params.fr, params.vr, params.wr, 
                                          params.Omr, which)
    
    # Compute and process multipoles
    if radiation:
        eta = 2.*np.pi / np.log(kmax/kmin)
        fm2R, fm4R = compute_radiation_multipoles(params.Dr, params.fr, 
                                                   params.Omr, Hr, params.ar, which)
        
        # Apply differential operators
        WDr2 = params.Dr**2 * params.Wr
        fm2_proc = apply_operator(fm2R * WDr2, params.r_list, which, ell, 
                                  params.mathcalR, Hr)
        fm4_proc = apply_operator(fm4R * WDr2, params.r_list, which, ell,
                                  params.mathcalR, Hr)
        
        n_components = 4
    else:
        f0, fm2, fm4 = compute_f_multipoles(alpha, beta, gamma, Hr, which)
        WDr2 = params.Dr**2 * params.Wr
        
        # Process based on which
        if which == 'F2':
            fm2_proc = fm2 * WDr2
            fm4_proc = fm4 * WDr2
        else:
            fm2_proc = apply_operator(fm2 * WDr2, params.r_list, which, ell,
                                      params.mathcalR, Hr)
            fm4_proc = apply_operator(fm4 * WDr2, params.r_list, which, ell,
                                      params.mathcalR, Hr)
        
        # Apply final mathcalD for F2
        if which == 'F2':
            from mathematica import mathcalD
            fm2_proc = mathcalD(params.r_list, fm2_proc, ell, axis=1)
        
        n_components = 5
    
    # Integration loop
    results = np.zeros((n_components, len(chi_list)))
    
    for ind, chi in enumerate(chi_list):
        if ind % 10 == 0:
            print(f'   {ind}/{len(chi_list)}')
        
        # Interpolate multipoles at evaluation points
        r_eval = params.r_list[:, None]
        fm2_interp = np.array([np.interp(r_eval[:, 0], params.r_list, fm2_proc[i]) 
                               for i in range(len(fm2_proc))])
        
        if radiation:
            # Combine for radiation case
            f_interp = np.vstack([fm2_interp, 
                                 np.array([np.interp(r_eval[:, 0], params.r_list, 
                                                    fm4_proc[i]) 
                                          for i in range(len(fm4_proc))])])
            
            integrand = compute_integrand_radiation(r_eval, chi, ell, f_interp,
                                                   cp_tr, bphi, Nphi, eta,
                                                   is_F2=(which=='F2'))
        else:
            fm4_interp = np.array([np.interp(r_eval[:, 0], params.r_list, 
                                            fm4_proc[i] if hasattr(fm4_proc, '__len__') 
                                            else fm4_proc) 
                                  for i in range(2)])
            
            integrand = compute_integrand_relativistic(r_eval, chi, ell, 
                                                      fm2_interp, fm4_interp)
        
        # Integrate over r
        results[:, ind] = simpson(integrand.T, x=params.r_list) * chi**2
    
    return np.vstack([chi_list, results])
