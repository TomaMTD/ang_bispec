import numpy as np
import os
from numba import njit
from math import comb
import cubature, time, h5py
from scipy.integrate import simpson
from sympy.physics.wigner import wigner_3j
from filelock import FileLock
from scipy.interpolate import UnivariateSpline
import sympy as sp
from scipy.integrate import quad
from scipy.special import erf  # Vectorized erf for lambdify


def compute_spline_derivatives(spline, data_grid, eval_grid, max_deriv=11, smooth_s=0):
    """
    Compute high-order derivatives of a spline using chain strategy.

    Parameters:
    -----------
    spline : UnivariateSpline
        The spline to differentiate
    data_grid : array
        Full grid where spline is defined (for computing derivatives)
    eval_grid : array
        Grid where derivatives should be evaluated
    max_deriv : int
        Maximum derivative order (default 11)
    smooth_s : float
        Smoothing parameter for intermediate splines (default 0)

    Returns:
    --------
    derivs : list of arrays
        List of derivatives [f, f', f'', ..., f^(max_deriv)] evaluated on eval_grid
    """

    # Compute derivatives on full data grid
    derivs_full = [None] * (max_deriv + 1)

    # Direct derivatives (0-5)
    derivs_full[0] = spline(data_grid)
    for i in range(1, min(6, max_deriv + 1)):
        derivs_full[i] = spline.derivative(i)(data_grid)

    # Derivatives 6-10 from spline of 5th derivative
    if max_deriv >= 6:
        d5_spline = UnivariateSpline(data_grid, derivs_full[5], k=5, s=smooth_s)
        derivs_full[i] = d5_spline(data_grid)
        for i in range(6, min(11, max_deriv + 1)):
            derivs_full[i] = d5_spline.derivative(i - 5)(data_grid)

    # Derivative 11 from spline of 10th derivative
    if max_deriv >= 11:
        d10_spline = UnivariateSpline(data_grid, derivs_full[10], k=5, s=smooth_s)
        derivs_full[11] = d10_spline.derivative(1)(data_grid)

    # Evaluate on target grid
    derivs = [None] * (max_deriv + 1)
    for i in range(max_deriv + 1):
        if np.array_equal(data_grid, eval_grid):
            derivs[i] = derivs_full[i]
        else:
            deriv_spline = UnivariateSpline(data_grid, derivs_full[i], k=5, s=0, ext=0)
            derivs[i] = deriv_spline(eval_grid)

    return derivs


# Optimized version that caches the SymPy derivatives
class WindowDerivatives:
    """
    Cache SymPy derivatives for W(r) and r^n * W(r) expressions for efficiency
    """
    def __init__(self, window_args, max_r_power=3, max_derivative=11):
        # Handle both old (5 args) and new (6 args) window_args format
        if len(window_args) == 6:
            self.xmin, self.xmax, self.H_over_a_data, self.sigma_z, self.window_type, self.n_angular_data = window_args
        else:
            # Backward compatibility
            self.xmin, self.xmax, self.H_over_a_data, self.sigma_z, self.window_type = window_args
            self.n_angular_data = None

        self.max_r_power = max_r_power
        self.max_derivative = max_derivative
        self._cached_expressions = {}
        self._compile_derivatives()

    def _compile_derivatives(self):
        """Pre-compile SymPy expressions for faster evaluation"""
        sp_x = sp.symbols('x')
        sp_xmin = sp.symbols('xmin')
        sp_xmax = sp.symbols('xmax')
        sp_sigma_z = sp.symbols('sigma_z')

        
        if self.window_type=='nbody':
            # Define unnormalized window function (single definition!)
            W_unnorm = (0.5+0.5*sp.tanh((sp_x-sp_xmin)/sp_sigma_z)) *\
                       (0.5-0.5*sp.tanh((sp_x-sp_xmax)/sp_sigma_z))

        else:
            # Use sympy versions:
            window_lower = 0.5 * (1 + sp.erf((sp_x - sp_xmin) / (sp.sqrt(2) * sp_sigma_z)))
            window_upper = 0.5 * (1 + sp.erf((sp_xmax - sp_x) / (sp.sqrt(2) * sp_sigma_z)))
            W_unnorm = window_lower * window_upper

        # Compute normalization: ∫ (H/a * W_unnorm) dr or ∫ (n_angular * H/a * W_unnorm) dr
        if self.H_over_a_data is not None:
            # Unpack H/a data: (ra_grid, H_over_a_values)
            ra_grid, H_over_a_values = self.H_over_a_data

            # Create H/a spline
            H_over_a_spline = UnivariateSpline(ra_grid, H_over_a_values, s=0, ext=0)

            # Create n_angular spline if available and not nbody
            if self.n_angular_data is not None and self.window_type != 'nbody':
                r_nz_grid, n_angular_values = self.n_angular_data
                n_angular_spline = UnivariateSpline(r_nz_grid, n_angular_values, s=0, ext=0)
                print(f"    Using n(z) angular normalization (SKA-type window)")
            else:
                n_angular_spline = None

            # Lambdify the unnormalized window for numerical integration
            W_unnorm_func = sp.lambdify(sp_x, W_unnorm.subs({sp_xmin: self.xmin,
                                                              sp_xmax: self.xmax,
                                                              sp_sigma_z: self.sigma_z}), 'numpy')

            # Integrand: includes n_angular if available (for SKA-type surveys)
            def integrand(r):
                H_over_a = H_over_a_spline(r)
                W = W_unnorm_func(r)
                if n_angular_spline is not None:
                    return n_angular_spline(r) * H_over_a * W
                else:
                    return H_over_a * W

            sp_normW, _ = quad(integrand, min(0, self.xmin - 20*self.sigma_z), self.xmax+ 20*self.sigma_z, epsrel=1e-4, epsabs=0)
            integral_type = "n_angular*H/a*W" if n_angular_spline is not None else "H/a*W"
            print(f"        Computed normW = ∫({integral_type})dr = {sp_normW:.4e}")
        else:
            # Fallback: use analytical formula for ∫ W dr (old behavior)
            sp_normW = 1./4.*self.sigma_z*(1. + 1./np.tanh((self.xmax - self.xmin)/self.sigma_z))*2./self.sigma_z*(self.xmax-self.xmin)
            print(f"  Warning: H/a data not provided, using ∫W dr normalization = {sp_normW:.6e}")

        # Store normW as instance variable for later use
        self.normW = sp_normW

        # Normalized window: W = W_unnorm / normW
        base_W = W_unnorm / sp_normW
        
        # For each r power and derivative order
        for r_power in range(self.max_r_power + 1):  # 0 to max_r_power
            for deriv_order in range(self.max_derivative + 1):  # 0 to max_derivative
                
                expr = sp.diff(sp_x**r_power * base_W, sp_x, deriv_order)
                
                # Substitute constants
                expr = expr.subs({sp_xmin: self.xmin, 
                                 sp_xmax: self.xmax, 
                                 sp_normW: self.normW,
                                 sp_sigma_z: self.sigma_z})
                
                # Lambdify for fast numerical evaluation with custom namespace
                # Include both erf (for gaussian window) and tanh (for nbody window)
                custom_namespace = {'erf': erf, 'exp': np.exp, 'sqrt': np.sqrt, 'tanh': np.tanh}
                self._cached_expressions[(r_power, deriv_order)] = sp.lambdify(sp_x, expr, modules=[custom_namespace, 'numpy'])
    
    def __call__(self, x, r_power=0, num_derivative=9):
        """
        Evaluate the nth derivative of r^r_power * W(r) at x
        
        Parameters:
        - x: evaluation points
        - r_power: power of r prefactor (0 for just W, 1 for r*W, etc.)
        - num_derivative: order of derivative to take
        
        Returns:
        - Array of evaluated derivatives
        """
        if (r_power, num_derivative) not in self._cached_expressions:
            raise ValueError(f"Derivative not cached for r_power={r_power}, num_derivative={num_derivative}")
        
        return self._cached_expressions[(r_power, num_derivative)](x)
    
    def get_all_derivatives(self, x, r_power=0, max_deriv=None):
        """
        Get all derivatives up to max_deriv for r^r_power * W(r)
        For SKA-type surveys (window_type != 'nbody'), this returns derivatives of n_angular(r) * W(r)

        Returns list [f, f', f'', f''', ...] evaluated at x
        """
        if max_deriv is None:
            max_deriv = self.max_derivative

        # Compute W derivatives (without n_angular)
        W_derivs = [self(x, r_power=r_power, num_derivative=i) for i in range(max_deriv + 1)]

        # If we have n_angular and not nbody, multiply by n_angular using product rule
        if self.n_angular_data is not None and self.window_type != 'nbody':
            r_nz_grid, n_angular_values = self.n_angular_data
            n_angular_spline = UnivariateSpline(r_nz_grid, n_angular_values, k=5, s=0, ext=0)

            # Compute n_angular derivatives using the new function
            # Use full data grid for accuracy, then evaluate on x
            n_angular_derivs = compute_spline_derivatives(n_angular_spline, r_nz_grid, x,
                                                          max_deriv=max_deriv, smooth_s=1e-6)

            # Apply product rule: d^n/dr^n [n_angular * W] = sum_{k=0}^n C(n,k) * n_angular^(k) * W^(n-k)
            W_eff_derivs = []
            for n in range(max_deriv + 1):
                deriv_n = sum(comb(n, k) * n_angular_derivs[k] * W_derivs[n - k] for k in range(n + 1))
                W_eff_derivs.append(deriv_n)

            ## Save for visualization
            #np.save('r_nz_grid', x)
            #np.save('n_angular', W_eff_derivs)

            return W_eff_derivs
        else:
            # No n_angular: return W derivatives as-is
            return W_derivs


def load_or_compute_window_derivatives(p, window_args, r_list, output_dir, max_deriv=11):
    """
    Load window derivatives from cache if available and valid, otherwise compute and save.

    Parameters:
    -----------
    window_args : tuple
        (xmin, xmax, H_over_a_data, sigma_z, window_type) for window function
        where H_over_a_data is a tuple (ra_grid, H_over_a_values)
    r_list : array
        Radial coordinates
    output_dir : str
        Directory to save/load cache file
    max_deriv : int, optional
        Maximum derivative order (default: 11)

    Returns:
    --------
    W_derivs_list : list of arrays
        List of window derivatives [W, dW, d2W, ..., d^max_deriv W]
    """
    window_cache_file = f'{output_dir}window_derivs_cache.npz'

    # Check if cached file exists and is valid
    load_from_cache = False
    if not p.force and os.path.exists(window_cache_file):
        print('  Found cached window derivatives, checking validity...')
        try:
            cache = np.load(window_cache_file, allow_pickle=True)
            cached_params = cache['params'].item()

            # Check if parameters match
            # Need special handling for window_args since it contains numpy arrays
            # Handle both old (5 elements) and new (6 elements) formats
            cached_window_args = cached_params['window_args']

            if len(cached_window_args) == 6:
                cached_xmin, cached_xmax, cached_H_data, cached_sigma_z, cached_window_type, cached_n_angular = cached_window_args
            else:
                cached_xmin, cached_xmax, cached_H_data, cached_sigma_z, cached_window_type = cached_window_args
                cached_n_angular = None

            if len(window_args) == 6:
                xmin, xmax, H_data, sigma_z, window_type, n_angular_data = window_args
            else:
                xmin, xmax, H_data, sigma_z, window_type = window_args
                n_angular_data = None

            # Check n_angular data match
            n_angular_match = True
            if n_angular_data is not None and cached_n_angular is not None:
                n_angular_match = (np.allclose(cached_n_angular[0], n_angular_data[0]) and
                                  np.allclose(cached_n_angular[1], n_angular_data[1]))
            elif n_angular_data is None and cached_n_angular is None:
                n_angular_match = True
            else:
                n_angular_match = False

            window_args_match = (
                cached_xmin == xmin and
                cached_xmax == xmax and
                cached_sigma_z == sigma_z and
                cached_window_type == window_type and
                n_angular_match and
                H_data is not None and cached_H_data is not None and
                np.allclose(cached_H_data[0], H_data[0]) and  # ra_grid
                np.allclose(cached_H_data[1], H_data[1])      # H_over_a values
            )

            params_match = (
                window_args_match and
                cached_params['n_r'] == len(r_list) and
                np.isclose(cached_params['r_min'], r_list[0]) and
                np.isclose(cached_params['r_max'], r_list[-1]) and
                cached_params['max_deriv'] == max_deriv
            )

            if params_match:
                print('  Cache valid! Loading precomputed window derivatives...')
                W_derivs_list = [cache[f'deriv_{i}'] for i in range(max_deriv + 1)]
                load_from_cache = True
            else:
                print('  Cache invalid (parameters changed), will recompute...')
        except Exception as e:
            print(f'  Error loading cache: {e}, will recompute...')

    if not load_from_cache:
        print('  Computing window function derivatives...')
        W_derivs = WindowDerivatives(window_args)
        W_derivs_list = W_derivs.get_all_derivatives(r_list, r_power=0, max_deriv=max_deriv)

        # Save to cache
        print('  Saving window derivatives to cache...')
        cache_params = {
            'window_args': window_args,
            'n_r': len(r_list),
            'r_min': r_list[0],
            'r_max': r_list[-1],
            'max_deriv': max_deriv
        }
        save_dict = {'params': cache_params}
        for i, deriv in enumerate(W_derivs_list):
            save_dict[f'deriv_{i}'] = deriv
        np.savez(window_cache_file, **save_dict)
        print(f'  Cache saved to {window_cache_file}')

    return W_derivs_list


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
            0: (7. - 3.*time_dict['va']) / 14.,
            1: 4.*time_dict['fa'] + 1.5*time_dict['Oma'] - 9./7.*time_dict['wa'] if not p.Newton else np.zeros_like(time_dict['fa']),
            2: 18.*time_dict['fa']**2 + 9.*time_dict['fa']**2*time_dict['Oma'] - 4.5*time_dict['fa']*time_dict['Oma'] if not p.Newton else np.zeros_like(time_dict['fa'])
        }
        beta = {
            0: np.ones_like(time_dict['fa']),
            1: -2*time_dict['fa']**2 + 6*time_dict['fa'] - 4.5*time_dict['Oma'] if not p.Newton else np.zeros_like(time_dict['fa']),
            2: 36.*time_dict['fa']**2 + 18.*time_dict['fa']**2*time_dict['Oma'] if not p.Newton else np.zeros_like(time_dict['fa'])
        }
        gamma = {
            0: np.zeros_like(time_dict['fa']),
            1: 0.5*(-time_dict['fa']**2 + time_dict['fa'] - 3.*time_dict['Oma']) if not p.Newton else np.zeros_like(time_dict['fa']),
            2: 0.25*(18*time_dict['fa']**2 + 9.*(time_dict['fa']**2 - time_dict['fa'])*time_dict['Oma']) if not p.Newton else np.zeros_like(time_dict['fa'])
        }
    else:  # G2 or dv2
        alpha = {
            0: time_dict['fa'] - 3./7.*time_dict['wa'],
            1: -7.5*time_dict['Oma']*time_dict['fa'] if not p.Newton else np.zeros_like(time_dict['fa']),
            2: np.zeros_like(time_dict['fa']) if not p.Newton else np.zeros_like(time_dict['fa'])
        }
        beta = {
            0: time_dict['fa'],
            1: -12.*time_dict['Oma']*time_dict['fa'] if not p.Newton else np.zeros_like(time_dict['fa']),
            2: np.zeros_like(time_dict['fa']) if not p.Newton else np.zeros_like(time_dict['fa'])
        }
        gamma = {
            0: np.zeros_like(time_dict['fa']),
            1: -2.25*time_dict['Oma']*time_dict['fa'] if not p.Newton else np.zeros_like(time_dict['fa']),  # Single value for all orders
            2: np.zeros_like(time_dict['fa']) if not p.Newton else np.zeros_like(time_dict['fa'])
        }
    
    return alpha, beta, gamma

# ============================================================================
# Coefficients f_nm - Unified computation following eq. B10
# ============================================================================

def compute_f_nm_unified(p, alpha, beta, gamma, time_dict, h_power):
    """
    Unified computation of f^(m)_nm multipoles for any m following eq. B10.

    Can compute f^(-4), f^(-2), f^(0), f^(2), f^(4) using the same pattern.

    Pattern from eq. B10:
    - f^(m)_{0,0}   = (β - α)/2 - 2γ   [with appropriate coeff index]
    - f^(m)_{2,-2}  = (α - β)/4 + γ     [only for h_power=0]
    - f^(m)_{0,-2}  = H²/2 * (β/2 - α)  [uses next coeff index]
    - f^(m)_{-2,-2} = α/4               [only for h_power=0,4]

    Parameters:
    -----------
    p : parameter object
    alpha, beta, gamma : dict
        Coefficient dictionaries with keys {0, 1, 2}
    time_dict : dict
        Cosmological time-dependent functions
    h_power : int
        Power m in f^(m): -4, -2, 0, 2, or 4

    Returns:
    --------
    list or array
        list of UnivariateSplines
    """

    # Helper functions following eq. B10 pattern
    # Return 0 when the component doesn't exist for that h_power
    def f_00(i):
        """f^(i)_{0,0} = H^|i| * ((β - α)/2 - 2γ), exists for i ≤ 0"""
        if i > 0:
            return np.zeros_like(time_dict['Ha'])
        index = np.abs(i)//2
        H_power = time_dict['Ha']**(np.abs(i))
        return H_power*((beta[index]-alpha[index])/2. - 2.*gamma[index])

    def f_0m2(i):
        """f^(i)_{0,-2} = H^|i-2| * (β/2 - α)/2, exists for -2 ≤ i ≤ 2"""
        if i < -2 or i > 2:
            return np.zeros_like(time_dict['Ha'])
        index = np.abs(i-2)//2
        H_power = time_dict['Ha']**(np.abs(i-2))
        return H_power*0.5*(beta[index]/2.-alpha[index])

    def f_m2m2(i):
        """f^(i)_{-2,-2} = H^|i-4| * α/4, exists for 0 ≤ i ≤ 4"""
        if i < 0:
            return np.zeros_like(time_dict['Ha'])
        index = np.abs(i-4)//2
        H_power = time_dict['Ha']**(np.abs(i-4))
        return H_power*alpha[index]/4.

    # Compute all components and filter out zero arrays
    # Note: f_{2,-2} = -f_{0,0}/2 is recovered later (not stored here)
    components = [c for c in [f_00(h_power), f_0m2(h_power), f_m2m2(h_power)] if not np.all(c == 0)]

    # Apply prefactor: D² * H/a
    if p.which == 'F2':
        prefactor = time_dict['Da']**2 * time_dict['Ha'] / time_dict['a']
    elif p.which == 'G2': 
        prefactor = -time_dict['Da']**2 * time_dict['Ha'] / time_dict['a']
    else: # dv2
        prefactor = time_dict['Da']**2 * time_dict['Ha'] / time_dict['a'] * time_dict['Ha'] * time_dict['mathcalR']

    # Create splines
    splines = []
    for comp in components:
        splines.append(UnivariateSpline(time_dict['ra'], comp * prefactor, k=5, s=0))

    return splines



# ============================================================================
# Radiation f_nm
# ============================================================================

def compute_radiation_f_nm(p, time_dict):
    """
    Compute radiation-specific multipoles.
    Only returns independent components: [fm2R_0, fm4R_0]
    Note: fm2R_1 = -fm2R_0/2, fm4R_1 = -fm4R_0/2 (derived in general_ps.py)
    """
    H2 = time_dict['Ha']**2
    H4 = time_dict['Ha']**4
    if p.which=='dv2':
        prefactor = time_dict['Ha']*time_dict['mathcalR']*time_dict['Da'] / time_dict['a'] * time_dict['Ha']/time_dict['a']*time_dict['Da']**2
    else:
        prefactor = time_dict['Da'] / time_dict['a'] * time_dict['Ha']/time_dict['a']*time_dict['Da']**2

    # Only compute independent components
    base_fm2 = time_dict['fa'] + 1.5*time_dict['Oma']
    fm2R_0 = base_fm2 * H2

    if p.which == 'F2':
        base_fm4 = 3.*time_dict['fa']*(time_dict['fa'] + 1.5*time_dict['Oma'])
    else:
        base_fm4 = 3.*(time_dict['fa'] - 1)*(time_dict['fa'] + 1.5*time_dict['Oma'])
    fm4R_0 = base_fm4 * H4

    # Create splines only for independent components
    splines = []
    splines.append(UnivariateSpline(time_dict['ra'], prefactor * fm2R_0, k=5, s=0))
    splines.append(UnivariateSpline(time_dict['ra'], prefactor * fm4R_0, k=5, s=0))

    # Return list of 2 splines: [fm2R_0, fm4R_0]
    return splines 

# Product rule derivatives using Leibniz rule: d^n(uv) = sum_k C(n,k) u^(k) v^(n-k)
def product_deriv(n, fctr_derivs, W_derivs_list):
    """Compute nth derivative of fctr*W product"""
    return sum(comb(n, k) * fctr_derivs[k] * W_derivs_list[n - k] for k in range(n + 1))
 

def fct_of_r_analytical(p, ell_list, r_list, time_dict, window_args, lterm_list, W_derivs_list=None):
    """
    Optimized version using cached SymPy derivatives

    Parameters:
    -----------
    lterm_list : list of str
        List of lterm values to compute (e.g., ['density', 'rsd'])
    W_derivs_list : list, optional
        Precomputed window derivatives. If None, will compute from window_args.

    Returns:
    --------
    dict: y_list organized by lterm
    """

    def compute_L3_f_analytical(f, df, d2f, d3f, d4f, d5f, d6f, r, alpha):
        """
        Compute L³[f] where L = -d²/dr² + 2/r*d/dr + alpha/r²
        """

        c6 = -1
        c5 = 6/r
        c4 = 3*(alpha - 8)/r**2
        c3 = 24*(3 - alpha)/r**3
        c2 = 3*(-alpha**2 + 34*alpha - 48)/r**4
        c1 = 18*(alpha**2 - 14*alpha + 8)/r**5
        c0 = alpha*(alpha**2 - 38*alpha + 280)/r**6

        return c6*d6f + c5*d5f + c4*d4f + c3*d3f + c2*d2f + c1*df + c0*f


    # Use precomputed derivatives if available, otherwise create new evaluator
    if W_derivs_list is None:
        W_derivs = WindowDerivatives(window_args)
    else:
        W_derivs = None  # Won't need to call get_all_derivatives

    y_list = {'r_list': r_list, 'ell_list': ell_list}
    if p.which in ['F2', 'G2', 'dv2']:

        if p.which == 'F2': derive_start=0
        elif p.which == 'G2': derive_start=2
        else: derive_start=1  # dv2: fctr already includes mathcalR*Ha, take gradient with derive_start=1

        # Setup fctr_list and qterm_list based on radiation vs non-radiation
        if p.rad:
            # Radiation case: use radiation multipoles (only independent components)
            fctr_list = compute_radiation_f_nm(p, time_dict)  # Returns [fm2R_0, fm4R_0]
            qterm_list = list(range(len(fctr_list)))  # [0, 1]
            output_key = '{}_rad'.format(p.which)
        else:
            # Non-radiation case: compute Am terms
            # Handle Newton cases 
            if p.Newton and p.which in ['F2']:
                return 0  # No Newtonian terms for F2

            # Get coefficients
            alpha_coeff, beta_coeff, gamma_coeff = get_coefficients(p, time_dict)

            if p.which == 'F2':
                # For F2: use fm2 and fm4 (only independent components)
                # fm2: [f_00, f_0m2], fm4: [f_00], concatenated to [fm2_0, fm2_2, fm4_0]
                fctr_list_fm2 = compute_f_nm_unified(p, alpha_coeff, beta_coeff, gamma_coeff, time_dict, h_power=-2)
                fctr_list_fm4 = compute_f_nm_unified(p, alpha_coeff, beta_coeff, gamma_coeff, time_dict, h_power=-4)
                fctr_list = fctr_list_fm2 + fctr_list_fm4
                qterm_list = list(range(len(fctr_list)))  # [0, 1, 2]

            else:
                # For G2 and dv2: use f0 and optionally fm2 (only independent components)
                # f0: [f_00, f_0m2, f_m2m2], take first 2: [f0_0, f0_2]
                fctr_list_f0 = compute_f_nm_unified(p, alpha_coeff, beta_coeff, gamma_coeff, time_dict, h_power=0)

                if p.Newton:
                    # Newton case for G2: only f0 components
                    fctr_list = fctr_list_f0  # [f0_0, f0_2]
                    qterm_list = list(range(len(fctr_list)))  # [0, 1]
                else:
                    # Full GR case: include both f0 and fm2
                    # fm2: [f_00, f_0m2], take first 1: [fm2_0]
                    fctr_list_fm2 = compute_f_nm_unified(p, alpha_coeff, beta_coeff, gamma_coeff, time_dict, h_power=-2)
                    fctr_list = fctr_list_f0 + fctr_list_fm2  # [f0_0, f0_2, fm2_0]
                    qterm_list = list(range(len(fctr_list)))  # [0, 1, 2]

            output_key = p.which

        # Initialize y_list
        y_list[output_key] = np.zeros((2, len(qterm_list), len(ell_list), len(r_list)), dtype=np.float64)

        # Compute fctr derivatives and apply operators
        # derive_start is already set: F2=0, dv2=1, G2=2
        max_deriv = 2 + derive_start

        # For F2/G2/dv2: compute b1 derivatives if available
        if p.which=='F2' and 'data' in time_dict and 'b1' in time_dict['data']:
            print('         Adding linear bias b1 to F2/G2/dv2 terms')
            b1_spline = UnivariateSpline(time_dict['data']['r'], time_dict['data']['b1'], k=5, s=0)
            b1_derivs_list = compute_spline_derivatives(b1_spline, time_dict['data']['r'], r_list,
                                                        max_deriv=max_deriv+derive_start, smooth_s=1e-6)
            b1_derivs = np.array(b1_derivs_list)
            use_b1 = True
        else:
            b1_derivs = None
            use_b1 = False

        for qt_ind, qt in enumerate(qterm_list):
            fctr = fctr_list[qt]

            # Compute fctr and its derivatives using the new function
            fctr_derivs_list = compute_spline_derivatives(fctr, time_dict['ra'], r_list,
                                                          max_deriv=max_deriv, smooth_s=0)
            fctr_derivs = np.array(fctr_derivs_list)

            # Get analytical derivatives of W
            if W_derivs_list is None:
                W_derivs_list_local = W_derivs.get_all_derivatives(r_list, r_power=0, max_deriv=max_deriv+derive_start)
            else:
                # Use precomputed derivatives (check if we have enough)
                required_derivs = max_deriv + derive_start + 1
                if len(W_derivs_list) < required_derivs:
                    raise ValueError(f"W_derivs_list has {len(W_derivs_list)} derivatives but need {required_derivs}")
                W_derivs_list_local = W_derivs_list[:required_derivs]

            # Compute fctr * W derivatives
            fctr_W_derivs = [product_deriv(i+derive_start, fctr_derivs, W_derivs_list_local) for i in range(3)]

            # If b1 is present, apply second product rule: b1 * (fctr * W)
            if use_b1:
                f, df, d2f = [product_deriv(i, b1_derivs, fctr_W_derivs) for i in range(3)]
            else:
                f, df, d2f = fctr_W_derivs

            # Store both levels and apply mathcalD operator
            for ind_ell, ell in enumerate(ell_list):
                alpha = ell*(ell+1) - 2.
                y_list[output_key][0, qt_ind, ind_ell] = f
                y_list[output_key][1, qt_ind, ind_ell] = -d2f + 2./r_list*df + alpha/r_list**2*f

    else:
        # lterm_list is now passed as argument

        if p.which=='d2v':
            qterm_list=[1,2,3]
        elif p.which in ['d1v', 'd1d']:
            qterm_list=[1,2]
        elif p.which in ['d0d']:
            qterm_list=[1]
        elif p.which=='d3v':
            qterm_list=[1,2,3,4]
        else:
            qterm_list=[0]

        for lterm in lterm_list:
            print('     lterm={}'.format(lterm))
            y_list[lterm] = np.zeros((4, len(qterm_list), len(ell_list), len(r_list)), dtype=np.float64)

            # Create spline for fctr(r)
            if lterm == 'density':
                fctr = UnivariateSpline(time_dict['ra'], time_dict['Ha']/time_dict['a']*time_dict['Da'], k=5, s=0)
                derive_start = 0
            elif lterm == 'rsd': # -D*f*H/a
                fctr = UnivariateSpline(time_dict['ra'], -time_dict['Ha']/time_dict['a']*time_dict['Da']*time_dict['fa'], k=5, s=0)
                derive_start = 2
            elif lterm == 'doppler': # H*D*H/a*f*R
                fctr = UnivariateSpline(time_dict['ra'], \
                        time_dict['Ha']*time_dict['Da']*time_dict['Ha']/time_dict['a']*time_dict['fa']*time_dict['mathcalR'], k=5, s=0)
                derive_start = 1
            elif lterm == 'pot_gr': # H/a*D*3*f*H**2
                fctr = UnivariateSpline(time_dict['ra'], \
                            time_dict['Ha']/time_dict['a']*time_dict['Da']*(3.*time_dict['fa']*time_dict['Ha']**2), k=5, s=0)
                derive_start = 0
            elif lterm == 'pot': # H/a*D*(1.-R)/a
                fctr = UnivariateSpline(time_dict['ra'], \
                            time_dict['Ha']/time_dict['a']*time_dict['Da']*(1.-time_dict['mathcalR'])/time_dict['a'], k=5, s=0)
                derive_start = 0
            elif lterm == 'dpot': # -D*(f-1)/a*H/a
                fctr = UnivariateSpline(time_dict['ra'], -time_dict['Da']*(time_dict['fa']-1.)*time_dict['Ha']/time_dict['a']**2, k=5, s=0)
                derive_start = 0
            else:
                print('no code for {}'.format(lterm))
                raise ValueError(f"Invalid 'lterm' parameter: {lterm}")
            
            # Compute fctr derivatives using the new function
            # Use time_dict['ra'] as the full data grid for accuracy
            fctr_derivs_list = compute_spline_derivatives(fctr, time_dict['ra'], r_list, max_deriv=11, smooth_s=0)
            fctr_derivs = np.array(fctr_derivs_list)

            # For density term: compute b1 derivatives if available
            if lterm == 'density' and 'data' in time_dict and 'b1' in time_dict['data']:
                print('Adding linear bias b1 to linear terms')
                b1_spline = UnivariateSpline(time_dict['data']['r'], time_dict['data']['b1'], k=5, s=0)
                b1_derivs_list = compute_spline_derivatives(b1_spline, time_dict['data']['r'], r_list,
                                                            max_deriv=11, smooth_s=1e-6)
                b1_derivs = np.array(b1_derivs_list)
                #np.save('b1', b1_derivs)


                use_b1 = True
            else:
                b1_derivs = None
                use_b1 = False

            for qt_ind, qt in enumerate(qterm_list):
                r_power_and_derivative = qt-1  if qt in [4, 3, 2] else 0

                # FIXED: Apply derive_start to fctr*W first, then qterm operations
                # Step 1: Compute B and its derivatives analytically
                # B^(k) = d^(derive_start+k)/dx^(derive_start+k)[fctr * W]
                # For qterm operations, product_deriv(n+6,...) needs B_derivs[0] through B_derivs[n+6]
                n = r_power_and_derivative
                max_B_deriv = n + 7 if n > 0 else 7  # n+7 because we need indices 0 through n+6
                max_W_deriv = derive_start + max_B_deriv - 1  # product_deriv needs this many W derivatives

                if W_derivs_list is None:
                    W_derivs_base = W_derivs.get_all_derivatives(r_list, r_power=0, max_deriv=max_W_deriv)
                else:
                    required_derivs = max_W_deriv + 1
                    if len(W_derivs_list) < required_derivs:
                        raise ValueError(f"W_derivs_list has {len(W_derivs_list)} derivatives but need {required_derivs}")
                    W_derivs_base = W_derivs_list[:required_derivs]

                # Compute fctr * W derivatives
                fctr_W_derivs = [product_deriv(derive_start + i, fctr_derivs, W_derivs_base) for i in range(max_B_deriv)]

                # If b1 is present (density term), apply second product rule: b1 * (fctr * W)
                if use_b1:
                    B_derivs = [product_deriv(i, b1_derivs, fctr_W_derivs) for i in range(max_B_deriv)]
                else:
                    B_derivs = fctr_W_derivs

                # Step 2: Apply qterm by computing derivatives of r^n * B
                if n > 0:
                    # Compute derivatives of r^n analytically
                    # product_deriv(n+6,...) needs rn_derivs[0] through rn_derivs[n+6]
                    rn_derivs = [r_list**n]  # 0th derivative
                    for k in range(1, n + 7):  # Need indices 0 through n+6
                        # d^k/dx^k[r^n] = n*(n-1)*...*(n-k+1) * r^(n-k)
                        coeff = np.prod([n - j for j in range(k)])
                        rn_derivs.append(coeff * r_list**(n - k) if n >= k else np.zeros_like(r_list))

                    # Compute d^n/dx^n[r^n * B], d^(n+1)/dx^(n+1)[r^n * B], etc.
                    # because mathematica.py returns d^n/dx^n[x^n * B] for qterm
                    f, df, d2f, d3f, d4f, d5f, d6f = \
                        [product_deriv(n + i, B_derivs, rn_derivs) for i in range(7)]
                else:
                    # No qterm, just use B derivatives directly
                    f, df, d2f, d3f, d4f, d5f, d6f = B_derivs
                 
                # Compute y_list by applying D operator
                for ind_ell, ell in enumerate(ell_list):
                    alpha = ell*(ell+1) - 2.
                    
                    y_list[lterm][0, qt_ind, ind_ell] = f
                    y_list[lterm][1, qt_ind, ind_ell] = -d2f + 2./r_list*df + alpha/r_list**2*f
                    y_list[lterm][2, qt_ind, ind_ell] = (d4f - 4./r_list*d3f 
                                        +(8./r_list**2 - 2.*alpha/r_list**2)*d2f 
                                        +(-8./r_list**3 + 8.*alpha/r_list**3)*df + 
                                         (alpha**2/r_list**4 - 10.*alpha/r_list**4)*f)
                    
                    y_list[lterm][3, qt_ind, ind_ell] = compute_L3_f_analytical(f, df, d2f, d3f, d4f, d5f, d6f, r_list, alpha)
   
    return y_list


def get_bispectrum_kernels_analytical(p, ell_list, r_list, time_dict, window_args=None, W_derivs_list=None):
    """
    Compute A0, A2, A4 kernel factors for bispectrum using analytical derivatives.

    For F2/G2/dv2:
        A0 = f^(0)_nm * D² * W                   (no derivatives)
        A2 = operator(f^(2)_nm * D² * W)         (mathcalD/∇²/∇ applied)
        A4 = mathcalD(operator(f^(4)_nm * D² * W)) (double operator)

    For other cases (d2vd0d, d1vd1d, etc.):
        Only A0 is computed from direct formulas

    Parameters:
    -----------
    p : parameter object
        Must have .which attribute
    ell_list : int or array-like
        Angular multipole(s) for mathcalD operator
    r_list : array
        Radial coordinates
    time_dict : dict
        Contains cosmological functions (Dr, fr, Ha, etc.)
    window_args : tuple, optional
        (r0, ddr, normW) for window function. Required if W_derivs_list is None.
    W_derivs_list : list of arrays, optional
        Precomputed window derivatives [W, dW, d2W, ...]. If provided, window_args is ignored.
        Must contain at least 5 derivatives (indices 0-4).

    Returns:
    --------
    dict with keys 'A0', 'A2', 'A4' containing arrays of shape (n_ell, n_components, n_r)
    (A2 and A4 may be None for non-F2/G2/dv2 cases)
    """

    # Handle both single ell and list of ells
    if not isinstance(ell_list, (list, np.ndarray)):
        ell_list = [ell_list]
    ell_list = np.array(ell_list)
    n_ell = len(ell_list)
    n_r = len(r_list)

    if p.which in ['F2', 'G2', 'dv2']:
        # F2/G2/dv2 cases: compute A0, A2, A4 using f-coefficients

        # Get coefficients (independent of ell)
        alpha_coeff, beta_coeff, gamma_coeff = get_coefficients(p, time_dict)

        # Determine derivative orders needed
        max_deriv_inner = 1 if p.which == 'dv2' else 2  # For A2
        max_deriv_total = 4  # For A4

        # Initialize window derivatives - compute once for all ells (or use precomputed)
        if W_derivs_list is None:
            if window_args is None:
                raise ValueError("Either window_args or W_derivs_list must be provided")
            W_derivs = WindowDerivatives(window_args)
            W_derivs_list = W_derivs.get_all_derivatives(r_list, r_power=0, max_deriv=max_deriv_total)
        else:
            # Verify we have enough derivatives
            if len(W_derivs_list) < max_deriv_total + 1:
                raise ValueError(f"W_derivs_list must contain at least {max_deriv_total+1} derivatives (0 to {max_deriv_total})")

        # Determine number of components for each kernel
        # A0: for F2 can have up to 4 components, for G2/dv2 typically 1-2
        f0_splines = compute_f_nm_unified(p, alpha_coeff, beta_coeff, gamma_coeff, time_dict, h_power=0)
        n_A0_comp = len(f0_splines)

        # Initialize output arrays: (n_ell, n_components, n_r)
        A0_all = np.zeros((n_ell, n_A0_comp, n_r))
        A2_all = np.zeros((n_ell, 2, n_r))  # Always 2 components for A2
        A4_all = np.zeros((n_ell, 1, n_r))  # Always 1 component for A4

        # Get f^(2) and f^(4) splines (independent of ell)
        f2_splines = compute_f_nm_unified(p, alpha_coeff, beta_coeff, gamma_coeff, time_dict, h_power=2)
        [f4_spline] = compute_f_nm_unified(p, alpha_coeff, beta_coeff, gamma_coeff, time_dict, h_power=4)

        # ====================================================================
        # Precompute b1 derivatives if available (for F2 only)
        # ====================================================================
        if p.which == 'F2' and 'data' in time_dict and 'b1' in time_dict['data']:
            print('Adding linear bias b1 to F2 kernels')
            b1_spline = UnivariateSpline(time_dict['data']['r'], time_dict['data']['b1'], k=5, s=0)
            b1_derivs_list = compute_spline_derivatives(b1_spline, time_dict['data']['r'], r_list,
                                                        max_deriv=max_deriv_total, smooth_s=1e-6)
            b1_derivs = np.array(b1_derivs_list)
            use_b1 = True
        else:
            b1_derivs = None
            use_b1 = False

        # ====================================================================
        # Precompute all spline derivatives ONCE (independent of ell!)
        # ====================================================================
        # A0: f^(0) evaluated at r_list
        f0_values = np.array([f0_spline(r_list) for f0_spline in f0_splines])  # (n_A0_comp, n_r)

        # A2: f^(2) and its derivatives
        f2_derivs_list = []  # List of (max_deriv_inner+1, n_r) arrays
        for f2_spline in f2_splines:
            fctr_derivs_list = compute_spline_derivatives(f2_spline, time_dict['ra'], r_list,
                                                          max_deriv=max_deriv_inner, smooth_s=0)
            fctr_derivs = np.array(fctr_derivs_list)
            f2_derivs_list.append(fctr_derivs)

        # A4: f^(4) and its derivatives
        f4_derivs_list = compute_spline_derivatives(f4_spline, time_dict['ra'], r_list,
                                                    max_deriv=max_deriv_total, smooth_s=0)
        f4_derivs = np.array(f4_derivs_list)

        # Precompute products of fctr*W for A2
        f2_products_list = []  # List of product derivatives for each f2
        for fctr_derivs in f2_derivs_list:
            if p.which == 'F2':
                # Compute fctr * W
                fctr_W = [product_deriv(j, fctr_derivs, W_derivs_list) for j in range(3)]
                # If b1 present, multiply by b1
                if use_b1:
                    f, df, d2f = [product_deriv(j, b1_derivs, fctr_W) for j in range(3)]
                else:
                    f, df, d2f = fctr_W
                f2_products_list.append((f, df, d2f))
            elif p.which == 'G2':
                d2f = product_deriv(2, fctr_derivs, W_derivs_list)
                f2_products_list.append((None, None, d2f))
            else:  # dv2
                df = product_deriv(1, fctr_derivs, W_derivs_list)
                f2_products_list.append((None, df, None))

        # Precompute products of f4*W for A4
        # Compute fctr * W
        fctr_W_f4 = [product_deriv(j, f4_derivs, W_derivs_list) for j in range(5)]
        # If b1 present for F2, multiply by b1
        if use_b1:
            f, df, d2f, d3f, d4f = [product_deriv(j, b1_derivs, fctr_W_f4) for j in range(5)]
        else:
            f, df, d2f, d3f, d4f = fctr_W_f4
        f4_products = (f, df, d2f, d3f, d4f)

        # ====================================================================
        # Loop over ells (now just applying ell-dependent alpha factors)
        # ====================================================================
        for ell_idx, ell in enumerate(ell_list):
            alpha = ell*(ell+1) - 2.

            # A0: f^(0) * W (already precomputed)
            A0_all[ell_idx, :, :] = f0_values * W_derivs_list[0]

            # A2: apply ell-dependent operator
            for i, products in enumerate(f2_products_list):
                if p.which == 'F2':
                    f, df, d2f = products
                    A2_all[ell_idx, i, :] = -d2f + 2./r_list*df + alpha/r_list**2*f
                elif p.which == 'G2':
                    _, _, d2f = products
                    A2_all[ell_idx, i, :] = d2f
                else:  # dv2
                    _, df, _ = products
                    A2_all[ell_idx, i, :] = df

            # A4: apply ell-dependent operators
            f, df, d2f, d3f, d4f = f4_products

            # Apply inner operator to get y and its derivatives
            if p.which == 'F2':
                y = -d2f + 2./r_list*df + alpha/r_list**2*f
                dy = -d3f + 2./r_list*d2f + (alpha-2.)/r_list**2*df - 2.*alpha/r_list**3*f
                d2y = -d4f + 2./r_list*d3f + (alpha-4.)/r_list**2*d2f - 4.*(alpha-1.)/r_list**3*df + 6.*alpha/r_list**4*f
            elif p.which == 'G2':
                y, dy, d2y = d2f, d3f, d4f
            else:  # dv2
                y, dy, d2y = df, d2f, d3f

            # Apply outer mathcalD
            A4_all[ell_idx, 0, :] = -d2y + 2./r_list*dy + alpha/r_list**2*y

        return {'r_list': r_list, 'ell_list': ell_list, 'A0': A0_all, 'A2': A2_all, 'A4': A4_all}

    else:
        # Other cases (d2vd0d, d1vd1d, d1vd0d, etc.): compute A0 directly from formulas
        # These cases don't depend on ell, so compute once and tile across all ells

        # Compute cosmological factors on ra grid, then interpolate to r_list
        ra = time_dict['ra']

        # Compute cosmological factor on ra grid (without window)
        if p.which in ['d2vd0d', 'd1vd1d', 'd1vd0d']:
            cosmo_factor_ra = time_dict['Da']**2 * time_dict['fa']
            if p.which in ['d1vd0d']:
                cosmo_factor_ra *= time_dict['Ha'] * time_dict['mathcalR']

        elif p.which in ['d0dd0d']:
            # Similar to d2vd0d but with b2/2 instead of fa
            if 'data' not in time_dict or 'b2' not in time_dict['data']:
                raise ValueError("d0dd0d requires b2 in time_dict['data']")
            # For d0dd0d: need to handle b2 separately to avoid extrapolation
            # Don't create cosmo_factor_ra here, will handle below
            cosmo_factor_ra = None
            use_b2 = True

        elif p.which in ['d1vdod']:
            cosmo_factor_ra = time_dict['Da'] * time_dict['fa']

        elif p.which in ['d0pd3v', 'd0pd1d', 'd1vd2p']:
            cosmo_factor_ra = time_dict['Da']**2 / time_dict['Ha'] / time_dict['a']
            if 'v' in p.which:
                cosmo_factor_ra *= time_dict['fa']

        else:
            # Default case: d2vd2v, d1vd2v, davd1v, etc.
            cosmo_factor_ra = time_dict['Da']**2 * time_dict['fa']**2
            if p.which in ['d1vd2v']:
                cosmo_factor_ra *= time_dict['Ha'] * \
                        (1. + 3.*time_dict['dHa']/time_dict['Ha']**2 + 4./time_dict['Ha']/ra)
            elif p.which in ['davd1v']:
                cosmo_factor_ra *= time_dict['Ha']

        # Get window function on r_list
        if W_derivs_list is None:
            if window_args is None:
                raise ValueError("Either window_args or W_derivs_list must be provided")
            W_derivs = WindowDerivatives(window_args)
            Wr = W_derivs.get_all_derivatives(r_list, r_power=0, max_deriv=0)[0]
        else:
            # Use precomputed (just need 0th derivative = window function itself)
            Wr = W_derivs_list[0]

        # Handle d0dd0d separately to avoid extrapolation of b2
        if p.which in ['d0dd0d']:
            # Create Da² * H/a spline on ra grid
            cosmo_Da2_Ha_ra = time_dict['Da']**2 * time_dict['Ha'] / time_dict['a']
            cosmo_Da2_Ha_spline = UnivariateSpline(ra, cosmo_Da2_Ha_ra, s=0, ext=0)
            cosmo_Da2_Ha = cosmo_Da2_Ha_spline(r_list)

            # Create b2/2 spline on data grid
            b2_spline = UnivariateSpline(time_dict['data']['r'], time_dict['data']['b2']/2.0, k=5, s=0)
            #b2_spline = UnivariateSpline(time_dict['ra'], time_dict['Da']**2 * time_dict['Ha'] / time_dict['a'], k=5, s=0)
            b2_half = b2_spline(r_list)

            # Multiply: (Da² * H/a) * (b2/2) * W
            A0_tab = cosmo_Da2_Ha * b2_half * Wr
        else:
            # Standard case: create spline and interpolate to r_list
            # For quadratic terms, include H/a factor to convert W to W_tilde = H/a * W (as in old code)
            cosmo_spline = UnivariateSpline(ra, cosmo_factor_ra * time_dict['Ha'] / time_dict['a'], s=0, ext=0)
            cosmo_factor = cosmo_spline(r_list)

            # Multiply by window
            A0_tab = cosmo_factor * Wr

        if p.which in ['d2vd0d', 'd1vd1d', 'd1vd2v', 'd1vdod', 'd0pd3v', 'davd1v']:
            A0_tab*=-1

        # Apply r-power division (on r_list, not ra)
        try:
            A0_tab /= r_list**(int(p.which[1]) + int(p.which[4]))
        except ValueError:
            if p.which == 'davd1v':
                A0_tab /= r_list**2

            try:
                A0_tab /= r_list**(int(p.which[1]))
            except ValueError:
                A0_tab /= r_list**(int(p.which[4]))

        # Reshape: add extra dimension and apply factor
        A0_tab = A0_tab[None, :]
        
        # Tile across all ells: shape (n_ell, n_components, n_r)
        A0_all = np.tile(A0_tab[None, :, :], (n_ell, 1, 1))

        # For these cases, there are no A2 or A4 contributions
        return {'r_list': r_list, 'ell_list': ell_list, 'A0': A0_all, 'A2': None, 'A4': None}
