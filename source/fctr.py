import numpy as np
import os, copy
from numba import njit
from math import comb, factorial
import cubature, time, h5py
from scipy.integrate import simpson
from sympy.physics.wigner import wigner_3j
from filelock import FileLock
from scipy.interpolate import UnivariateSpline
import sympy as sp
from scipy.integrate import quad
from scipy.special import erf  # Vectorized erf for lambdify

from param_used import *
import lincosmo

# TESTING FLAG: Set to False to disable b_s (keeping b1) for comparison
COMPUTE_BS = True
COMPUTE_B1 = True
COMPUTE_C1_C2 = True # Set to False to disable GR bias corrections (c1 and c2)
COMPUTE_FNL = True # Set to False to disable the delta_{2,fNL} scale-dependent bias term



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

    return np.array(derivs)


# Optimized version that caches the SymPy derivatives
class WindowDerivatives:
    """
    Cache SymPy derivatives for W(r) and r^n * W(r) expressions for efficiency
    """
    def __init__(self, window_type, window_args, max_r_power=3, max_derivative=11):
        self.window_type = window_type

        if window_type=='ska':
            self.xmin, self.xmax, self.H_over_a_data, self.sigma_z, self.n_angular_data = window_args
        elif window_type=='nbody':
            # Backward compatibility
            self.xmin, self.xmax, self.H_over_a_data, self.sigma_z = window_args
            self.n_angular_data = None
        elif window_type=='euclid':
            # based on eq 112-115 of ref 1910.09273
            self.bin_idx, self.H_over_a_data = window_args

            self.BIN_EDGES = np.array([0.001, 0.56, 0.79, 1.02, 1.32, 2.50])
            #[0.001, 0.42, 0.56, 0.68, 0.79, 0.90, 1.02, 1.15, 1.32, 1.58, 2.50]

            # Photo-z parameters (Table 5 of arXiv:1910.09273)
            self.CB, self.ZB, self.SIGMAB = 1.0, 0.0, 0.05
            self.CO, self.ZO, self.SIGMAO = 1.0, 0.1, 0.05
            self.FOUT = 0.1

            # Optimised R0_eff per bin (matched to exact n_i(z) shape via L2 minimisation)
            self.R0_EFF_LIST = np.array([2975.8, 2448.7, 2101.7, 1821.8,1425.7])
            #[ 2293.5, 1922.7, 1702.4, 1553.2, 1438.6, 1339.7, 1246.7, 1157.5, 1055.7, 883.3]

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

        elif self.window_type=='ska':
            # Use sympy versions:
            window_lower = 0.5 * (1 + sp.erf((sp_x - sp_xmin) / (sp.sqrt(2) * sp_sigma_z)))
            window_upper = 0.5 * (1 + sp.erf((sp_xmax - sp_x) / (sp.sqrt(2) * sp_sigma_z)))
            W_unnorm = window_lower * window_upper

        elif self.window_type == 'euclid':
            # r-space approximation: same functional form as z-space but with
            # per-bin optimised R0_eff and all z-parameters converted via comoving distance.
            zlo = self.BIN_EDGES[self.bin_idx]
            zhi = self.BIN_EDGES[self.bin_idx + 1]
            z_c = (zlo + zhi) / 2
            R0_eff = self.R0_EFF_LIST[self.bin_idx]

            # r-space bin limits: comoving_distance(c*z + z_off)
            rlo_b = lincosmo.get_distance(self.CB * zlo + self.ZB)[0]
            rhi_b = lincosmo.get_distance(self.CB * zhi + self.ZB)[0]
            rlo_o = lincosmo.get_distance(self.CO * zlo + self.ZO)[0]
            rhi_o = lincosmo.get_distance(self.CO * zhi + self.ZO)[0]

            # sigma in r-space: r(z_c + sigma*(1+z_c)) - r(z_c)
            r_c   = lincosmo.get_distance(z_c)[0]
            sigma_r_b = lincosmo.get_distance(z_c + self.SIGMAB*(1+z_c))[0] - r_c
            sigma_r_o = lincosmo.get_distance(z_c + self.SIGMAO*(1+z_c))[0] - r_c

            # SymPy W_unnorm(r): nz_r * sel_r
            nz_r   = (sp_x / R0_eff)**2 * sp.exp(-(sp_x / R0_eff)**sp.Rational(3, 2))
            sel_b  = (1 - self.FOUT)/self.CB * (sp.erf((sp_x - rlo_b)/(sp.sqrt(2)*sigma_r_b)) - sp.erf((sp_x - rhi_b)/(sp.sqrt(2)*sigma_r_b))) / 2
            sel_o  = self.FOUT/self.CO       * (sp.erf((sp_x - rlo_o)/(sp.sqrt(2)*sigma_r_o)) - sp.erf((sp_x - rhi_o)/(sp.sqrt(2)*sigma_r_o))) / 2
            W_unnorm = nz_r * (sel_b + sel_o)

            # integration bounds for the derivative loop
            self.xmin    = min(rlo_b, rlo_o) - 3*sigma_r_b
            self.xmax    = max(rhi_b, rhi_o) + 3*sigma_r_b
            self.sigma_z = sigma_r_b

        # Compute normalization: ∫ (H/a * W_unnorm) dr or ∫ (n_angular * H/a * W_unnorm) dr
        if self.H_over_a_data is not None:
            # Unpack H/a data: (ra_grid, H_over_a_values)
            ra_grid, H_over_a_values = self.H_over_a_data

            # Create H/a spline
            self.H_over_a_spline = UnivariateSpline(ra_grid, H_over_a_values, s=0, ext=0, k=5)

            # Create n_angular spline if available and not nbody
            if self.n_angular_data is not None and self.window_type != 'nbody':
                r_nz_grid, n_angular_values = self.n_angular_data
                self.n_angular_spline = UnivariateSpline(r_nz_grid, n_angular_values, s=0, ext=0, k=5)
                print(f"    Using n(z) angular normalization (SKA-type window)")
            else:
                self.n_angular_spline = None

            # Lambdify the unnormalized window for numerical integration
            W_unnorm_func = sp.lambdify(sp_x, W_unnorm.subs({sp_xmin: self.xmin,
                                                              sp_xmax: self.xmax,
                                                              sp_sigma_z: self.sigma_z}), 'numpy')

            # Integrand: includes n_angular if available (for SKA-type surveys)
            def integrand(r):
                H_over_a = self.H_over_a_spline(r)
                W = W_unnorm_func(r)
                if self.n_angular_spline is not None:
                    return self.n_angular_spline(r) * H_over_a * W
                else:
                    return H_over_a * W

            sp_normW, _ = quad(integrand, max(1.0, self.xmin - 20*self.sigma_z), self.xmax + 20*self.sigma_z, epsrel=1e-4, epsabs=0)
            integral_type = "n_angular*H/a*W" if self.n_angular_spline is not None else "H/a*W"
            print(f"        Computed normW = ∫({integral_type})dr = {sp_normW:.4e}")
        else:
            self.H_over_a_spline = None
            self.n_angular_spline = None
            # Fallback: use analytical formula for ∫ W dr (old behavior)
            sp_normW = 1./4.*self.sigma_z*(1. + 1./np.tanh((self.xmax - self.xmin)/self.sigma_z))*2./self.sigma_z*(self.xmax-self.xmin)
            print(f"  Warning: H/a data not provided, using ∫W dr normalization = {sp_normW:.6e}")

        # Store normW as instance variable for later use
        self.normW = sp_normW

        # Normalized window: W = W_unnorm / normW
        base_W = W_unnorm / sp_normW
        
        # For each r power and derivative order
        for r_power in range(self.max_r_power + 1):  # 0 to max_r_power
            if self.window_type == 'euclid':  # slow symbolic diff: show progress
                print(f'  compiling derivatives: r_power {r_power}/{self.max_r_power}')
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

    def get_lensing_derivatives(self, x, W_derivs_list, max_deriv=11):
        """
        Derivatives of the lensing efficiency window, evaluated on x (ascending):

            W_phi(r') = ∫_{r'}^∞ dr Wh(r) (r-r')/(r r') = g0(r')/r' - g1(r'),
            Wh = (H/a)*n_angular*W/normW,  g0 = ∫_{r'}^∞ Wh dr,  g1 = ∫_{r'}^∞ Wh/r dr.

        Wh carries the H/a Jacobian that the other lterms keep inside fctr: here the
        source integral is done up front, so it has to be inside it (∫Wh dr = 1).

        Only g0 and g1 are numerical. Differentiating once, the g1 term cancels,
        W_phi' = -Wh/r' - g0/r'^2 + Wh/r' = -g0/r'^2, so every order n>=1 is Leibniz on
        -g0*r'^-2 with g0^(k) = -Wh^(k-1) -- the analytic window derivatives we already have.

        W_derivs_list : derivatives of n_angular*W/normW on x (get_all_derivatives).
        """
        # source density Wh (W_derivs_list already holds n_angular*W/normW, so only H/a is missing)
        n_ang = self.n_angular_spline if self.n_angular_spline is not None else (lambda r: 1.)
        Wh = lambda r: self.H_over_a_spline(r) * n_ang(r) * self._cached_expressions[(0, 0)](r)

        # computation of g0 and g1 interatively with a backward loop!
        g0, g1 = np.zeros_like(x), np.zeros_like(x)
        for i in range(len(x)-2, -1, -1):
            g0[i] = g0[i+1] + quad(Wh, x[i], x[i+1])[0]
            g1[i] = g1[i+1] + quad(lambda r: Wh(r)/r, x[i], x[i+1])[0]
        if abs(g0[0] - 1.) > 1e-3:
            print(f'  Warning: lensing source integral captures {g0[0]:.4f} of the window '
                  f'(expected 1): r grid [{x[0]:.0f}, {x[-1]:.0f}] does not bracket its support')

        # derivatives of Wh = (H/a) * W, by the product rule
        H_derivs = compute_spline_derivatives(self.H_over_a_spline, self.H_over_a_data[0], x,
                                              max_deriv=max_deriv-2, smooth_s=0)
        Wh_derivs = [product_deriv(i, H_derivs, W_derivs_list) for i in range(max_deriv-1)]

        # W_phi^(n) = d^(n-1)/dr^(n-1)[-g0/r^2], with d^m/dr^m[r^-2] = (-1)^m (m+1)!/r^(m+2)
        g0_derivs = [g0] + [-Wh_derivs[k-1] for k in range(1, max_deriv)]
        derivs = [g0/x - g1]
        for n in range(1, max_deriv+1):
            derivs.append(-sum(comb(n-1, k) * g0_derivs[k]
                               * (-1.)**(n-1-k) * factorial(n-k) / x**(n+1-k)
                               for k in range(n)))
        return derivs


def load_or_compute_window_derivatives(p, window_args, r_list, output_dir, max_deriv=11):
    """
    Load window derivatives from cache if available and valid, otherwise compute and save.

    Parameters:
    -----------
    window_args : tuple
        Window-type dependent (see WindowDerivatives.__init__):
          nbody  : (xmin, xmax, H_over_a_data, sigma_z)
          ska    : (xmin, xmax, H_over_a_data, sigma_z, n_angular_data)
          euclid : (bin_idx, H_over_a_data)
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
    W_lens_derivs_list : list of arrays
        Same for the lensing efficiency window (see get_lensing_derivatives)
    """
    window_cache_file = f'{output_dir}window_derivs_cache.npz'
    window_type = p.window_type

    def _build_cache_params(window_type, window_args, r_list, max_deriv):
        """Extract flat, comparable parameters from window_args for caching."""
        cp = {'window_type': window_type, 'n_r': len(r_list),
              'r_min': r_list[0], 'r_max': r_list[-1], 'max_deriv': max_deriv}
        if window_type == 'euclid':
            cp['bin_idx'] = window_args[0]
            cp['H_data']  = window_args[1]
        elif window_type == 'ska':
            cp['xmin'], cp['xmax'], cp['H_data'], cp['sigma_z'], cp['n_angular_data'] = window_args
        else:  # nbody
            cp['xmin'], cp['xmax'], cp['H_data'], cp['sigma_z'] = window_args
            cp['n_angular_data'] = None
        return cp

    def _params_match(cp, cached_cp):
        if cached_cp.get('window_type') != cp['window_type']:
            return False
        if not (cached_cp['n_r'] == cp['n_r'] and
                np.isclose(cached_cp['r_min'], cp['r_min']) and
                np.isclose(cached_cp['r_max'], cp['r_max']) and
                cached_cp['max_deriv'] == cp['max_deriv']):
            return False
        wt = cp['window_type']
        if wt == 'euclid':
            return cached_cp['bin_idx'] == cp['bin_idx']
        # ska / nbody: compare xmin, xmax, sigma_z, H_data
        if not (np.isclose(cached_cp['xmin'], cp['xmin']) and
                np.isclose(cached_cp['xmax'], cp['xmax']) and
                np.isclose(cached_cp['sigma_z'], cp['sigma_z'])):
            return False
        H, cH = cp['H_data'], cached_cp['H_data']
        if not (np.allclose(cH[0], H[0]) and np.allclose(cH[1], H[1])):
            return False
        if wt == 'ska':
            na, cna = cp['n_angular_data'], cached_cp.get('n_angular_data')
            if na is None and cna is None:
                return True
            if na is None or cna is None:
                return False
            return np.allclose(cna[0], na[0]) and np.allclose(cna[1], na[1])
        return True  # nbody

    load_from_cache = False
    if os.path.exists(window_cache_file):
        print('  Found cached window derivatives, checking validity...')
        try:
            cache = np.load(window_cache_file, allow_pickle=True)
            cached_cp = cache['params'].item()
            cp = _build_cache_params(window_type, window_args, r_list, max_deriv)
            if _params_match(cp, cached_cp) and 'lens_deriv_0' in cache:
                print('  Cache valid! Loading precomputed window derivatives...')
                W_derivs_list = [cache[f'deriv_{i}'] for i in range(max_deriv + 1)]
                W_lens_derivs_list = [cache[f'lens_deriv_{i}'] for i in range(max_deriv + 1)]
                load_from_cache = True
            else:
                print('  Cache invalid (parameters changed), will recompute...')
        except Exception as e:
            print(f'  Error loading cache: {e}, will recompute...')

    if not load_from_cache:
        print('  Computing window function derivatives...')
        W_derivs = WindowDerivatives(window_type, window_args)
        W_derivs_list = W_derivs.get_all_derivatives(r_list, r_power=0, max_deriv=max_deriv)

        print('  Computing lensing efficiency window derivatives...')
        W_lens_derivs_list = W_derivs.get_lensing_derivatives(r_list, W_derivs_list,
                                                             max_deriv=max_deriv)

        print('  Saving window derivatives to cache...')
        cp = _build_cache_params(window_type, window_args, r_list, max_deriv)
        save_dict = {'params': cp}
        for i, deriv in enumerate(W_derivs_list):
            save_dict[f'deriv_{i}'] = deriv
        for i, deriv in enumerate(W_lens_derivs_list):
            save_dict[f'lens_deriv_{i}'] = deriv
        np.savez(window_cache_file, **save_dict)
        print(f'  Cache saved to {window_cache_file}')

    return W_derivs_list, W_lens_derivs_list


# ============================================================================
# Coefficient Functions (α, β, γ)
# ============================================================================
def get_coefficients(p, time_dict, compute_c1_c2=''):
    """
    Compute alpha, beta, gamma coefficients for different orders.

    Parameters:
    -----------
    p : parameter object
    time_dict : dict with cosmological functions on 'ra' grid
    compute_c1_c2 : str (default '')
        If '': returns base coefficients (alpha, beta, gamma) from eq 39-41
        If 'c1': returns GR bias correction coefficients proportional to c1 from eq 47-49
        If 'c2': returns GR bias correction coefficients proportional to c2 from eq 50-52
        If 'b2': returns GR bias correction coefficients proportional to b2 (δ_1^2 term)
        If 'fnl': returns the delta_{2,fNL} scale-dependent bias coefficients (biases baked in)

    Returns:
    --------
    (alpha, beta, gamma) : tuple of dicts
        Each coefficient is a dict {order: array} where order in {0, 1, 2}
        Content depends on compute_c1_c2:
        - '': base coefficients from eq 39-41
        - 'c1': corrections proportional to c1 = 1 - b1 from eq 47-49
        - 'c2': corrections proportional to c2 = -db1/dr + 3*H*b2 from eq 50-52
        - 'b2': corrections proportional to b2 (δ_1^2 term)
        - 'fnl': delta_{2,fNL} term with b_phi, b_phidelta, b_phi2, b_n baked in
                 (local fNL, f~NL=0; requires b1 and b2 in time_dict['data'])
    """

    # Extract cosmological variables for readability
    f = time_dict['fa']
    Om = time_dict['Oma']
    w = time_dict['wa']
    H = time_dict['Ha']
    va = time_dict['va']
    D  = time_dict['Da']
    a  = time_dict['a']
    ra = time_dict['ra']

    zeros = np.zeros_like(f)

    if compute_c1_c2=='':
        # Base coefficients from equations 39-41
        if p.which == 'F2':
            alpha = {
                0: (7. - 3.*va) / 14.,
                1: 4.*f + 1.5*Om - 9.*w/7. if not p.Newton else zeros,
                2: 18.*f**2 + 9.*f**2*Om - 4.5*f*Om if not p.Newton else zeros
            }
            beta = {
                0: np.ones_like(f),
                1: -2.*f**2 + 6.*f - 4.5*Om if not p.Newton else zeros,
                2: 36.*f**2 + 18.*f**2*Om if not p.Newton else zeros
            }
            gamma = {
                0: zeros,
                1: 0.5*(-f**2 + f - 3.*Om) if not p.Newton else zeros,
                2: 0.25*(18.*f**2 + 9.*(f**2 - f)*Om) if not p.Newton else zeros
            }
        else:  # G2 or dv2
            alpha = {
                0: f - 3.*w/7.,
                1: -7.5*Om*f if not p.Newton else zeros,
                2: zeros
            }
            beta = {
                0: f,
                1: -12.*Om*f if not p.Newton else zeros,
                2: zeros
            }
            gamma = {
                0: zeros,
                1: -2.25*Om*f if not p.Newton else zeros,
                2: zeros
            }

    elif compute_c1_c2=='c1':
        # Corrections proportional to c1 from equations 47-49
        if p.which == 'F2' and not p.Newton:
            alpha = {
                0: zeros,
                1: 0.5*(1.5*Om + 4.*f - 9.*w/7.),
                2: 0.5*Om*f * (-9.*f + 4.5 - 12.*f/Om + 9./H - 6.*f/(H*Om))
            }
            beta = {
                0: zeros,
                1: -0.5*4.5*Om,
                2: -0.5*Om*f * (18.*f - 18. + 24.*f/Om - 18./H + 12.*f/(H*Om))
            }
            gamma = {
                0: zeros,
                1: 0.5*(-1.5*Om - f),
                2: -0.5*Om*f * (2.25*f + 3.*f/Om - 2.25/H + 1.5*f/(H*Om))
            }
        else:
            # No c1 corrections for G2/dv2 or Newton mode
            alpha = {0: zeros, 1: zeros, 2: zeros}
            beta = {0: zeros, 1: zeros, 2: zeros}
            gamma = {0: zeros, 1: zeros, 2: zeros}

    elif compute_c1_c2=='c2':
        # Corrections proportional to c2 from equations 50-52
        if p.which == 'F2' and not p.Newton:
            alpha = {
                0: zeros,
                1: zeros,
                2: -0.5*6.*f/(H*Om) 
            }
            beta = {
                0: zeros,
                1: -0.5*2.*f/H,
                2: -0.5*12.*f/(H*Om)
            }
            gamma = {
                0: zeros,
                1: -0.5*f/(2.*H),
                2: -0.5*3.*f/(2.*H*Om)
            }
        else:
            # No c2 corrections for G2/dv2 or Newton mode
            alpha = {0: zeros, 1: zeros, 2: zeros}
            beta = {0: zeros, 1: zeros, 2: zeros}
            gamma = {0: zeros, 1: zeros, 2: zeros}

    elif compute_c1_c2=='b2':
        # Corrections proportional to b2 (δ_1^2 term)
        if p.which == 'F2' and not p.Newton:
            alpha = {
                0: zeros,
                1: zeros,
                2: 18.*f**2
            }
            beta = {
                0: zeros,
                1: 6.*f,
                2: 36.*f**2
            }
            gamma = {
                0: zeros,
                1: 1.5*f,
                2: 4.5*f**2
            }
        else:
            # No b2 corrections for G2/dv2 or Newton mode
            alpha = {0: zeros, 1: zeros, 2: zeros}
            beta = {0: zeros, 1: zeros, 2: zeros}
            gamma = {0: zeros, 1: zeros, 2: zeros}

    elif compute_c1_c2=='fnl':
        # Second-order scale-dependent bias, local fNL (f~NL=0), 1/2 convention throughout
        # (delta = delta_1 + delta_2 here vs delta_1 + delta_2/2 in the derivation). Dispatch
        # on which: F2 -> density delta_{2,fNL}; G2/dv2 -> velocity v_{2,fNL}.
        if p.which == 'F2' and not p.Newton and 'data' in time_dict and 'b2' in time_dict['data']:
            # --- density delta_{2,fNL}: biases baked in (unlike c1/c2/b2 which factor out a data field) ---
            g      = D/a
            g_in   = g * 3./5.*(1. + 2./3.*f/Om)
            deltac = 1.686
            fnl    = p.fnl_local

            # Lagrangian biases from the Eulerian b1, b2, splined onto the ra grid so they
            # align with the cosmological arrays (data['r'] == ra for euclid, differs for ska): b^L = b - 1
            b1L = UnivariateSpline(time_dict['data']['r'], time_dict['data']['b1']-1., k=5, s=0)(ra)
            b2L = UnivariateSpline(time_dict['data']['r'], time_dict['data']['b2']-1., k=5, s=0)(ra)

            b_phi    = 2.*fnl*g_in*deltac*b1L
            b_phidel = 2.*fnl*g_in*(-b1L + deltac*b2L)
            b_phi2   = 2.*fnl**2*g_in**2*deltac*(-2.*b1L + deltac*b2L)
            bn_ND    = fnl*g_in*b1L                       # b_n / (N D): the N*D cancels

            # b_phi^L' / H : conformal-time derivative of the Lagrangian b_phi,
            # same convention as c2's -db1/dr (prime = d/dtau = -d/dr on the lightcone)
            b_phiL_prime = -UnivariateSpline(ra, b_phi, k=5, s=0).derivative(1)(ra)
            Dphi = b_phiL_prime / H                       # = b_phi^L' / H

            pref = 3.*Om/g   # common 3 Omega_m / g factor

            # index 1 -> H^2/k^2 bracket, index 2 -> H^4/k^4 bracket; overall factor 1/2
            alpha = {
                0: zeros,
                1: 0.5*(-pref)*(2.*bn_ND - 4.*g_in*fnl + 2.*b_phi),
                2: 0.5*( pref)*(2.*f*(2.*Dphi - 6.*b_phi) + 6.*(Om/g)*b_phi2 + 12.*f*g_in*fnl)
            }
            beta = {
                0: zeros,
                1: 0.5*(-pref)*(4.*b_phi + 2.*b_phidel - 8.*g_in*fnl + 2.*bn_ND),
                2: 0.5*( pref)*(4.*f*(2.*Dphi - 6.*b_phi) + 12.*(Om/g)*b_phi2 + 24.*f*g_in*fnl)
            }
            gamma = {
                0: zeros,
                1: 0.5*(-pref/2.)*(b_phi + b_phidel - 2.*g_in*fnl),
                2: 0.5*( pref/2.)*(f*(2.*Dphi - 6.*b_phi) + 3.*(Om/g)*b_phi2 + 6.*f*g_in*fnl)
            }
        elif p.which in ['G2', 'dv2'] and not p.Newton:
            # --- velocity v_{2,fNL}: unbiased, pure cosmology x fNL, only H^2/k^2 (index 1) ---
            g    = D/a
            g_in = g * 3./5.*(1. + 2./3.*f/Om)
            fnl  = p.fnl_local
            alpha = {0: zeros, 1: 0.5*(-6.*Om/g)*f*g_in*(2.*fnl), 2: zeros}
            beta  = {0: zeros, 1: 0.5*(-6.*Om/g)*f*g_in*(4.*fnl), 2: zeros}
            gamma = {0: zeros, 1: 0.5*(-3.*Om/g)*f*g_in*(   fnl), 2: zeros}
        else:
            # Newton mode, missing bias data (F2), or any other which -> no fNL correction
            alpha = {0: zeros, 1: zeros, 2: zeros}
            beta = {0: zeros, 1: zeros, 2: zeros}
            gamma = {0: zeros, 1: zeros, 2: zeros}

    else:
        raise ValueError(f"Invalid compute_c1_c2 value: '{compute_c1_c2}'. Must be '', 'c1', 'c2', 'b2', or 'fnl'.")

    return alpha, beta, gamma

# ============================================================================
# Coefficients f_nm - Unified computation following eq. B10
# ============================================================================

def compute_f_nm_unified(p, alpha, beta, gamma, time_dict, h_power, use_b1=False):
    """
    Unified computation of f^(m)_nm multipoles for any m following eq. B10.

    Can compute f^(-4), f^(-2), f^(0), f^(2), f^(4) using the same pattern.

    Pattern from eq. B10:
    - f^(m)_{0,0}   = (β - α)/2 - 2γ   [with appropriate coeff index]
    - f^(m)_{2,-2}  = (α - β)/4 + γ     [only for h_power=0]
    - f^(m)_{0,-2}  = 1/2 * (β/2 - α)  [uses next coeff index]
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
    use_b1 : bool
        If True, return just the prefactor spline (for b_s computation)

    Returns:
    --------
    list of UnivariateSplines (if use_b1=False) or single UnivariateSpline (if use_b1=True)
    """

    # Apply prefactor: D² * H/a
    if p.which == 'F2':
        prefactor = time_dict['Da']**2 * time_dict['Ha'] / time_dict['a']
    elif p.which == 'G2':
        prefactor = -time_dict['Da']**2 * time_dict['Ha'] / time_dict['a']
    else: # dv2
        prefactor = time_dict['Da']**2 * time_dict['Ha'] / time_dict['a'] * time_dict['Ha'] * time_dict['mathcalR']

    if use_b1:
        # For b_s terms: just return the prefactor as a spline
        # The b_s coefficients (1/6, 1/4, -1/2, etc.) will be applied in get_bispectrum_kernels_analytical
        return UnivariateSpline(time_dict['ra'], prefactor, k=5, s=0)

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
        W_derivs = WindowDerivatives(p.window_type, window_args)
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

            if COMPUTE_FNL and p.fnl_local != 0 and not p.Newton and p.which in ['G2', 'dv2']:
                print(f'         Adding v_{{2,fNL}} to {p.which} f0/fm2 multipoles')
                a_v, b_v, g_v = get_coefficients(p, time_dict, compute_c1_c2='fnl')
                for key in (0, 1, 2):
                    alpha_coeff[key] = alpha_coeff[key] + a_v[key]
                    beta_coeff[key]  = beta_coeff[key]  + b_v[key]
                    gamma_coeff[key] = gamma_coeff[key] + g_v[key]

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
        y_list[output_key] = np.zeros((3, len(qterm_list), len(ell_list), len(r_list)), dtype=np.float64)

        # Compute fctr derivatives and apply operators
        # derive_start is already set: F2=0, dv2=1, G2=2
        max_deriv = 4 + derive_start

        # For F2: compute b1 derivatives if available
        use_b1 = False
        use_c1_c2 = False
        if COMPUTE_B1 and p.which=='F2' and 'data' in time_dict and 'b1' in time_dict['data']:
            print('         Adding linear bias b1 to F2 terms')
            b1_spline = UnivariateSpline(time_dict['data']['r'], time_dict['data']['b1'], k=5, s=0)
            b1_derivs= compute_spline_derivatives(b1_spline, time_dict['data']['r'], r_list,
                                                        max_deriv=max_deriv+derive_start, smooth_s=1e-6)
            use_b1 = True

            # Compute c1 GR bias corrections (c2 corrections are all zero for f^(-2) and f^(-4)).
            # NOT for radiation ('not p.rad'): c1/c2 must vanish for the radiation term -- that
            # velocity coupling already lives in the non-rad density, so applying it here too
            # double-counts. Radiation is therefore just b1 * delta^(2)_mP,rad.
            if COMPUTE_C1_C2 and not p.Newton and not p.rad and 'b2' in time_dict['data']:
                print('         Adding GR bias corrections (c1 terms) to F2 terms (all c2 coeffs vanishe!)')

                # Get c1 correction coefficients
                alpha_c1_simple, beta_c1_simple, gamma_c1_simple = get_coefficients(p, time_dict, compute_c1_c2='c1')

                # Compute correction f_nm splines for c1 (c2 is zero for f^(-2) and f^(-4))
                fctr_list_fm2_c1 = compute_f_nm_unified(p, alpha_c1_simple, beta_c1_simple, gamma_c1_simple, time_dict, h_power=-2)
                fctr_list_fm4_c1 = compute_f_nm_unified(p, alpha_c1_simple, beta_c1_simple, gamma_c1_simple, time_dict, h_power=-4)
                fctr_list_c1 = fctr_list_fm2_c1 + fctr_list_fm4_c1

                use_c1_c2 = True

            # --- DISABLED: radiation density velocity term 3(b1-1) H T_2^rad = 3(1-b1) H v_2^rad ---
            # Wrong approach; kept for reference only. It was built as 3 * H^2 * (G2 radiation
            # multipoles) and fed through the same b1*base + (1-b1)*fctr_c1 combination below.
            # if p.rad:
            #     p_g2 = copy.copy(p); p_g2.which = 'G2'
            #     rad_v2 = compute_radiation_f_nm(p_g2, time_dict)   # [fm2R, fm4R] velocity splines
            #     _ra  = time_dict['ra']
            #     _fac = 3.*time_dict['Ha']**2
            #     fctr_list_c1 = [UnivariateSpline(_ra, _fac*s(_ra), k=5, s=0) for s in rad_v2]
            #     print('         Adding radiation density velocity term 3(1-b1) H v_2^rad to F2')
            #     use_c1_c2 = True
        else:
            b1_derivs = None

        for qt_ind, qt in enumerate(qterm_list):
            fctr = fctr_list[qt]

            # Compute fctr and its derivatives using the new function
            fctr_derivs = compute_spline_derivatives(fctr, time_dict['ra'], r_list,
                                                          max_deriv=max_deriv, smooth_s=0)
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
            fctr_W_derivs = [product_deriv(i+derive_start, fctr_derivs, W_derivs_list_local) for i in range(5)]

            # If b1 is present, apply second product rule: b1 * (fctr * W)
            if use_b1:
                # Always compute b1 * base * W
                f, df, d2f, d3f, d4f = [product_deriv(i, b1_derivs, fctr_W_derivs) for i in range(5)]

                # Add GR corrections: b1*base + c1*f_c1 (c2 is zero for f^(-2) and f^(-4))
                # Since c1 = 1-b1: b1*base + (1-b1)*f_c1 = b1*base - b1*f_c1 + f_c1
                if use_c1_c2:
                    # Compute c1 correction derivatives
                    fctr_derivs_c1 = compute_spline_derivatives(fctr_list_c1[qt], time_dict['ra'], r_list,
                                                               max_deriv=max_deriv, smooth_s=0)

                    # Compute fctr_c1 * W
                    fctr_W_derivs_c1 = [product_deriv(i+derive_start, fctr_derivs_c1, W_derivs_list_local) for i in range(5)]

                    # b1 * f_c1 * W
                    b1_c1 = [product_deriv(i, b1_derivs, fctr_W_derivs_c1) for i in range(5)]

                    # Apply c1 corrections: b1*base - b1*f_c1 + f_c1
                    f = f - b1_c1[0] + fctr_W_derivs_c1[0]
                    df = df - b1_c1[1] + fctr_W_derivs_c1[1]
                    d2f = d2f - b1_c1[2] + fctr_W_derivs_c1[2]
                    d3f = d3f - b1_c1[3] + fctr_W_derivs_c1[3]
                    d4f = d4f - b1_c1[4] + fctr_W_derivs_c1[4]
            else:
                f, df, d2f , d3f, d4f = fctr_W_derivs

            # Store both levels and apply mathcalD operator
            for ind_ell, ell in enumerate(ell_list):
                alpha = ell*(ell+1) - 2.
                y_list[output_key][0, qt_ind, ind_ell] = f
                y_list[output_key][1, qt_ind, ind_ell] = -d2f + 2./r_list*df + alpha/r_list**2*f
                y_list[output_key][2, qt_ind, ind_ell] = (d4f - 4./r_list*d3f 
                                        +(8./r_list**2 - 2.*alpha/r_list**2)*d2f 
                                        +(-8./r_list**3 + 8.*alpha/r_list**3)*df + 
                                         (alpha**2/r_list**4 - 10.*alpha/r_list**4)*f)
 

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
            elif lterm == 'pot_fnl': # scale dependent bias
                gin = time_dict['Da']/time_dict['a'] * 3./5. * (1+2./3.*time_dict['fa']/time_dict['Oma'])
                fctr = UnivariateSpline(time_dict['ra'], \
                            -time_dict['Ha']/time_dict['a']*time_dict['Da']*2.*1.686*gin/time_dict['a'], k=5, s=0)
                derive_start = 0
            else:
                print('no code for {}'.format(lterm))
                raise ValueError(f"Invalid 'lterm' parameter: {lterm}")
            
            # Compute fctr derivatives using the new function
            # Use time_dict['ra'] as the full data grid for accuracy
            fctr_derivs = compute_spline_derivatives(fctr, time_dict['ra'], r_list, max_deriv=11, smooth_s=0)

            # For density term: compute b1 derivatives if available
            if COMPUTE_B1 and lterm in ['density', 'pot_fnl'] and 'data' in time_dict and 'b1' in time_dict['data']:
                print('Adding linear bias b1 to linear terms')
                if lterm == 'density':
                    b1_spline = UnivariateSpline(time_dict['data']['r'], time_dict['data']['b1'], k=5, s=0)
                else:
                    b1_spline = UnivariateSpline(time_dict['data']['r'], time_dict['data']['b1']-1, k=5, s=0)

                b1_derivs = compute_spline_derivatives(b1_spline, time_dict['data']['r'], r_list,
                                                            max_deriv=11, smooth_s=1e-6)
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

        # Second-order velocity fNL: fold v_{2,fNL} into the G2/dv2 base coefficients.
        # Additive and unbiased; only H^2/k^2, which (given beta1=2*alpha1, gamma1=alpha1/4)
        # lands purely in A2 via f_nm - A0/A4 and the fm2/fm4 path are left unchanged.
        if COMPUTE_FNL and p.fnl_local != 0 and not p.Newton and p.which in ['G2', 'dv2']:
            print(f'     Adding v_{{2,fNL}} to {p.which} kernel')
            a_v, b_v, g_v = get_coefficients(p, time_dict, compute_c1_c2='fnl')
            for key in (0, 1, 2):
                alpha_coeff[key] = alpha_coeff[key] + a_v[key]
                beta_coeff[key]  = beta_coeff[key]  + b_v[key]
                gamma_coeff[key] = gamma_coeff[key] + g_v[key]

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


        # ====================================================================
        # Precompute b1 and b_s derivatives if available (for F2 only)
        # ====================================================================
        use_b1 = False
        use_c1_c2 = False
        use_fnl = False
        bs_terms_all = None  # Will be (n_ell, 3, n_r) if computed

        if COMPUTE_B1 and p.which == 'F2' and 'data' in time_dict and 'b1' in time_dict['data']:
            # Compute b1 derivatives (for f-coefficient product rule)
            b1_spline = UnivariateSpline(time_dict['data']['r'], time_dict['data']['b1'], k=5, s=0)
            b1_derivs = compute_spline_derivatives(b1_spline, time_dict['data']['r'], r_list,
                                                   max_deriv=max_deriv_total, smooth_s=1e-6)

            # Compute c1, c2, and b2 for GR bias corrections (only for non-Newton F2)
            if COMPUTE_C1_C2 and not p.Newton and 'b2' in time_dict['data']:
                print('     Adding GR bias corrections (c1, c2, and b2 terms)')

                # c2 = -db1/dr + 3*H*b2
                # Compute db1/dr on original grid (already in b1_derivs[1])
                # Interpolate H to b1 grid
                H_spline = UnivariateSpline(time_dict['ra'], time_dict['Ha'], k=5, s=0)
                H_at_b1_grid = H_spline(time_dict['data']['r'])

                # c2 on original grid
                c2_data = -b1_spline.derivative(1)(time_dict['data']['r']) + 3.*H_at_b1_grid*time_dict['data']['b2']

                # Create spline and compute derivatives
                c2_spline = UnivariateSpline(time_dict['data']['r'], c2_data, k=5, s=0)
                c2_derivs = compute_spline_derivatives(c2_spline, time_dict['data']['r'], r_list,
                                                      max_deriv=max_deriv_total, smooth_s=1e-6)

                # Get GR correction coefficients
                alpha_c1, beta_c1, gamma_c1 = get_coefficients(p, time_dict, compute_c1_c2='c1')
                alpha_c2, beta_c2, gamma_c2 = get_coefficients(p, time_dict, compute_c1_c2='c2')

                # Compute f_nm for c1 corrections (only f0 and f2, no f4 since it has no relativistic parts)
                f0_splines_c1 = compute_f_nm_unified(p, alpha_c1, beta_c1, gamma_c1, time_dict, h_power=0)
                f0_values_c1 = np.array([f0_spline(r_list) for f0_spline in f0_splines_c1])  # Only 2 components (H², H⁴)

                f2_splines_c1 = compute_f_nm_unified(p, alpha_c1, beta_c1, gamma_c1, time_dict, h_power=2)

                # Compute f_nm for c2 corrections
                f0_splines_c2 = compute_f_nm_unified(p, alpha_c2, beta_c2, gamma_c2, time_dict, h_power=0)
                f0_values_c2 = np.array([f0_spline(r_list) for f0_spline in f0_splines_c2])

                # Compute b2 derivatives for b2 GR bias corrections
                b2_spline = UnivariateSpline(time_dict['data']['r'], time_dict['data']['b2'], k=5, s=0)
                b2_derivs = compute_spline_derivatives(b2_spline, time_dict['data']['r'], r_list,
                                                      max_deriv=max_deriv_total, smooth_s=1e-6)

                # Get b2 correction coefficients
                alpha_b2, beta_b2, gamma_b2 = get_coefficients(p, time_dict, compute_c1_c2='b2')

                # Compute f_nm for b2 corrections (only f0, no f2 since all f^(2)_b2 components are zero)
                f0_splines_b2 = compute_f_nm_unified(p, alpha_b2, beta_b2, gamma_b2, time_dict, h_power=0)
                f0_values_b2 = np.array([f0_spline(r_list) for f0_spline in f0_splines_b2])

                use_c1_c2 = True
            else:
                use_c1_c2 = False
                if not p.Newton:
                    print('     Warning: b1 found but b2 not found, skipping GR bias corrections')

            # Second-order scale-dependent bias delta_{2,fNL}: independent of the c1/c2/b2
            # GR corrections (its own flag), needs only b2 (for b2^L) and full GR (non-Newton).
            # Produces f0 (2 components: H^2, H^4) and f2 (1 component: H^2); no f4 (index 0 is zero).
            if COMPUTE_FNL and p.fnl_local != 0 and not p.Newton and 'b2' in time_dict['data']:
                print('     Adding delta_{2,fNL} scale-dependent bias term')
                alpha_fnl, beta_fnl, gamma_fnl = get_coefficients(p, time_dict, compute_c1_c2='fnl')
                f0_splines_fnl = compute_f_nm_unified(p, alpha_fnl, beta_fnl, gamma_fnl, time_dict, h_power=0)
                f0_values_fnl = np.array([f0_spline(r_list) for f0_spline in f0_splines_fnl])
                f2_splines_fnl = compute_f_nm_unified(p, alpha_fnl, beta_fnl, gamma_fnl, time_dict, h_power=2)
                use_fnl = True

            use_b1 = True
            if COMPUTE_BS:
                print('     Adding linear bias b1 and b_s = -2/7*(b1-1) to F2 kernels')
                # d^n/dr^n[b_s] = -2/7 * d^n/dr^n[b1] for n≥1
                # For n=0: b_s = -2/7 * (b1 - 1)
                bs_derivs = -2./7. * b1_derivs.copy()
                bs_derivs[0] = -2./7. * (b1_derivs[0] - 1.)
            else:
                print('     Adding linear bias b1 to F2 kernels (b_s disabled for testing)')
                bs_derivs = None

            # Compute b_s terms only if enabled
            if bs_derivs is not None:
                # Get prefactor spline (D² * H/a) for b_s terms
                prefactor_spline = compute_f_nm_unified(p, alpha_coeff, beta_coeff, gamma_coeff,
                                                        time_dict, h_power=0, use_b1=True)

                # Compute prefactor derivatives
                prefactor_derivs = compute_spline_derivatives(prefactor_spline, time_dict['ra'], r_list,
                                                             max_deriv=max_deriv_total, smooth_s=0)

                # Compute product: fctr_bs = prefactor * b_s
                fctr_bs_derivs = [product_deriv(j, prefactor_derivs, bs_derivs) for j in range(5)]

                # Compute product: fctr_bs * W
                fctr_bs_W_derivs = [product_deriv(j, fctr_bs_derivs, W_derivs_list) for j in range(5)]

                # Now we have f_bs = prefactor * b_s * W and its derivatives
                # Initialize bs_terms_all: (n_ell, 3, n_r)
                # [0]: f_bs (for A0 terms)
                # [1]: D[f_bs] (for A2 terms)
                # [2]: D²[f_bs] (for A4 terms)
                bs_terms_all = np.zeros((n_ell, 3, n_r))

                f_bs, df_bs, d2f_bs, d3f_bs, d4f_bs = fctr_bs_W_derivs

                # Loop over ells to apply ell-dependent operators
                for ell_idx, ell in enumerate(ell_list):
                    alpha_ell = ell*(ell+1) - 2.

                    # Level 0: f_bs (no operator)
                    bs_terms_all[ell_idx, 0, :] = f_bs

                    # Level 1: D[f_bs] = -d²f_bs + 2/r * df_bs + α/r² * f_bs
                    # This is the "inner operator" for F2
                    y_bs = -d2f_bs + 2./r_list*df_bs + alpha_ell/r_list**2*f_bs
                    bs_terms_all[ell_idx, 1, :] = y_bs

                    # Level 2: D²[f_bs] = D[D[f_bs]] for A4
                    # Compute derivatives of y_bs = D[f_bs]
                    dy_bs = -d3f_bs + 2./r_list*d2f_bs + (alpha_ell-2.)/r_list**2*df_bs - 2.*alpha_ell/r_list**3*f_bs
                    d2y_bs = -d4f_bs + 2./r_list*d3f_bs + (alpha_ell-4.)/r_list**2*d2f_bs - 4.*(alpha_ell-1.)/r_list**3*df_bs + 6.*alpha_ell/r_list**4*f_bs

                    # Apply D operator again (outer operator)
                    bs_terms_all[ell_idx, 2, :] = -d2y_bs + 2./r_list*dy_bs + alpha_ell/r_list**2*y_bs
        else:
            if  p.which == 'F2': print('     Assuming b1=1, b_s=0')
            b1_derivs = None

        # Determine number of components for each kernel
        # A0: for F2 can have up to 4 components, for G2/dv2 typically 1-2
        f0_splines = compute_f_nm_unified(p, alpha_coeff, beta_coeff, gamma_coeff, time_dict, h_power=0)

        # Get f^(2) and f^(4) splines (independent of ell)
        f2_splines = compute_f_nm_unified(p, alpha_coeff, beta_coeff, gamma_coeff, time_dict, h_power=2)
        [f4_spline] = compute_f_nm_unified(p, alpha_coeff, beta_coeff, gamma_coeff, time_dict, h_power=4)

        # ====================================================================
        # Precompute all spline derivatives ONCE (independent of ell!)
        # ====================================================================
        # A0: f^(0) evaluated at r_list
        f0_values = np.array([f0_spline(r_list) for f0_spline in f0_splines])  # (n_A0_comp, n_r)

        # If b1 present, multiply by b1 and add GR corrections
        if use_b1:
            # Newtonian (H^0): only b1, no GR corrections
            # Relativistic components: b1*base + c1*f_c1 + c2*f_c2 + b2*f_b2
            # Since c1 = 1-b1: b1*base + (1-b1)*f_c1 + c2*f_c2 + b2*f_b2 = b1*(base - f_c1) + f_c1 + c2*f_c2 + b2*f_b2
            if use_c1_c2:
                f0_values[0] = b1_derivs[0] * f0_values[0]
                f0_values[1:] = b1_derivs[0] * (f0_values[1:] - f0_values_c1) + f0_values_c1 + c2_derivs[0] * f0_values_c2 + b2_derivs[0] * f0_values_b2

            else:
                # No GR corrections, just multiply relativistic parts by b1
                f0_values = b1_derivs[0] * f0_values

            # delta_{2,fNL} is an additive second-order term (biases already baked in, not
            # scaled by b1) and is independent of the c1/c2/b2 GR corrections above
            if use_fnl:
                f0_values[1:] = f0_values[1:] + f0_values_fnl

        # A2: f^(2) and its derivatives
        f2_derivs_list = []  # List of (max_deriv_inner+1, n_r) arrays
        for f2_spline in f2_splines:
            fctr_derivs= compute_spline_derivatives(f2_spline, time_dict['ra'], r_list,
                                                          max_deriv=max_deriv_inner, smooth_s=0)
            f2_derivs_list.append(fctr_derivs)

        # Compute f2 derivatives for c1 corrections (c2 and b2 have no f^(2) corrections)
        if use_c1_c2:
            f2_derivs_list_c1 = []
            for f2_spline in f2_splines_c1:
                fctr_derivs = compute_spline_derivatives(f2_spline, time_dict['ra'], r_list,
                                                        max_deriv=max_deriv_inner, smooth_s=0)
                f2_derivs_list_c1.append(fctr_derivs)

        # Compute f2 derivatives for the delta_{2,fNL} correction (1 component: H^2)
        if use_fnl:
            f2_derivs_list_fnl = []
            for f2_spline in f2_splines_fnl:
                fctr_derivs = compute_spline_derivatives(f2_spline, time_dict['ra'], r_list,
                                                        max_deriv=max_deriv_inner, smooth_s=0)
                f2_derivs_list_fnl.append(fctr_derivs)

        # A4: f^(4) and its derivatives
        f4_derivs= compute_spline_derivatives(f4_spline, time_dict['ra'], r_list,
                                                    max_deriv=max_deriv_total, smooth_s=0)

        n_A0_comp = len(f0_splines)
        # Initialize output arrays: (n_ell, n_components, n_r)
        A0_all = np.zeros((n_ell, n_A0_comp, n_r))
        A2_all = np.zeros((n_ell, 2, n_r))  # Always 2 components for A2
        A4_all = np.zeros((n_ell, 1, n_r))  # Always 1 component for A4

        # Precompute products of fctr*W for A2
        f2_products_list = []  # List of product derivatives for each f2
        for idx, fctr_derivs in enumerate(f2_derivs_list):
            if p.which == 'F2':
                # Compute fctr * W
                fctr_W = [product_deriv(j, fctr_derivs, W_derivs_list) for j in range(3)]

                if use_b1:
                    # Always compute b1 * base * W
                    f, df, d2f = [product_deriv(j, b1_derivs, fctr_W) for j in range(3)]

                    # Add GR corrections only to H^2 component (idx=1)
                    if idx == 1 and use_c1_c2:
                        # Relativistic H^2: b1*base + c1*f_c1 (no c2 or b2 since alpha_c2[1]=0 and all f^(2)_b2=0)
                        # Since c1 = 1-b1: b1*base + (1-b1)*f_c1 = b1*base - b1*f_c1 + f_c1
                        fctr_W_c1 = [product_deriv(j, f2_derivs_list_c1[0], W_derivs_list) for j in range(3)]

                        # b1 * f_c1 * W
                        b1_c1 = [product_deriv(j, b1_derivs, fctr_W_c1) for j in range(3)]

                        # Add corrections: b1*base - b1*f_c1 + f_c1
                        f = f - b1_c1[0] + fctr_W_c1[0]
                        df = df - b1_c1[1] + fctr_W_c1[1]
                        d2f = d2f - b1_c1[2] + fctr_W_c1[2]

                    # delta_{2,fNL}: additive H^2 correction (biases baked in, not scaled by b1),
                    # independent of the c1 GR correction above
                    if idx == 1 and use_fnl:
                        fctr_W_fnl = [product_deriv(j, f2_derivs_list_fnl[0], W_derivs_list) for j in range(3)]
                        f = f + fctr_W_fnl[0]
                        df = df + fctr_W_fnl[1]
                        d2f = d2f + fctr_W_fnl[2]
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
        # f^(4) has only ONE component which is Newtonian (H^0), so multiply entire thing
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

        result = {'r_list': r_list, 'ell_list': ell_list, 'A0': A0_all, 'A2': A2_all, 'A4': A4_all}

        # Add bs_terms if computed
        if bs_terms_all is not None:
            result['bs_terms'] = bs_terms_all

        return result

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
