import numpy as np
from numba import njit
import sympy as sp
from math import comb
import os
from mathematica import *
from lincosmo import *
from param_used import *
from scipy.interpolate import UnivariateSpline


class FFTLogProcessor:
    """
    A class to handle FFTLog processing with multiple qterm values efficiently
    """
    
    def __init__(self, k, fctk, p):
        self.k = k
        self.fctk = fctk
        self.Nk = len(k)
        self.kmin, self.kmax = np.min(k), np.max(k)
        self.which = p.which
        self.lterm = p.lterm
        self.qterm = p.qterm
        self.rad = p.rad
        
        # Pre-compute common quantities
        self.l = np.arange(self.Nk)
        self.p_list = np.arange(-self.Nk//2, self.Nk//2+1)
        self.eta_p_list = 2.*np.pi*self.p_list/np.log(self.kmax/self.kmin)

    def get_qterm_list(self):
        """Get the list of qterm values based on 'which' parameter"""
        qterm_map = {
            'd2v': [1, 2, 3],
            'd1v': [1, 2],
            'd1d': [1, 2], 
            'd0d': [1],
            'd3v': [1, 2, 3, 4]
        }
        return qterm_map.get(self.which, [0])
    
    def compute_quadratic_terms(self, k, fctk):
        """
        Calculate quadratic terms
        
        Parameters:
        -----------
        k : array_like
            Wave number array
        fctk : array_like
            Input array fctk
        lterm : str
            Linear term type ('density' or other)
        which : str
            Derivative type ('d2v', 'd1v', 'd1d', 'd3v', 'd0d')
        
        Returns:
        --------
        array_like
            Computed quadratic terms
        """
        
        fctr_scaled =fctk*k**4
        if self.which == 'd2v':
            if self.lterm == 'density':
                fctr_scaled *= k**2
    
            spline = UnivariateSpline(k, fctr_scaled, k=5, s=0)
            return [spline.derivative(2)(k),\
                    -2./k * spline.derivative(1)(k),\
                    fctk/k**2]
    
        elif self.which in ['d1v', 'd1d']:
            if self.lterm == 'density':
                fctr_scaled *= k
            else:
                fctr_scaled /= k
            
            if self.which == 'd1d':
                fctr_scaled *= k**2
    
            spline = UnivariateSpline(k, fctr_scaled, k=5, s=0)
            return [spline.derivative(1)(k), -fctk/k]
    
        elif self.which == 'd3v':
            if self.lterm == 'density':
                fctr_scaled *= k**3
            else:
                fctr_scaled *= k
    
            spline = UnivariateSpline(k, fctr_scaled, k=5, s=0)
            return [spline.derivative(3)(k),
                    -3./k * spline.derivative(2)(k),
                    3./k**2 * spline.derivative(1)(k),
                    -fctk / k**3]
    
        elif self.which == 'd0d':
            if self.lterm != 'density':
                fctr_scaled /= k**2
            return [fctr_scaled * k**2]
        
        else:
            raise ValueError(f"Invalid 'which' parameter: {self.which}")
    

    def set_bias(self, fctk_list):
        """Compute bias for a list of functions"""
        b_list = np.zeros(len(fctk_list), dtype=np.float64)
        
        for ind_b, fctk in enumerate(fctk_list):
            ind = 0
            for i in range(5, 16):
                b_list[ind_b] += (np.log(np.abs(fctk[-i])) - np.log(np.abs(fctk[i]))) / \
                                 (np.log(self.k[-i]) - np.log(self.k[i]))
                ind += 1
            b_list[ind_b] /= ind
            
        return b_list
    
    def get_cp_eta_p(self, fctk_list, b_list):
        """Compute cp for given functions and biases"""
        res = np.zeros((len(fctk_list), self.Nk+1), dtype=np.complex128)
        
        for ind_fct, fct_k in enumerate(fctk_list):
            b = b_list[ind_fct]
            for p in range(-self.Nk//2, self.Nk//2+1):
                res[ind_fct, p+self.Nk//2] = np.sum(
                    fct_k * self.k**(-b) * self.kmin**(-1j*self.eta_p_list[p+self.Nk//2]) * 
                    np.exp(-2.*1j*np.pi*p*self.l/self.Nk)
                ) / self.Nk
                
        return res
    
    def process_all_qterms(self):
        """
        Process all qterm values for given parameters
        
        Returns:
        --------
        dict: Results organized by qterm containing cp, eta_p, and b
        """


        out_dict = {'eta_p': self.eta_p_list,
                    'k': self.k,
                    'qterm_list': self.get_qterm_list()}

        if self.which in ['FG2', 'F2', 'G2', 'dv2', 'local']:
            # Handle special cases
            if not self.rad:
                fctk_list = [self.fctk * self.k**4]
            else:
                fctk_list = [self.fctk]
                
            b = self.set_bias(fctk_list)
            cp = self.get_cp_eta_p(fctk_list, b)
            
            out_dict[0] = {'cp': cp[0],  'b': b[0], 'fctk': fctk_list[0]}
            np.save(f'{output_dir}/fct_k_{self.which}_lterm{self.lterm}', out_dict)
            return out_dict
        
        else:
            # Handle quadratic terms with multiple qterms
            fctk_list = self.compute_quadratic_terms(self.k, self.fctk)
            
            b = self.set_bias(fctk_list)
            print(f'qterm {self.qterm} biases: {b}')
            cp = self.get_cp_eta_p(fctk_list, b)
            for fctk_ind, fctk in enumerate(fctk_list):
                out_dict[fctk_ind+1] = {'cp': cp[fctk_ind], 'b': b[fctk_ind], 'fctk': fctk }
                
                np.save(f'{output_dir}/fct_k_{self.which}_lterm{self.lterm}', out_dict)
            
            return out_dict

# Usage example:
def apply_fftlog(k, fctk, p):
    """
    Generalized FFTLog application that handles multiple qterm values
    """
    processor = FFTLogProcessor(k, fctk, p)
    results = processor.process_all_qterms()
    
    return results






# Optimized version that caches the SymPy derivatives
class WindowDerivatives:
    """
    Cache SymPy derivatives for W(r) and r^n * W(r) expressions for efficiency
    """
    def __init__(self, window_args, max_r_power=3, max_derivative=9):
        self.xmin, self.xmax, self.normW, self.bb = window_args
        self.max_r_power = max_r_power
        self.max_derivative = max_derivative
        self._cached_expressions = {}
        self._compile_derivatives()
    
    def _compile_derivatives(self):
        """Pre-compile SymPy expressions for faster evaluation"""
        sp_x = sp.symbols('x')
        sp_xmin = sp.symbols('xmin')
        sp_xmax = sp.symbols('xmax')
        sp_normW = sp.symbols('normW')
        sp_bb = sp.symbols('bb')
        
        # fctrase window expression
        base_W = (0.5+0.5*sp.tanh((sp_x-sp_xmin)/sp_bb)) *\
                 (0.5-0.5*sp.tanh((sp_x-sp_xmax)/sp_bb)) / sp_normW
        
        # For each r power and derivative order
        for r_power in range(self.max_r_power + 1):  # 0 to max_r_power
            for deriv_order in range(self.max_derivative + 1):  # 0 to max_derivative
                
                expr = sp.diff(sp_x**r_power * base_W, sp_x, deriv_order)
                
                # Substitute constants
                expr = expr.subs({sp_xmin: self.xmin, 
                                 sp_xmax: self.xmax, 
                                 sp_normW: self.normW,
                                 sp_bb: self.bb})
                
                # Lambdify for fast numerical evaluation
                self._cached_expressions[(r_power, deriv_order)] = sp.lambdify(sp_x, expr, 'numpy')
    
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
        
        Returns list [f, f', f'', f''', ...] evaluated at x
        """
        if max_deriv is None:
            max_deriv = self.max_derivative
            
        return [self(x, r_power=r_power, num_derivative=i) for i in range(max_deriv + 1)]



def fct_of_r_analytical(p, ell_list, r_list, time_dict, window_args, ):
    """
    Optimized version using cached SymPy derivatives
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

    # Product rule derivatives using Leibniz rule: d^n(uv) = sum_k C(n,k) u^(k) v^(n-k)
    def product_deriv(n, fctr_derivs, W_derivs_list):
        """Compute nth derivative of fctr*W product"""
        return sum(comb(n, k) * fctr_derivs[k] * W_derivs_list[n - k] for k in range(n + 1))
    
    # Create spline for fctr(r)
    if p.lterm == 'density':
        fctr = UnivariateSpline(time_dict['ra'], time_dict['Ha']/time_dict['a']*time_dict['Da'], k=5, s=0)
    elif p.lterm == 'rsd': # -D*f*H/a
        fctr = UnivariateSpline(time_dict['ra'], -time_dict['Ha']/time_dict['a']*time_dict['Da']*time_dict['fa'], k=5, s=0)
    elif p.lterm == 'doppler': # H*D*H/a*f*R
        fctr = UnivariateSpline(time_dict['ra'], time_dict['Ha']*time_dict['Da']*time_dict['Ha']/time_dict['a']*time_dict['fa']*time_dict['Ra'], k=5, s=0)
    elif p.lterm == 'pot': 
        if not p.Newton: # H/a*D*((1.-R)/a+3*f*H**2) 
            fctr = UnivariateSpline(time_dict['ra'], time_dict['Ha']/time_dict['a']*time_dict['Da']*(1.-time_dict['Ra'])/time_dict['a'] \
                                          + 3.*time_dict['fa']*time_dict['Ha']**2, k=5, s=0)
        else: # H/a*D*((1.-R)/a
            fctr = UnivariateSpline(time_dict['ra'], time_dict['Ha']/time_dict['a']*time_dict['Da']*(1.-time_dict['Ra'])/time_dict['a'], k=5, s=0)
    elif p.lterm == 'dpot': # -D*(f-1)/a*H/a
        fctr = UnivariateSpline(time_dict['ra'], -time_dict['Da']*(time_dict['fa']-1.)*time_dict['Ha']/time_dict['a'], k=5, s=0)
    else:
        print('no code for {}'.format(p.lterm))
        raise ValueError(f"Invalid 'p.lterm' parameter: {p.lterm}")
    

    fctr_derivs = np.zeros((10, len(r_list)), dtype=np.float64)  # 0th through 8th derivatives
    fctr_derivs[0] = fctr(r_list)
    for i in range(1, 6):  # 1st through 8th derivatives
        fctr_derivs[i] = fctr.derivative(i)(r_list)

    d5fctr = UnivariateSpline(r_list, fctr_derivs[5], k=5, s=0)
    for i in range(6, 10):  # 6th through 8th derivatives    
       # For higher derivatives, use spline of d5H
       fctr_derivs[i] = d5fctr.derivative(i-5)(r_list)

    # Create cached derivative evaluator
    W_derivs = WindowDerivatives(window_args)

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

    y_list = np.zeros((4, len(qterm_list), len(ell_list), len(r_list)), dtype=np.float64)
    for qt_ind, qt in enumerate(qterm_list):
        r_power_and_derivative = qt-1  if qt in [4, 3, 2] else 0

        # Get analytical derivatives of W - store in list
        W_derivs_list = W_derivs.get_all_derivatives(r_list, r_power_and_derivative, max_deriv=9) # 0th through 8th
         
        # density, rsd, doppler, pot, dpot
        if p.lterm in ['density', 'pot', 'dpot']:
            # f = fctr * W, derivatives start from 0
            derivs = [product_deriv(i+r_power_and_derivative, fctr_derivs, W_derivs_list) for i in range(7)]  # f through d6f
        elif p.lterm=='rsd':
            # f = d²(fctr * W), so shift by 2 orders
            derivs = [product_deriv(i+2+r_power_and_derivative, fctr_derivs, W_derivs_list) for i in range(7)]  # f through d6f
        elif p.lterm== 'doppler':
            # f = d(fctr * W)
            derivs = [product_deriv(i+1+r_power_and_derivative, fctr_derivs, W_derivs_list) for i in range(7)]
        else:
            raise ValueError(f"Invalid 'p.lterm' parameter: {p.lterm}")
        
        f, df, d2f, d3f, d4f, d5f, d6f = derivs
        
        # Compute y_list by applying D operator
        for ind_ell, ell in enumerate(ell_list):
            alpha = ell*(ell+1) - 2.
            
            y_list[0, qt_ind, ind_ell] = f
            y_list[1, qt_ind, ind_ell] = -d2f + 2./r_list*df + alpha/r_list**2*f
            y_list[2, qt_ind, ind_ell] = (d4f - 4./r_list*d3f 
                                +(8./r_list**2 - 2.*alpha/r_list**2)*d2f 
                                +(-8./r_list**3 + 8.*alpha/r_list**3)*df + 
                                 (alpha**2/r_list**4 - 10.*alpha/r_list**4)*f)
            
            # y_list[3, ind_ell] = mathcalD(r_list, y_list[2, ind_ell], ell) 
            y_list[3, qt_ind, ind_ell] = compute_L3_f_analytical(f, df, d2f, d3f, d4f, d5f, d6f, r_list, alpha)
   
    return y_list






























##old stuff
############################################################################# fftlog
#def get_cp_eta_p(k, fctk_list, b_list):
#    Nk = len(k)
#    kmin, kmax = np.min(k), np.max(k)
#
#    res = np.zeros((len(fctk_list), Nk+1), dtype=np.complex128)
#    l=np.arange(Nk)
#    p_list = np.arange(-Nk//2, Nk//2+1)
#    eta_p_list = 2.*np.pi*p_list/np.log(kmax/kmin)
#
#    for ind_fct, fct_k in enumerate(fctk_list):
#        b = b_list[ind_fct]
#        for p in range(-Nk//2, Nk//2+1):
#            res[ind_fct, p+Nk//2] = np.sum(fct_k * k**(-b) * kmin**(-1j*eta_p_list[p+Nk//2]) * np.exp(-2.*1j*np.pi*p*l/Nk)) / Nk
#
#    return res, eta_p_list
#
#def set_bias(k, fctk_list):
#    b_list = np.zeros(len(fctk_list), dtype=np.float64)
#
#    for ind_b, fctk in enumerate(fctk_list):
#        ind=0
#        for i in range(1, 11):
#            b_list[ind_b]+=(np.log(np.abs(fctk[-i])) - np.log(np.abs(fctk[i]))) / (np.log(k[-i]) - np.log(k[i]))
#            ind+=1
#
#        b_list[ind_b]/=ind
#
#    print(' biases: {}'.format(b_list))
#    return b_list
#
#def quadratic_terms(k, fctk, p):
#    """
#    Calculate quadratic terms
#    
#    Parameters:
#    -----------
#    k : array_like
#        Wave number array
#    fctk : array_like
#        Input array fctk
#    lterm : str
#        Linear term type ('density' or other)
#    which : str
#        Derivative type ('d2v', 'd1v', 'd1d', 'd3v', 'd0d')
#    
#    Returns:
#    --------
#    array_like
#        Computed quadratic terms
#    """
#    
#    fctr_scaled =fctk*k**4
#    if p.which == 'd2v':
#        if p.lterm == 'density':
#            fctr_scaled *= k**2
#
#        spline = UnivariateSpline(k, fctr_scaled, k=5, s=0)
#        return [spline.derivative(2)(k),\
#                -2./k * spline.derivative(1)(k),\
#                fctk/k**2]
#
#    elif p.which in ['d1v', 'd1d']:
#        if p.lterm == 'density':
#            fctr_scaled *= k
#        else:
#            fctr_scaled /= k
#        
#        if p.which == 'd1d':
#            fctr_scaled *= k**2
#
#        spline = UnivariateSpline(k, fctr_scaled, k=5, s=0)
#        return [spline.derivative(1)(k), -fctk/k]
#
#    elif p.which == 'd3v':
#        if p.lterm == 'density':
#            fctr_scaled *= k**3
#        else:
#            fctr_scaled *= k
#
#        spline = UnivariateSpline(k, fctr_scaled, k=5, s=0)
#        return [spline.derivative(3)(k),
#                -3./k * spline.derivative(2)(k),
#                3./k**2 * spline.derivative(1)(k),
#                -fctk / k**3]
#
#    elif p.which == 'd0d':
#        if p.lterm != 'density':
#            fctr_scaled /= k**2
#        return [fctr_scaled * k**2]
#    
#    else:
#        raise ValueError(f"Invalid 'which' parameter: {p.which}")
#
#
#def apply_fftlog_old(k, Pk, p):
#    if p.which in ['FG2', 'F2', 'G2', 'dv2']: 
#        if not p.rad:
#            fct_k=[Pk*k**4]
#        else:
#            fct_k=[Pk]
#        np.save(output_dir+'fct_k', np.vstack([k, fct_k]).T)
#    else: 
#        fct_k = quadratic_terms(k, Pk, p) 
#        np.save(output_dir+'fct_k_{}_lterm{}_qterm{}'.format(p.which, p.lterm, p.qterm), np.vstack([k, fct_k]).T)
#
#    b=set_bias(k, fct_k)
#    cp, eta = get_cp_eta_p(k, fct_k, b)
#    return cp, eta, b



