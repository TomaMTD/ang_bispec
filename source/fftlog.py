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
        self.fctk = fctk if p.lterm!='density' else fctk*k**2
        self.Nk = len(k)
        self.kmin, self.kmax = np.min(k), np.max(k)
        self.which = p.which
        self.lterm = p.lterm
        self.mode = p.mode
        self.qterm = p.qterm
        self.rad = p.rad
        
        # Pre-compute common quantities
        self.l = np.arange(self.Nk)
        self.p_list = np.arange(-(self.Nk//2), self.Nk//2) # TAKE CARE -(self.Nk//2) != -self.Nk//2
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
            spline = UnivariateSpline(k, fctr_scaled, k=5, s=0)
            return [spline.derivative(2)(k),\
                    -2./k * spline.derivative(1)(k),\
                    fctr_scaled/k**2]
    
        elif self.which in ['d1v', 'd1d']:
            fctr_scaled /= k
            
            if self.which == 'd1d':
                fctr_scaled *= k**2
    
            spline = UnivariateSpline(k, fctr_scaled, k=5, s=0)
            return [spline.derivative(1)(k), -fctr_scaled/k]
    
        elif self.which == 'd3v':
            fctr_scaled *= k
    
            spline = UnivariateSpline(k, fctr_scaled, k=5, s=0)
            return [spline.derivative(3)(k),
                    -3./k * spline.derivative(2)(k),
                    3./k**2 * spline.derivative(1)(k),
                    -fctr_scaled / k**3]
    
        elif self.which == 'd0d':
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
        res = np.zeros((len(fctk_list), self.Nk), dtype=np.complex128)
        
        for ind_fct, fct_k in enumerate(fctk_list):
            b = b_list[ind_fct]
            for p in range(-self.Nk//2, self.Nk//2):
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

        if self.which in ['FG2', 'F2', 'G2', 'dv2', 'local', 'ortho', 'equi', 'primordial']:
            # Handle special cases
            if self.rad:
                fctk_list = [self.fctk]
            elif self.which in ['local', 'ortho', 'equi', 'primordial']:
                fctk_list = [-self.fctk * self.k]
            else:
                fctk_list = [self.fctk * self.k**4]

            b = self.set_bias(fctk_list)
            cp = self.get_cp_eta_p(fctk_list, b)

            # For radiation F2/G2/dv2, return flattened structure (no qterm nesting)
            if self.rad and self.which in ['F2', 'G2', 'dv2']:
                out_dict['cp'] = cp[0]
                out_dict['b'] = b[0]
                out_dict['fctk'] = fctk_list[0]
                # Remove qterm_list since it's not needed for radiation
                del out_dict['qterm_list']
            else:
                out_dict[0] = {'cp': cp[0],  'b': b[0], 'fctk': fctk_list[0]}

            return out_dict
        
        else:
            # Handle quadratic terms with multiple qterms
            fctk_list = self.compute_quadratic_terms(self.k, self.fctk)
            
            b = self.set_bias(fctk_list)
            print(f'    qterm {self.qterm} biases: {b}')
            cp = self.get_cp_eta_p(fctk_list, b)
            for fctk_ind, fctk in enumerate(fctk_list):
                out_dict[fctk_ind+1] = {'cp': cp[fctk_ind], 'b': b[fctk_ind], 'fctk': fctk }
                
                np.save(f'{output_dir}/fct_k_{self.which}_{self.lterm}', out_dict)
            
            return out_dict

def apply_fftlog(k, fctk, p):
    """
    Generalized FFTLog application that handles multiple qterm values
    """
    processor = FFTLogProcessor(k, fctk, p)
    results = processor.process_all_qterms()

    return results

def apply_fftlog_dict(k, fctk, p):
    """
    Wrapper that returns cp as a dict organized by lterm, similar to fctr structure.

    Parameters:
    -----------
    k : array
        Wave number array
    fctk : array
        Function of k array
    p : parameters object
        Contains which, lterm, qterm, rad, etc.
    lterm_list : list of str
        List of lterm values to compute (e.g., ['density', 'rsd'])

    Returns:
    --------
    dict: cp_dict organized by lterm
        cp_dict = {lterm: {cp data from apply_fftlog}}
        For radiation F2/G2/dv2: cp_dict = {'rad': {cp data}}
    """
    print(f'    fftlog processing')

    cp_dict = {}

    # Special handling for radiation F2/G2/dv2 cases
    if p.rad and p.which in ['F2', 'G2', 'dv2']:
        # For radiation, only compute cp coefficients once (same for F2, G2, dv2)
        processor = FFTLogProcessor(k, fctk, p)
        cp_dict['rad'] = processor.process_all_qterms()
        return cp_dict

    # Apply fftlog for each lterm (non-radiation or other cases)

    lterm_back = p.lterm
    for p.lterm in ['density', 'not_density']:
        processor = FFTLogProcessor(k, fctk, p)
        cp_dict[p.lterm] = processor.process_all_qterms()
    p.lterm = lterm_back

    return cp_dict

