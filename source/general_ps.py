import numpy as np
import os, sys
import time
from numba import njit, prange
import h5py
import threading
from fractions import Fraction
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad

from param_used import *
from mathematica import *

# Global lock for HDF5 file access
# Note: threading.Lock() only works within a single process
# For multi-process safety (e.g., job arrays), HDF5's built-in locking is used in save_to_hdf5
hdf5_lock = threading.Lock()

@njit(parallel=True)
def compute_hyp21_grid_numba(t_grid, nu_p_grid, ell_grid):
    """
    Compute hyp21 values on the grid using numba for speed
    """
    n_t = len(t_grid)
    n_nu_p = len(nu_p_grid)
    n_ell = len(ell_grid)
    
    # Preallocate result array
    result = np.zeros((n_t, n_nu_p, n_ell), dtype=np.complex128)
    
    # Compute in parallel over t values
    for i_t in prange(n_t):
        t = t_grid[i_t]
        for i_nu_p, nu_p in enumerate(nu_p_grid):
            for i_ell, ell in enumerate(ell_grid):
                result[i_t, i_nu_p, i_ell] = I_nacked(nu_p, t, ell)
    return result


@njit
def cubic_spline_interp(xi, x, y):
    """Simple cubic spline interpolation for a single point"""
    n = len(x)
    if xi <= x[0]:
        return y[0]
    if xi >= x[-1]:
        return y[-1]
    
    # Find the interval
    i = 0
    while i < n-1 and x[i+1] < xi:
        i += 1
    
    if i == n-1:
        i = n-2
    
    # Cubic interpolation using 4 points when possible
    if i == 0:
        # Use points 0,1,2,3
        x0, x1, x2, x3 = x[0], x[1], x[2], x[3]
        y0, y1, y2, y3 = y[0], y[1], y[2], y[3]
    elif i == n-2:
        # Use points n-4,n-3,n-2,n-1
        x0, x1, x2, x3 = x[n-4], x[n-3], x[n-2], x[n-1]
        y0, y1, y2, y3 = y[n-4], y[n-3], y[n-2], y[n-1]
    else:
        # Use points i-1,i,i+1,i+2
        x0, x1, x2, x3 = x[i-1], x[i], x[i+1], x[i+2]
        y0, y1, y2, y3 = y[i-1], y[i], y[i+1], y[i+2]
    
    # Lagrange interpolation
    L0 = ((xi-x1)*(xi-x2)*(xi-x3))/((x0-x1)*(x0-x2)*(x0-x3))
    L1 = ((xi-x0)*(xi-x2)*(xi-x3))/((x1-x0)*(x1-x2)*(x1-x3))
    L2 = ((xi-x0)*(xi-x1)*(xi-x3))/((x2-x0)*(x2-x1)*(x2-x3))
    L3 = ((xi-x0)*(xi-x1)*(xi-x2))/((x3-x0)*(x3-x1)*(x3-x2))
    
    return y0*L0 + y1*L1 + y2*L2 + y3*L3
    
@njit
def quadratic_interp(xi, x, y):
    """Quadratic interpolation using 3 nearest points"""
    n = len(x)
    if xi <= x[0]:
        return y[0]
    if xi >= x[-1]:
        return y[-1]
    
    # Find the interval
    i = 0
    while i < n-1 and x[i+1] < xi:
        i += 1
    
    # Choose 3 points for quadratic interpolation
    if i == 0:
        idx = [0, 1, 2]
    elif i == n-1:
        idx = [n-3, n-2, n-1]
    else:
        idx = [i-1, i, i+1]
    
    x0, x1, x2 = x[idx[0]], x[idx[1]], x[idx[2]]
    y0, y1, y2 = y[idx[0]], y[idx[1]], y[idx[2]]
    
    # Lagrange interpolation with 3 points
    L0 = ((xi-x1)*(xi-x2))/((x0-x1)*(x0-x2))
    L1 = ((xi-x0)*(xi-x2))/((x1-x0)*(x1-x2))
    L2 = ((xi-x0)*(xi-x1))/((x2-x0)*(x2-x1))
    
    return y0*L0 + y1*L1 + y2*L2


@njit
def sump_cp_I_vectorized_precompute(r_list, chi_list, t_grid, nu_p_grid, cp_list, F12):
    """Vectorized version computing for all r,chi pairs at once"""
    N = len(nu_p_grid)  # Number of nu_p values

    # Pre-allocate result array
    result = np.zeros((len(chi_list), len(r_list)), dtype=np.complex128)
    
    # Compute t_chi ratios for all combinations
    for i_chi, chi in enumerate(chi_list):
        for i_r, r in enumerate(r_list):
            t_chi = r / chi
            
            for i in range(N//2):
                eval = chi**(-nu_p_grid[i])*cubic_spline_interp(t_chi, t_grid, F12[:, i])
                result[i_chi, i_r] += 2*cp_list[i] * eval
            i=N//2
            eval = chi**(-nu_p_grid[i])*cubic_spline_interp(t_chi, t_grid, F12[:, i])
            result[i_chi, i_r] += cp_list[i] * eval
            
    return result.real


@njit
def simpson_numba(y, x):
    n = len(x)
    if n < 3 or n % 2 == 0:
        print('problem with Simpson\'s rule: n =', n)
        raise ValueError("Simpson's rule requires an odd number of samples.")
    h = (x[-1] - x[0]) / (n - 1)
    result = y[0] + y[-1]
    for i in range(1, n - 1, 2):
        result += 4 * y[i]
    for i in range(2, n - 2, 2):
        result += 2 * y[i]
    return result * h / 3.0

@njit
def r_integration_vectorized_precompute(Nchi, r_list, chi_list, y1, t_grid, nu_p, cp_list, F12):
    """
    Vectorized version of r_integration that computes sump_cp_I for all chi values at once
    """
    # Compute sump_cp_I for all (chi, r) pairs for both nu_p sets
    sump_cp_I_matrix = sump_cp_I_vectorized_precompute(r_list, chi_list, t_grid, nu_p, cp_list, F12)  # shape: (Nchi, Nr)

    # Initialize output arrays
    s_cp_I_list = np.zeros(Nchi)
    
    # Perform integration for each chi
    for ind in range(Nchi):
        # Extract the row for this chi value
        
        # Compute integrands
        integrand1 = y1 * sump_cp_I_matrix[ind, :]
        
        # Integrate using Simpson's rule
        s_cp_I_list[ind] = simpson_numba(integrand1, r_list)

        #spline = UnivariateSpline(r_list, integrand1, k=5, s=0)
        #s_cp_I_list[ind]= quad(spline, r_list[0], r_list[-1])[0]
    
    return s_cp_I_list


@njit(parallel=True)
def compute_integral_precompute(ell_list, chi_list, r_list, t_grid, nu_p, cp_list, F12, y1):
    Nchi = len(chi_list)

    s_cp_I_tab = np.zeros((len(ell_list), len(chi_list)))
    for ind_ell in prange(len(ell_list)):
        print('         Computing ell='+str(ell_list[ind_ell]))

        s_cp_I_tab[ind_ell] = r_integration_vectorized_precompute(Nchi, r_list, chi_list, \
                y1[ind_ell], t_grid, nu_p, cp_list, F12[:,:,ind_ell])
    return s_cp_I_tab/ (4*np.pi) 



@njit(parallel=True)
def compute_integral_Am_numba(chi_list, r_list, fctr_r, ell):
    """
    Compute Am integral for non-radiation case using Il(-1, t, ell) directly

    Returns: array of shape (len(chi_list),)
    """
    result = np.zeros(len(chi_list))

    for i_chi in prange(len(chi_list)):
        chi = chi_list[i_chi]
        integrand = np.zeros(len(r_list))

        for i_r, r in enumerate(r_list):
            t = r / chi
            if t > 1:
                fact = t
                t_use = 1. / t
            else:
                fact = 1.
                t_use = t

            # Compute Il(-1, t, ell)
            Il_val = chi * fact * Il(-1+0.j, t_use+0.j, ell)
            integrand[i_r] = fctr_r[i_r] * Il_val.real

        # Integrate using Simpson's rule
        result[i_chi] = simpson_numba(integrand, r_list) / (2 * np.pi**2)

    return result




def save_to_hdf5(p, filename, group_path, data, metadata=None):
    """
    Multi-process safe HDF5 saving function with simple structure.

    New structure: Instead of storing data as (n_ell, n_chi) arrays,
    each ell is stored as a separate dataset under each lterm subgroup:

    group_path/
        lterm1/
            ell_2: (n_chi,)
            ell_3: (n_chi,)
            ...
        lterm2/
            ell_2: (n_chi,)
            ...
        chi_list: (n_chi,)

    This eliminates the need for array expansion and complex merging.
    """
    max_retries = 10
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            with h5py.File(filename, 'a', locking=True) as f:
                # Create or get the main group
                if group_path not in f:
                    group = f.create_group(group_path)
                    print(f'    Created group: {group_path}')
                else:
                    group = f[group_path]

                # Save chi_list at group level (only once)
                if 'chi_list' not in group:
                    group.create_dataset('chi_list', data=data['chi_list'])

                # Get ell_list from data
                ell_list = data['ell_list']

                # For each data key (e.g., 'density', 'rsd', 'f0', 'fm2'), create subgroup
                for key, value in data.items():
                    if key in ['chi_list', 'ell_list']:
                        continue  # Skip coordinate arrays

                    # Create subgroup for this lterm/multipole if it doesn't exist
                    if key not in group:
                        subgroup = group.create_group(key)
                        print(f'    Created subgroup: {group_path}/{key}')
                    else:
                        subgroup = group[key]

                    # Save data for each ell separately
                    # value shape can be (n_ell, n_chi) or (n_components, n_ell, n_chi)
                    for i, ell in enumerate(ell_list):
                        ell_key = f'ell_{ell}'

                        # Extract data for this ell
                        if len(value.shape) == 3:
                            # Shape is (n_components, n_ell, n_chi)
                            ell_data = value[:, i, :]  # shape: (n_components, n_chi)
                        elif len(value.shape) == 2:
                            # Shape is (n_ell, n_chi)
                            ell_data = value[i, :]  # shape: (n_chi,)
                        else:
                            print(f'      WARNING: Unexpected shape for {key}: {value.shape}')
                            continue

                        # Save or overwrite this ell's data
                        if ell_key in subgroup:
                            # Overwrite existing
                            del subgroup[ell_key]

                        subgroup.create_dataset(ell_key, data=ell_data)
                        print(f'      Saved {group_path}/{key}/{ell_key} with shape {ell_data.shape}')

                # Save metadata
                if metadata:
                    for key, value in metadata.items():
                        group.attrs[key] = value

                return True

        except (BlockingIOError, OSError) as e:
            # Both BlockingIOError and OSError with errno 11 are locking errors
            if hasattr(e, 'errno') and e.errno == 11:
                # errno 11 = Resource temporarily unavailable (lock conflict)
                if attempt < max_retries - 1:
                    print(f'    File locked (attempt {attempt+1}/{max_retries}), retrying in {retry_delay}s...')
                    time.sleep(retry_delay)
                else:
                    print(f'    ERROR: Failed to acquire lock after {max_retries} attempts')
                    raise
            elif isinstance(e, BlockingIOError):
                if attempt < max_retries - 1:
                    print(f'    File locked (attempt {attempt+1}/{max_retries}), retrying in {retry_delay}s...')
                    time.sleep(retry_delay)
                else:
                    print(f'    ERROR: Failed to acquire lock after {max_retries} attempts')
                    raise
            else:
                # Some other OSError, re-raise immediately
                raise
        except Exception as e:
            print(f'    ERROR in save_to_hdf5: {e}')
            raise


def compute_integral_F2_G2_dv2(p, ell_list, chi_list, r_list, t_grid, cp_dict, fctr_dict):
    """
    Compute integrals for F2/G2/dv2 cases (bispectrum Am and Il terms)

    Structure:
    - For F2: fm2 (3 components), fm4 (2 components)
    - For G2/dv2: f0 (4 components), fm2 (3 components)
    - For radiation: fm2_rad (2 components), fm4_rad (2 components)
    """
    # Ensure ell_list is numpy array for numba compatibility
    ell_list = np.asarray(ell_list)

    print('----------------------------------------------------')
    print(f'  Computing integrals for {p.which}')

    output_filename = f'{output_dir}Cls.h5'
    # Note: File creation/initialization is now handled by save_to_hdf5 with proper locking

    # Helper function to save results
    def save_result(multipole_name, result, is_radiation):
        data_to_save = {
            multipole_name: result,
            'ell_list': ell_list,
            'chi_list': chi_list
        }
        metadata = {
            'which': p.which,
            'multipole': multipole_name,
            'rad': is_radiation
        }
        save_to_hdf5(p, output_filename, p.which, data_to_save, metadata)

    if p.rad:
        # Radiation case: use cp_dict for Il integrals
        # cp coefficients are the same for F2, G2, dv2 (stored under 'rad' key)
        rad_key = 'rad'
        fctr_key = f'{p.which}_rad'

        if rad_key not in cp_dict or fctr_key not in fctr_dict:
            print(f'  Warning: {rad_key} not found in cp_dict or {fctr_key} not in fctr_dict, skipping radiation')
            return

        cp = cp_dict[rad_key]
        fctr = fctr_dict[fctr_key]
        middle = len(cp['eta_p']) // 2

        # Radiation: only compute independent components (fm2R_0, fm4R_0)
        # Note: fm2R_1 = -fm2R_0/2, fm4R_1 = -fm4R_0/2 (will be derived later)
        # F2: fm2(level=1, nu=-1), fm4(level=0, nu=-1)
        # G2/dv2: fm2(level=0, nu=-1), fm4(level=0, nu=-3)
        is_F2 = (p.which == 'F2')
        multipoles = {
            'fm2_rad': {'qterms': [0], 'level': 1 if is_F2 else 0, 'nu_offset': -1},
            'fm4_rad': {'qterms': [1], 'level': 0, 'nu_offset': -1 if is_F2 else -3}
        }

        # Compute integral for each multipole type (radiation uses cp expansion)
        for multipole_name, multipole_info in multipoles.items():
            print(f'    Computing {multipole_name}')

            qterm_indices = multipole_info['qterms']
            level = multipole_info['level']
            nu_offset = multipole_info['nu_offset']
            n_components = len(qterm_indices)

            # Compute nu_p for this multipole using the appropriate offset
            nu_p = nu_offset + cp['b'] + 1j*cp['eta_p']

            # Precompute hypergeometric function for this nu_p
            F12 = compute_hyp21_grid_numba(t_grid, nu_p[:middle+1], ell_list)

            # Initialize result array: (n_components, n_ell, n_chi)
            result = np.zeros((n_components, len(ell_list), len(chi_list)), dtype=np.float64)

            # Loop over components to compute
            for comp_idx, qt in enumerate(qterm_indices):
                print(f'      Computing component {comp_idx} (fctr index {qt}, level {level}, nu_offset {nu_offset})')

                # Compute integral using the appropriate fctr component and level
                start_time = time.time()
                integral_result = compute_integral_precompute(
                    ell_list, chi_list, r_list, t_grid,
                    nu_p, cp['cp'], F12, fctr[level, qt]  # Use specified level
                )

                result[comp_idx] = chi_list**2 * integral_result
                print(f'        Integral done in {time.time()-start_time:.2f} seconds')

            # Save to HDF5
            save_result(multipole_name, result, is_radiation=True)

    else:
        # Non-radiation case: compute Am terms (no cp needed, just Il(-1, t, ell))
        if p.Newton:
            if p.which in ['F2']:
                print(f'  No Newtonian terms for {p.which}, skipping non-radiation')
                return
            # For G2 Newton: only f0 terms

        key = p.which
        if key not in fctr_dict:
            print(f'  Warning: {key} not found in fctr_dict, skipping non-radiation')
            return

        fctr = fctr_dict[key]

        # Determine multipole structure for non-radiation
        # Pattern: First multipole has qterms [0, 1], level 1
        #          Second multipole (if non-Newton) has qterms [2], level 0
        # Note: component[1] = -component[0]/2 for all multipoles (derived later)

        # First multipole name and structure
        if p.which in ['G2', 'dv2']:
            if p.Newton:
                first_name = 'f0_newton'
            else:
                first_name = 'f0'
        else:
            first_name = 'fm2'

        multipoles = {
            first_name: {'qterms': [0, 1] if not p.Newton else [0], 'level': 1}
        }

        # Second multipole (only for non-Newton)
        if not p.Newton:
            second_name = 'fm4' if p.which == 'F2' else 'fm2'
            multipoles[second_name] = {'qterms': [2], 'level': 0}

        # Compute Am integrals (use Il(-1, t, ell) analytically - no cp expansion)
        for multipole_name, multipole_info in multipoles.items():
            print(f'    Computing {multipole_name} (Am terms)')

            qterm_indices = multipole_info['qterms']
            level = multipole_info['level']
            n_components = len(qterm_indices)

            # Initialize result array: (n_components, n_ell, n_chi)
            result = np.zeros((n_components, len(ell_list), len(chi_list)), dtype=np.float64)

            # Loop over ell values
            for ind_ell, ell in enumerate(ell_list):
                print(f'      ell={ell} ({ind_ell+1}/{len(ell_list)})')

                # Compute integrals for specified components
                for comp_idx, qt in enumerate(qterm_indices):
                    # Get fctr for this component - use specified level
                    fctr_r = fctr[level, qt, ind_ell, :]  # shape: (len(r_list),)

                    # Compute integral over r for all chi values
                    start_time = time.time()
                    integral_result = compute_integral_Am_numba(
                        chi_list, r_list, fctr_r, ell
                    )

                    result[comp_idx, ind_ell, :] = chi_list**2 * integral_result
                    print(f'        Computed component {comp_idx} (qterm {qt}) in {time.time()-start_time:.2f} seconds')

            # Save to HDF5
            save_result(multipole_name, result, is_radiation=False)


def get_nm_values(which):
    """Get the (n,m) pairs for different 'which' cases"""
    nm_mapping = {
        'FG2': [(-2, 0), (0, 0), (2, 0)],
        'd1v': [(-1, 1)],
        'd2v': [(0, 2)],
        'd3v': [(1, 3)],
        'd1d': [(1, 1)],
        'all_primordial': [(1, 0), (0, 0), (1./3., 0), (2./3., 0)],
        'local': [(1, 0), (0, 0)],
        'equi':  [(1, 0), (1./3., 0), (2./3., 0)],  # Needs all three: λ=1, λ=1/3, λ=2/3
        'ortho': [(2./3., 0)],
    }
    return nm_mapping.get(which, [(0, 0)]) 


def compute_integral_generalized(p, ell_list, chi_list, r_list, t_grid, cp_dict, fctr_dict, lterm_list):
    """
    Generalized computation function

    Parameters:
    -----------
    p : parameter object with attributes omega_m, H0, etc.
    cp_dict : dict organized by lterm containing cp data
    fctr_dict : dict organized by lterm containing fctr arrays
    chi_list, ell_list, r_list, t_grid : coordinate arrays
    """
    # Ensure ell_list is numpy array for numba compatibility
    ell_list = np.asarray(ell_list)

    print('---------------------------------------------------- Integration processing')
    

    def check_computation_exists(filename, group_path, lterm):
        """Check if a specific computation already exists"""
        try:
            with h5py.File(filename, 'r') as f:
                exists = group_path in f and lterm in f[group_path]
                return exists
        except (OSError, KeyError, IOError):
            print(f'        group {nm_name} does not exist')
            return False

    # Determine if we're processing F2/G2/dv2 or FG2/d1v/etc
    if p.which in ['F2', 'G2', 'dv2']:
        # F2/G2/dv2: different structure
        compute_integral_F2_G2_dv2(p, ell_list, chi_list, r_list, t_grid, cp_dict, fctr_dict)
        return

    # Original behavior for FG2/d1v/etc
    for lterm in lterm_list:
        # Extract cp and fctr for current lterm
        cp = cp_dict[lterm if lterm=='density' else 'not_density']
        fctr = fctr_dict[lterm]

        middle = len(cp['eta_p']) // 2

        output_filename = f'{output_dir}Cls.h5'
        # Note: File creation/initialization is now handled by save_to_hdf5 with proper locking


        # Main computation loop
        nm_pairs = get_nm_values(p.which)
        
        # Compute factors that depend on lterm and which
        if p.mode=='primordial':
            stuff2 = (2./3./omega_m/H0**2)
        else: 
            stuff2 = (2./3./omega_m/H0**2)**2

        if lterm in ['pot', 'dpot']:
            stuff2 /= (2./3./omega_m/H0**2)

        print(f'  Computing for lterm = {lterm}')
        for n, m in nm_pairs:
            # for m==0, the n values are taken into account in cp
            n_eff = n if m == 0 else 0

            if n_eff == 2:
                power_reduction = 1
            elif n_eff == 4:
                power_reduction = 2
            else:
                power_reduction = 0
            
            group_path = f'{"primordial_" if p.mode=="primordial" else ""}n_{n if isinstance(n, int) else f"{n:.2f}"}_m_{m}' 

            # Check if computation already exists
            if not p.force and check_computation_exists(output_filename, group_path, lterm):
                print(f'    Results for (n,m)=({group_path}), lterm={lterm} already exist, skipping (use force=True to overwrite)')
                continue


            # Initialize result for this (which, lterm, n) combination
            result = np.zeros((len(ell_list),len(chi_list)), dtype=np.float64)
            
            # Loop over qterms
            for qt_ind, qt in enumerate(cp['qterm_list']):
                if len(cp['qterm_list'])>1: print(f'      Computing qterm: {qt}/{len(cp["qterm_list"])}')
                
                # Compute nu_p
                if p.mode == 'primordial':
                    power_reduction = 2 if n_eff==0 else 1

                    nu_p = 2 + cp[qt]['b'] + 1j*cp['eta_p'] + n_eff*(n_s-4) - 2*power_reduction
                else:
                    nu_p = 1 + cp[qt]['b'] + 1j*cp['eta_p'] + n_eff - 2*power_reduction
                
                # Precompute hypergeometric function
                #start_time = time.time()
                F12 = compute_hyp21_grid_numba(t_grid, nu_p[:middle+1], ell_list)
                #print(f'        2F1 precomputation done in {time.time()-start_time:.2f} seconds')

                # Compute integral
                start_time = time.time()
                integral_result = compute_integral_precompute(
                    ell_list, chi_list, r_list, t_grid, 
                    nu_p, cp[qt]['cp'], F12, fctr[power_reduction, qt_ind]
                )
                
                # Sum the contribution
                if p.mode == 'primordial':
                    result += (2*np.pi**2*A_s/(k_pivot/h)**(n_s-1))**n * stuff2 * integral_result
                else:
                    result += stuff2 * integral_result
                print(f'        Integral computation done in {time.time()-start_time:.2f} seconds')
            
            data_to_save = {
                lterm: 2./np.pi*result,
                'ell_list': ell_list,  # Add ell_list to each group
                'chi_list': chi_list   # Add chi_list to each group
            }
            metadata = {
                'n': n,
                'm': m,
                'which': p.which,
                'lterm': lterm,
            }
            
            save_to_hdf5(p, output_filename, group_path, data_to_save, metadata)

