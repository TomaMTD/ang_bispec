import numpy as np
import os, sys
import time
from numba import njit, prange
import h5py
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
from scipy.integrate import simpson

import fctr as fctr_module
import bispectrum
from param_used import *
from mathematica import *


@njit(parallel=True)
def compute_hyp21_grid_numba(t_grid, nu_p_grid, ell_grid):
    """
    Compute hyp21 values on the grid using numba for speed

    t_grid is (n_t, n_ell): each ell has its OWN t grid, concentrated on the support of I_ell
    (see mathematica.build_t_grid), so column i_ell must be used with ell_grid[i_ell].
    """
    n_t = t_grid.shape[0]
    n_nu_p = len(nu_p_grid)
    n_ell = len(ell_grid)

    # Preallocate result array
    result = np.zeros((n_t, n_nu_p, n_ell), dtype=np.complex128)

    # Compute in parallel over t values
    for i_t in prange(n_t):
        for i_nu_p, nu_p in enumerate(nu_p_grid):
            for i_ell, ell in enumerate(ell_grid):
                result[i_t, i_nu_p, i_ell] = I_nacked(nu_p, t_grid[i_t, i_ell], ell)
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
def cubic_interp_uniform(xi, x, y):
    """
    Same 4-point Lagrange as cubic_spline_interp, but for a UNIFORMLY spaced x, where the
    bracketing index is arithmetic instead of a linear scan. Identical results: the scan stops
    at x[i] < xi <= x[i+1] and floor() picks the same i everywhere except exactly on a node,
    where the two choose different 4-point stencils -- but both contain that node, and Lagrange
    evaluated at a node of its own stencil returns that node's value exactly.

    Used for the y1 lookup in r_integration, which happens n_t times per chi; the scan (~250
    steps over a 501-point grid) was the dominant cost there once I_ell stopped being interpolated.
    """
    n = len(x)
    if xi <= x[0]:
        return y[0]
    if xi >= x[-1]:
        return y[-1]

    dx = (x[-1] - x[0]) / (n - 1)
    i = int((xi - x[0]) / dx)
    if i > n - 2:
        i = n - 2

    # same stencil choice as cubic_spline_interp
    if i == 0:
        j = 0
    elif i == n - 2:
        j = n - 4
    else:
        j = i - 1

    x0, x1, x2, x3 = x[j], x[j+1], x[j+2], x[j+3]
    y0, y1, y2, y3 = y[j], y[j+1], y[j+2], y[j+3]

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
def sump_cp_I_vectorized_precompute(chi_list, t_grid, nu_p_grid, cp_list, F12):
    """
    sum_p cp_p chi^(-nu_p) I_ell(nu_p, t), for every (chi, t) with t on t_grid.

    Evaluated directly ON the nodes where F12 is tabulated, so I_ell is never interpolated:
    the caller integrates over r = chi*t, which places every sample exactly on a node.
    That is why r_list is not an argument any more. Shape (n_chi, n_t).
    """
    N = len(nu_p_grid)
    n_t = len(t_grid)
    result = np.zeros((len(chi_list), n_t), dtype=np.complex128)

    for i_chi, chi in enumerate(chi_list):
        # p-loop outside t-loop: chi**(-nu_p) is then computed once per (chi, p), not per node
        for i in range(N//2):
            w = 2*cp_list[i] * chi**(-nu_p_grid[i])
            for i_t in range(n_t):
                result[i_chi, i_t] += w * F12[i_t, i]
        i = N//2
        w = cp_list[i] * chi**(-nu_p_grid[i])
        for i_t in range(n_t):
            result[i_chi, i_t] += w * F12[i_t, i]

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
    Integrate y1(r) * sum_p cp_p I_ell(nu_p, r/chi) over r, for every chi.

    The integration variable is r = chi*t sampled on t_grid, NOT the global r_list:
      - I_ell is tabulated on t_grid, so every sample lands exactly on a node and I_ell is
        never interpolated. Only y1 is, and y1 is smooth (window times background functions).
      - build_t_grid concentrates t_grid on the support of I_ell, so the oscillation stays
        resolved as ell grows. With the fixed r_list it did not: the peak narrows like 1/ell
        while the grid stays put, so it slides between samples and aliases into sign-flipping
        noise (~1e-2 by ell~1000).
    """
    # (Nchi, n_t), on the t_grid nodes
    sump_cp_I_matrix = sump_cp_I_vectorized_precompute(chi_list, t_grid, nu_p, cp_list, F12)

    n_t = len(t_grid)
    rmin, rmax = r_list[0], r_list[-1]
    s_cp_I_list = np.zeros(Nchi)

    r_sub = np.empty(n_t)
    integrand = np.empty(n_t)
    integrand_r = np.empty(len(r_list))

    for ind in range(Nchi):
        chi = chi_list[ind]

        if t_grid[0] <= rmin/chi and t_grid[-1] >= rmax/chi:
            # The support of I_ell is WIDER than the physical range, so it restricts nothing
            # and there is nothing to concentrate on.
            for j in range(len(r_list)):
                integrand_r[j] = y1[j] * cubic_spline_interp(r_list[j]/chi, t_grid,
                                                             sump_cp_I_matrix[ind, :])
            s_cp_I_list[ind] = simpson_numba(integrand_r, r_list)

        else:
            for j in range(n_t):
                r = chi*t_grid[j]
                r_sub[j] = r
                if r < rmin or r > rmax:
                    integrand[j] = 0.
                else:
                    integrand[j] = cubic_interp_uniform(r, r_list, y1) * sump_cp_I_matrix[ind, j]

            s_cp_I_list[ind] = simpson_numba(integrand, r_sub)

    return s_cp_I_list


@njit(parallel=True)
def compute_integral_precompute(ell_list, chi_list, r_list, t_grid, nu_p, cp_list, F12, y1):
    Nchi = len(chi_list)

    s_cp_I_tab = np.zeros((len(ell_list), len(chi_list)))
    for ind_ell in prange(len(ell_list)):
        print('         Computing ell='+str(ell_list[ind_ell]))

        # t_grid is (n_t, n_ell): hand each ell its own column, matching F12[:,:,ind_ell]
        s_cp_I_tab[ind_ell] = r_integration_vectorized_precompute(Nchi, r_list, chi_list, \
                y1[ind_ell], t_grid[:, ind_ell], nu_p, cp_list, F12[:,:,ind_ell])
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


def _save_to_fallback_npy(hdf5_filename, group_path, data, metadata=None):
    """
    Fallback function to save data to a numpy file when HDF5 locking fails.

    Saves the entire data dictionary as a single .npy file.
    Filename format: fallback_{group_path}_{timestamp}_pid{pid}.npy

    Parameters:
    -----------
    hdf5_filename : str
        Original HDF5 filename (used to determine output directory)
    group_path : str
        HDF5 group path (e.g., 'n_0_m_0')
    data : dict
        Dictionary containing the data arrays
    metadata : dict, optional
        Dictionary containing metadata
    """
    # Create fallback directory next to the HDF5 file
    hdf5_dir = os.path.dirname(hdf5_filename)
    hdf5_basename = os.path.basename(hdf5_filename).replace('.h5', '')
    fallback_dir = os.path.join(hdf5_dir, f'{hdf5_basename}_fallback')
    os.makedirs(fallback_dir, exist_ok=True)

    # Clean up group_path for filename
    safe_group_path = group_path.replace('/', '_').replace(' ', '_')

    # Create unique filename using timestamp and process ID
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    pid = os.getpid()
    filename = f'fallback_{safe_group_path}_{timestamp}_pid{pid}.npy'
    filepath = os.path.join(fallback_dir, filename)

    # Prepare data to save (combine data and metadata)
    save_dict = {
        'group_path': group_path,
        'data': data,
        'metadata': metadata
    }

    # Save as numpy file
    np.save(filepath, save_dict, allow_pickle=True)

    print(f'    FALLBACK: Saved to {filepath}')
    print(f'    File contains group_path: {group_path}, data keys: {list(data.keys())}')


def save_to_hdf5(filename, group_path, data, metadata=None):
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

    If max_retries is reached due to file locking, saves to a fallback numpy file instead.
    """
    max_retries = 50
    retry_delay = 10  # seconds

    for attempt in range(max_retries):
        try:
            # Try to open the HDF5 file - may fail if file is corrupted or locked
            with h5py.File(filename, 'a', locking=True) as f:
                # Create or get the main group
                if group_path not in f:
                    group = f.create_group(group_path)
                    print(f'    Created group: {group_path}')
                else:
                    group = f[group_path]

                # Save chi_list at group level
                # If it exists but has wrong size, delete and recreate
                if 'chi_list' in group:
                    old_size = len(group['chi_list'])
                    new_size = len(data['chi_list'])
                    if old_size != new_size:
                        del group['chi_list']
                        group.create_dataset('chi_list', data=data['chi_list'])
                        print(f'    Updated chi_list (size changed from {old_size} to {new_size})')
                else:
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
            # Handle various HDF5 file access errors
            error_msg = str(e).lower()

            # Check if it's a locking error (errno 11 or BlockingIOError)
            is_locking_error = (
                (hasattr(e, 'errno') and e.errno == 11) or
                isinstance(e, BlockingIOError)
            )

            # Check if it's a file corruption error (truncated, corrupted, etc.)
            is_corruption_error = any(keyword in error_msg for keyword in
                ['truncated', 'corrupted', 'unable to open', 'unable to synchronously open'])

            if is_locking_error:
                # Locking error - retry
                if attempt < max_retries - 1:
                    print(f'    File locked (attempt {attempt+1}/{max_retries}), retrying in {retry_delay}s...')
                    time.sleep(retry_delay)
                else:
                    # Max retries reached - save to fallback file
                    print(f'    ERROR: Failed to acquire HDF5 lock after {max_retries} attempts')
                    print(f'    FALLBACK: Saving to independent numpy file instead...')
                    _save_to_fallback_npy(filename, group_path, data, metadata)
                    return False

            elif is_corruption_error:
                # File appears corrupted (likely being written by another process)
                if attempt < max_retries - 1:
                    print(f'    File appears corrupted/truncated (attempt {attempt+1}/{max_retries}), retrying in {retry_delay}s...')
                    print(f'    (This usually means another job is currently writing to the file)')
                    time.sleep(retry_delay)
                else:
                    # Max retries reached - save to fallback file
                    print(f'    ERROR: File still appears corrupted after {max_retries} attempts')
                    print(f'    FALLBACK: Saving to independent numpy file instead...')
                    _save_to_fallback_npy(filename, group_path, data, metadata)
                    return False

            else:
                # Some other OSError we don't recognize - re-raise
                print(f'    ERROR: Unexpected OSError: {e}')
                raise
        except Exception as e:
            print(f'    ERROR in save_to_hdf5: {e}')
            raise

    # Should never reach here, but just in case
    print(f'    ERROR: Exhausted all retries without proper error handling')
    _save_to_fallback_npy(filename, group_path, data, metadata)
    return False


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
        save_to_hdf5(output_filename, p.which, data_to_save, metadata)

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
        'primordial': [(1, 0), (0, 0), (1./3., 0), (2./3., 0)],
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
    

    def check_computation_exists(filename, group_path, lterm, ell_list):
        """Check if a specific computation already exists for all ells in ell_list"""
        max_retries = 5
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                with h5py.File(filename, 'r') as f:
                    # Check if group and lterm subgroup exist
                    if group_path not in f or lterm not in f[group_path]:
                        return False

                    # Check if all ells exist within the lterm subgroup
                    lterm_group = f[group_path][lterm]
                    for ell in ell_list:
                        ell_key = f'ell_{ell}'
                        if ell_key not in lterm_group:
                            return False  # At least one ell is missing

                    return True  # All ells exist
            except (BlockingIOError, OSError) as e:
                # Handle locking errors (errno 11)
                if hasattr(e, 'errno') and e.errno == 11 or isinstance(e, BlockingIOError):
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        # After max retries, assume doesn't exist
                        return False
                # Other OSErrors - file doesn't exist
                return False
            except (KeyError, IOError, FileNotFoundError):
                # File or group doesn't exist yet - this is expected on first run
                return False

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
        if p.which in ['local', 'equi', 'ortho', 'primordial']:
            stuff2 = (2./3./omega_m/H0**2)
        else: 
            stuff2 = (2./3./omega_m/H0**2)**2

        if lterm in ['pot', 'dpot', 'pot_fnl']:
            stuff2 /= (2./3./omega_m/H0**2)

        print(f'  Computing for lterm = {lterm}')
        for n, m in nm_pairs:
            # for m==0, the n values are taken into account in cp
            n_eff = n if m == 0 else 0

            group_path = f'{"primordial_" if p.which=="primordial" else ""}n_{n if isinstance(n, int) else f"{n:.2f}"}_m_{m}' 

            ## Check if computation already exists for all ells
            if not p.force and check_computation_exists(output_filename, group_path, lterm, ell_list):
                print(f'    Results for (n,m)=({group_path}), lterm={lterm} already exist for all ells, skipping (use force=True to overwrite)')
                continue

            # Initialize result for this (which, lterm, n) combination
            result = np.zeros((len(ell_list),len(chi_list)), dtype=np.float64)
            
            # Loop over qterms
            for qt_ind, qt in enumerate(cp['qterm_list']):

                power_reduction=0
                if p.which in ['local', 'equi', 'ortho', 'primordial']:
                    Renu = 2 + cp[qt]['b'] + n_eff*(n_s-4)
                else:
                    Renu = 1 + cp[qt]['b'] + n_eff

                while Renu-2*power_reduction>=-1:
                    power_reduction+=1

                if power_reduction > 3: power_reduction=3

                if len(cp['qterm_list'])>1: print(f'      Integrating {group_path} qterm: {qt}/{len(cp["qterm_list"])} with power_reduction {power_reduction} (Re(nu) = {(Renu - 2*power_reduction):.2f})')
                else: print(f'     Integrating {group_path} with power_reduction {power_reduction} (Re(nu) = {(Renu - 2*power_reduction):.2f})')

                nu_p = Renu - 2*power_reduction + 1j*cp['eta_p']
                
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
                if  p.which in ['local', 'equi', 'ortho', 'primordial']:
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

            save_to_hdf5(output_filename, group_path, data_to_save, metadata)


def merge_fallback_files(output_dir):
    """
    Merge fallback .npy files back into the main HDF5 file.

    This function:
    1. Scans the Cls_fallback directory for fallback .npy files
    2. Loads each file and extracts the data structure
    3. Writes the data into the main Cls.h5 file using save_to_hdf5
    4. Moves successfully merged files to a 'merged' subdirectory

    Parameters:
    -----------
    output_dir : str
        Output directory containing Cls.h5 and Cls_fallback/
    """
    import shutil

    hdf5_file = os.path.join(output_dir, 'Cls.h5')
    fallback_dir = os.path.join(output_dir, 'Cls_fallback')
    merged_dir = os.path.join(fallback_dir, 'merged')

    if not os.path.exists(fallback_dir):
        print(f'No fallback directory found at {fallback_dir}')
        return

    # Find all fallback .npy files
    fallback_files = [f for f in os.listdir(fallback_dir)
                     if f.startswith('fallback_') and f.endswith('.npy')]

    if not fallback_files:
        print(f'No fallback files found in {fallback_dir}')
        return

    print(f'Found {len(fallback_files)} fallback files to merge')
    print('='*70)

    # Create merged directory if it doesn't exist
    os.makedirs(merged_dir, exist_ok=True)

    success_count = 0
    failed_count = 0

    for fallback_file in fallback_files:
        filepath = os.path.join(fallback_dir, fallback_file)
        print(f'\nProcessing: {fallback_file}')

        try:
            # Load the fallback file
            save_dict = np.load(filepath, allow_pickle=True).item()

            group_path = save_dict['group_path']
            data = save_dict['data']
            metadata = save_dict.get('metadata', None)

            print(f'  Group path: {group_path}')
            print(f'  Data keys: {list(data.keys())}')

            # Try to save to HDF5
            success = save_to_hdf5(hdf5_file, group_path, data, metadata)

            if success:
                # Move the file to merged directory
                merged_path = os.path.join(merged_dir, fallback_file)
                shutil.move(filepath, merged_path)
                print(f'  SUCCESS: Merged and moved to {merged_dir}/')
                success_count += 1
            else:
                print(f'  FAILED: Could not merge (HDF5 locking issue persists)')
                failed_count += 1

        except Exception as e:
            print(f'  ERROR: Failed to process {fallback_file}: {e}')
            failed_count += 1

    print('\n' + '='*70)
    print(f'Merge complete: {success_count} successful, {failed_count} failed')
    if failed_count > 0:
        print(f'Failed files remain in {fallback_dir}')


def compute_power_spectrum(p, ell_list, r_list, time_dict, window_args, lterm_list,
                                 W_derivs_list=None):
    """
    Compute power spectrum using equation (36) from the PDF.

    C_ℓ = ∫ dr [-D_r W̃_r C_ℓ^(0,0)(r) + kernel(r) C_ℓ^(-2,0)(r)]

    Uses load_and_compute_all_terms from bispectrum.py to load C^(0,0) and C^(-2,0),
    and fct_of_r_analytical to compute the kernel functions.

    Parameters:
    -----------
    p : parameter object (with p.which = 'cl' to trigger correct loading)
    ell_list : array - multipoles to compute
    r_list : array - radial grid for integration
    time_dict : dict - cosmological functions
    window_args : tuple - window function arguments
    lterm_list : list - linear terms to include
    W_derivs_list : list, optional - precomputed window derivatives

    Returns:
    --------
    dict with keys 'ell' and 'C_ell'
    """
    n_ell = len(ell_list)
    n_r = len(r_list)

    print(f"\nComputing power spectrum from equation (36)")
    print(f"  ell range: {ell_list[0]} to {ell_list[-1]} ({n_ell} values)")
    print(f"  r range: {r_list[0]:.2f} to {r_list[-1]:.2f} ({n_r} points)")

    # ========================================================================
    # 1. Load C^(0,0) and C^(-2,0) using load_and_compute_all_terms
    # ========================================================================
    print("  Loading C^(0,0) and C^(-2,0) from Cls.h5...")

    # Load Cls (will return Cl_array with shape (n_ell, n_chi, 2) for [C^(-2,0), C^(0,0)])
    Cl_array = bispectrum.load_and_compute_all_terms(
        p, ell_list, r_list, time_dict, window_args, lterm_list,
        W_derivs_list=W_derivs_list, tr=None, Pk=None, t_grid=None
    )

    # Extract C^(-2,0) and C^(0,0)
    C_m20 = Cl_array[:, :, 0]  # shape (n_ell, n_r)
    C_00 = Cl_array[:, :, 1]   # shape (n_ell, n_r)

    print(f"    Loaded C^(-2,0) and C^(0,0)")

    # ========================================================================
    # 2. Compute kernel functions using fct_of_r_analytical
    # ========================================================================
    # Get y_list from fct_of_r_analytical
    # This computes all lterm contributions at level 0 (no D_ell operator)
    y_list = fctr_module.fct_of_r_analytical(p, ell_list, r_list, time_dict, window_args,
                                      lterm_list, W_derivs_list=W_derivs_list)

    # ========================================================================
    # 3. Build kernels for equation (36)
    # ========================================================================
    print("  Building kernels for C^(0,0) and C^(-2,0)...")

    # From equation (36):
    # C_ℓ = ∫ dr [-D_r W̃_r C_ℓ^(0,0)(r) + fact(r) C_ℓ^(-2,0)(r)]
    #
    # where the first term involves sum over lterms at level 0 (y_list[lterm][0, ...])
    # and the second term involves the derivative terms

    # Initialize kernels
    fact_00 = np.zeros((n_ell, n_r))  # For C^(0,0)
    fact_m20 = np.zeros((n_ell, n_r))  # For C^(-2,0)

    for i_ell, ell in enumerate(ell_list):
        # Sum contributions from all lterms at level 0 (no D_ell operator applied)
        for lterm in lterm_list:
            if lterm in y_list:
                # For C^(-2,0), we need derivative contributions
                # From eq. (36): includes RSD, doppler, potentials, etc.
                # These come from higher derivative levels or specific lterm combinations
                if lterm in ['rsd', 'doppler', 'pot', 'pot_gr', 'dpot']:
                    # Use level 0 for these correction terms
                    if lterm in ['pot', 'dpot']:
                        fact_m20[i_ell, :] += y_list[lterm][0, 0, i_ell, :] / (2./3./omega_m/H0**2)
                    else:
                        fact_m20[i_ell, :] += y_list[lterm][0, 0, i_ell, :]
                else:
                    # All lterms contribute to the C^(0,0) kernel at level 0
                    # This is the -D_r W̃_r term (times the appropriate fctr for each lterm)
                    # pot_fnl is stored PER UNIT fNL (see fctr.py), so apply the fNL amplitude
                    # here -- same weighting as the bl-side Cl loading in bispectrum.py.
                    lt_weight = p.fnl_local if lterm == 'pot_fnl' else 1.0
                    fact_00[i_ell, :] += lt_weight * y_list[lterm][0, 0, i_ell, :]



    # ========================================================================
    # 4. Integrate using spline + quad for accuracy
    # ========================================================================
    print("  Integrating over r to compute C_ℓ (using spline + quad)...")

    C_ell = np.zeros(n_ell)

    for i_ell in range(n_ell):
        integrand = (fact_00[i_ell, :] * C_00[i_ell, :] + fact_m20[i_ell, :] * C_m20[i_ell, :]) # 

        # Create spline of integrand for accurate integration
        integrand_spline = UnivariateSpline(r_list, integrand, k=5, s=0, ext=0)

        # Integrate using quad for high accuracy
        C_ell[i_ell], err = quad(integrand_spline, r_list[0], r_list[-1], epsrel=1e-4, epsabs=0)

        if i_ell % 10 == 0 or i_ell == n_ell - 1:
            print(f"    ell={ell_list[i_ell]}: C_ell={C_ell[i_ell]:.6e} (integration error: {err:.2e})")

    print("  Done!\n")

    # ========================================================================
    # 5. Save results to file
    # ========================================================================
    # Create filename with lterm list
    output_file = os.path.join(p.output_dir, f'Cl_{p.lterm}.h5')

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"  Saving results to {output_file}...")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('ell', data=ell_list)
        f.create_dataset('C_ell', data=C_ell)
        f.attrs['lterms'] = lterm_list
        f.attrs['n_ell'] = n_ell

    print(f"  Saved power spectrum to {output_file}")

    return {'ell': ell_list, 'C_ell': C_ell}

