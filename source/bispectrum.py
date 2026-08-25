import numpy as np
from numba import njit, prange
import cubature, time, h5py
from scipy.integrate import simpson, quad
from scipy.interpolate import interp1d
from sympy.physics.wigner import wigner_3j
from filelock import FileLock
import os
from scipy.interpolate import UnivariateSpline
from pywigxjpf import wig3jj, wig_temp_init, wig_table_init

from fftlog import *
from mathematica import *
from lincosmo import *
from param_used import *
import general_ps
import fctr

################################################################################ New efficient bispectrum computation


def check_and_compute(cls_file, p, ell_list, chi_list, nm_pairs, which_for_cls,
                      time_dict, window_args, lterm_list, tr=None, Pk=None, t_grid=None):
    # Check if Cls.h5 exists and has required data
    need_to_compute_cls = False
    need_to_compute_am = False
    need_to_compute_il = False
    missing_cls_ells = []
    missing_am_ells = []
    missing_il_ells = []

    if not os.path.exists(cls_file):
        need_to_compute_cls = True
        missing_cls_ells = list(ell_list)
        if p.which in ['F2', 'G2', 'dv2']:
            need_to_compute_am = True
            missing_am_ells = list(ell_list)
            if not p.Newton and p.rad:
                need_to_compute_il = True
                missing_il_ells = list(ell_list)
    else:
        # Check Cls, Am, and Il data separately
        try:
            with h5py.File(cls_file, 'r') as f:
                # Check Cls
                first_group_name = f'n_{nm_pairs[0][0]}_m_{nm_pairs[0][1]}'
                if first_group_name in f:
                    ell_list_file = f[first_group_name]['ell_list'][()]
                    missing_cls_ells = [ell for ell in ell_list if ell not in ell_list_file]
                else:
                    missing_cls_ells = list(ell_list)

                if missing_cls_ells:
                    need_to_compute_cls = True

                # Check Am data separately for F2/G2/dv2
                if p.which in ['F2', 'G2', 'dv2']:
                    if p.which in f:
                        am_ell_list = f[p.which]['ell_list'][()]
                        missing_am_ells = [ell for ell in ell_list if ell not in am_ell_list]

                        # Check Il data for radiation (only if not Newton and rad)
                        if not p.Newton and p.rad:
                            # Check if fm2_rad and fm4_rad datasets exist
                            group = f[p.which]
                            if 'fm2_rad' in group and 'fm4_rad' in group:
                                # Il uses same ell_list as Am
                                missing_il_ells = [ell for ell in ell_list if ell not in am_ell_list]
                            else:
                                missing_il_ells = list(ell_list)
                    else:
                        missing_am_ells = list(ell_list)
                        if not p.Newton and p.rad:
                            missing_il_ells = list(ell_list)

                    if missing_am_ells:
                        need_to_compute_am = True
                    if missing_il_ells:
                        need_to_compute_il = True

        except Exception as e:
            need_to_compute_cls = True
            missing_cls_ells = list(ell_list)
            if p.which in ['F2', 'G2', 'dv2']:
                need_to_compute_am = True
                missing_am_ells = list(ell_list)
                if not p.Newton and p.rad:
                    need_to_compute_il = True
                    missing_il_ells = list(ell_list)

    # Compute only what's missing
    if need_to_compute_cls or need_to_compute_am or need_to_compute_il:
        print(f"\n{'='*70}")
        if need_to_compute_cls:
            print(f"  Missing Cls for ells: {missing_cls_ells}")
        if need_to_compute_am:
            print(f"  Missing Am/f-coefficients for ells: {missing_am_ells}")
        if need_to_compute_il:
            print(f"  Missing Il (radiation) for ells: {missing_il_ells}")
        print(f"{'='*70}\n")

        if tr is None or Pk is None or t_grid is None:
            missing = missing_cls_ells if need_to_compute_cls else (missing_am_ells if need_to_compute_am else missing_il_ells)
            raise ValueError(
                f"Missing data for ells {missing}!\n"
                f"Please run first with mode='cl' to compute power spectra,\n"
                f"or ensure tr/Pk/t_grid parameters are passed to compute_all_bispectra_efficient()."
            )

        # r_list and chi_list are the same in this context
        r_list = chi_list

        # Temporarily set p.which to which_for_cls for Cls computation
        original_which = p.which

        # Step 1: Compute Cls if needed (using FG2)
        if need_to_compute_cls:
            if p.which in ['F2', 'G2', 'dv2']:
                p.which = 'FG2'
            else:
                p.which = which_for_cls

            print(f"  Computing Cls ({p.which})...")
            p.rad = 0
            fctr_dict = fctr.fct_of_r_analytical(p, missing_cls_ells, r_list, time_dict,
                                                  window_args, lterm_list)
            cp_dict = apply_fftlog_dict(tr['k'], Pk, p)
            general_ps.compute_integral_generalized(p, missing_cls_ells, chi_list, r_list, t_grid, cp_dict, fctr_dict, lterm_list)
            print(f"  Cls computed and saved")

        # Step 2: Compute Am/f-coefficient terms if needed (only for F2/G2/dv2)
        if need_to_compute_am:
            p.which = original_which
            print(f"  Computing Am/f-coefficients ({p.which})...")
            p.rad = 0
            fctr_dict = fctr.fct_of_r_analytical(p, missing_am_ells, r_list, time_dict,
                                                  window_args, lterm_list)
            cp_dict = apply_fftlog_dict(tr['k'], Pk, p)
            general_ps.compute_integral_generalized(p, missing_am_ells, chi_list, r_list, t_grid, cp_dict, fctr_dict, lterm_list)
            print(f"  Am/f-coefficients computed and saved")

        # Step 3: Compute Il (radiation) terms if needed (only for F2/G2/dv2 with radiation)
        if need_to_compute_il:
            p.which = original_which
            print(f"  Computing Il (radiation) ({p.which})...")
            p.rad = 1
            fctr_dict = fctr.fct_of_r_analytical(p, missing_il_ells, r_list, time_dict,
                                                  window_args, lterm_list)
            cp_dict = apply_fftlog_dict(tr['k'], tr['dTdk'], p)  # Always use dTdk for radiation
            general_ps.compute_integral_generalized(p, missing_il_ells, chi_list, r_list, t_grid, cp_dict, fctr_dict, lterm_list)
            print(f"  Il (radiation) computed and saved")

        # Restore original which
        p.which = original_which
        print(f"\n  All required data computed and saved to {cls_file}\n")


def get_nm_pairs_for_bispectrum(which, Newton=0):
    """
    Get nm_pairs needed for loading Cls in bispectrum (may include extra components for non-Newton)
    """
    nm_mapping = {
        'FG2': [(-2, 0), (0, 0), (2, 0)],
        'cl': [(-2, 0), (0, 0)],  # For computing C_ℓ from eq. (36): needs C^(-2,0) and C^(0,0)
        'd1v': [(-1, 1)],
        'd2v': [(0, 2)],
        'd3v': [(1, 3)],
        'd0p': [(-2, 0)],
        'd2p': [(0, 2)],
        'dav': [(-2, 0)],
        'd1d': [(1, 1), (-1, 1)] if not Newton else [(1, 1)],  # base + d1v for non-Newton
        'd0d': [(0, 0), (-2, 0)] if not Newton else [(0, 0)],
        'dod': [(0, 0), (-2, 0)] if not Newton else [(0, 0)],
        'local': [(1, 0), (0, 0)],
        'equi':  [(1, 0), (1./3., 0), (2./3., 0)],  # Needs all three: λ=1, λ=1/3, λ=2/3
        'ortho': [(2./3., 0)]
    }
    # Default: use general_ps.get_nm_values
    return nm_mapping.get(which, [(0, 0)])


def load_and_compute_all_terms(p, ell_list, chi_list, time_dict, window_args, lterm_list,
                               W_derivs_list=None, W_lens_derivs_list=None, tr=None, Pk=None, t_grid=None):
    """
    Unified loading function: loads Cls, Am, Il AND computes A0, A2, A4 kernels.
    Everything is evaluated/interpolated onto the same chi grid for efficient integration.

    Parameters:
    -----------
    p : Param object with .which, .Newton, .rad attributes
    ell_list : array of ell values to load/compute
    chi_list : array of chi values (comoving distance grid)
    time_dict : dict with cosmological functions
    window_args : tuple (r0, ddr, normW) for window function
    W_derivs_list : list of arrays, optional
        Precomputed window derivatives. If None, will be computed from window_args.

    Returns:
    --------
    ell_array : (n_ell,) - array of ell values
    Cl_array : (n_ell, n_chi, 4) - Power spectra [chi, Cl_0, Cl_m2, Cl_p2]
    kernels_array : (n_ell, n_chi, 16) - All kernel terms interpolated to chi:
        [0-3]: A0 (up to 4 components)
        [4-5]: A2 (2 components, 0 if not F2/G2/dv2)
        [6]:   A4 (1 component, 0 if not F2/G2/dv2)
        [7-11]: Am (5 terms, 0 if Newton)
        [12-15]: Il (4 terms, 0 if not rad)
    """
    n_ell = len(ell_list)
    n_chi = len(chi_list)

    # ========================================================================
    # 1. Load power spectra from Cls.h5 for all ells
    # ========================================================================
    # print("  Loading power spectra from Cls.h5...")

    cls_file = f"{output_dir}Cls.h5"

    # For F2, G2, dv2: use 'FG2' to get nm_pairs
    # For other quadratic terms: concatenate nm_pairs from both parts
    if p.which in ['F2', 'G2', 'dv2']:
        nm_pairs_list = [get_nm_pairs_for_bispectrum('FG2', p.Newton)]
        which_for_cls_list = ['FG2']
        
        # Single Cl_array for all cases
        Cl_array = np.zeros((n_ell, n_chi, 3))
    elif p.which in ('local', 'equi', 'ortho'):
        nm_pairs_list = [get_nm_pairs_for_bispectrum(p.which)]
        which_for_cls_list = [p.which]
        
        # Single Cl_array for all cases
        Cl_array = np.zeros((n_ell, n_chi, len(nm_pairs_list[0])))
    else:
        # Quadratic term: concatenate nm_pairs from which[:3] and which[3:]
        nm_pairs_list = [get_nm_pairs_for_bispectrum(p.which[:3], p.Newton), get_nm_pairs_for_bispectrum(p.which[3:], p.Newton)]
        which_for_cls_list = [p.which[:3], p.which[3:]]

        # Single Cl_array for all cases
        Cl_array = np.zeros((n_ell, n_chi, 2))


    kernels_array = np.zeros((n_ell, n_chi, 16))
    for idx, (nm_pairs, which_for_cls) in enumerate(zip(nm_pairs_list, which_for_cls_list)):
        if which_for_cls in ['d2p', 'd0p']:
            stuff = 2.0 / (3.0 * omega_m * H0**2)
        else:
            stuff=1

        # Call check and compute function`
        #check_and_compute(cls_file, p, ell_list, chi_list, nm_pairs, which_for_cls,
        #                  time_dict, window_args, lterm_list, tr=tr, Pk=Pk, t_grid=t_grid)

        with h5py.File(cls_file, 'r') as f:

            if idx == 0:
                # Get ell_list and chi_list from file
                # They should be in each group, let's get from first group
                n, m = nm_pairs[0][0], nm_pairs[0][1]

                first_group_name = f'{"primordial_" if p.which in ("local","equi","ortho") else ""}n_{n if isinstance(n, int) else f"{n:.2f}"}_m_{m}'
                if first_group_name not in f:
                    raise ValueError(f"Group {first_group_name} not found in {cls_file}")

                # NEW STRUCTURE: Get chi_list from group, get ell_list by scanning datasets
                first_group = f[first_group_name]
                chi_list_file = first_group['chi_list'][()]

                # Get available ells by scanning the first lterm subgroup (e.g., 'density')
                # Find first lterm subgroup
                lterm_subgroups = [key for key in first_group.keys() if key != 'chi_list']
                if len(lterm_subgroups) == 0:
                    raise ValueError(f"No lterm subgroups found in {first_group_name}")

                first_lterm = lterm_subgroups[0]
                lterm_group = first_group[first_lterm]

                # Extract ell values from dataset names (ell_2, ell_3, ...)
                ell_list_file = []
                for key in lterm_group.keys():
                    if key.startswith('ell_'):
                        ell_val = int(key.split('_')[1])
                        ell_list_file.append(ell_val)
                ell_list_file = np.array(sorted(ell_list_file))

                # Check if ell_list and chi_list match
                ell_match = np.array_equal(ell_list_file, ell_list)
                chi_match = np.array_equal(chi_list_file, chi_list)

                if not ell_match:
                    print(f"  Warning: ell_list mismatch - will extract requested ells from file")
                    print(f"    File has {len(ell_list_file)} ells: [{ell_list_file[0]}, {ell_list_file[-1]}]")
                    print(f"    Requested {len(ell_list)} ells: [{ell_list[0]}, {ell_list[-1]}]")

                if not chi_match:
                    print(f"  Warning: chi_list mismatch - Interpolation needed for chi")
                    print(f"    File has {len(chi_list_file)} chi points: [{chi_list_file[0]:.3f}, {chi_list_file[-1]:.3f}]")
                    print(f"    Requested {len(chi_list)} chi points: [{chi_list[0]:.3f}, {chi_list[-1]:.3f}]")

                # Check if requested ells are available
                missing_ells = [ell for ell in ell_list if ell not in ell_list_file]
                if missing_ells:
                    print(f"  Warning: ells {missing_ells} not found in file")

                # Create mask for valid ells
                valid_mask = np.array([ell in ell_list_file for ell in ell_list])
            
            # Linear bias on the density leg of a quadratic vertex: delta_g = b1*delta.
            b1_cl = b1_dot_cl = None
            # d0dd0d excluded: it is the b2/2*delta^2 vertex, whose legs are the matter
            # density (2011.13660 Table 1, the (delta_T)^2 row carries b20 and no b10).
            if fctr.COMPUTE_B1 and which_for_cls in ['d0d', 'd1d', 'dod'] and p.which != 'd0dd0d' \
                    and 'data' in time_dict and 'b1' in time_dict['data']:
                b1_spline = UnivariateSpline(time_dict['data']['r'], time_dict['data']['b1'], k=5, s=0)
                b1_cl = b1_spline(chi_list_file)
                # prime = d/dtau = -d/dr on the lightcone, as for c2 = -db1/dr + 3*H*b2 in fctr.py
                b1_dot_cl = -b1_spline.derivative(1)(chi_list_file)

            if which_for_cls not in ['d0d', 'd1d', 'dod'] or (p.Newton and which_for_cls not in ['dod']):
                # For each (n,m) pair, sum over lterms
                for cl_idx, (n, m) in enumerate(nm_pairs):
                    group_name = f'{"primordial_" if p.which in ("local","equi","ortho") else ""}n_{n if isinstance(n, int) else f"{n:.2f}"}_m_{m}' 

                    if group_name not in f:
                        print(f"  Warning: Group {group_name} not found, skipping")
                        continue

                    group = f[group_name]

                    # Sum over requested lterms
                    # NEW STRUCTURE: each lterm is a subgroup with ell_X datasets
                    Cl_nm_summed = np.zeros((len(ell_list_file), len(chi_list_file)))

                    for lt in lterm_list:
                        if lt in group:
                            lt_group = group[lt]
                            fnl_weight = p.fnl_local if lt == 'pot_fnl' else 1.0
                            # Load each ell from the subgroup
                            for i_ell, ell in enumerate(ell_list_file):
                                ell_key = f'ell_{ell}'
                                if ell_key in lt_group:
                                    try:
                                        Cl_nm_summed[i_ell, :] += fnl_weight * lt_group[ell_key][()]
                                    except ValueError:
                                        print(ell_key, Cl_nm_summed.shape, (lt_group[ell_key][()]).shape)
                                else:
                                    print(f"  Warning: {ell_key} not found in {group_name}/{lt}")
                        else:
                            print(f"  Warning: Subgroup {lt} not found in {group_name}")

                    # Newton d0d/d1d land here: the single nm pair is the Newtonian density
                    if b1_cl is not None:
                        Cl_nm_summed = Cl_nm_summed * b1_cl[None, :]

                    # Extract requested ells (they match indices now since we built ell_list_file by scanning)
                    valid_ell_indices = [i for i, ell in enumerate(ell_list_file) if ell in ell_list]
                    Cl_subset = Cl_nm_summed[valid_ell_indices, :]/stuff  # shape (n_valid_ells, n_chi_file)

                    # Interpolate all ells at once if needed
                    if not chi_match:
                        # Vectorized interpolation: interpolate all ells simultaneously
                        interp_func = interp1d(chi_list_file, Cl_subset, kind='linear',
                                              axis=1, bounds_error=False, fill_value=0.0)
                        Cl_array[valid_mask, :, cl_idx+idx] = interp_func(chi_list)
                    else:
                        Cl_array[valid_mask, :, cl_idx+idx] = Cl_subset

            else:
                # Special combinations for d0d, d1d, dod with non-Newton
                # print(f"    Applying non-Newton combination for {which_for_cls}")

                # Load all nm_pairs for this part
                Cl_nm_list = []
                for cl_idx, (n, m) in enumerate(nm_pairs):
                    group_name = f'n_{n}_m_{m}'

                    if group_name not in f:
                        print(f"  Warning: Group {group_name} not found, skipping")
                        Cl_nm_list.append(np.zeros((len(ell_list_file), len(chi_list_file))))
                        continue

                    group = f[group_name]

                    # Sum over requested lterms
                    # NEW STRUCTURE: each lterm is a subgroup with ell_X datasets
                    Cl_nm_summed = np.zeros((len(ell_list_file), len(chi_list_file)))

                    for lt in lterm_list:
                        if lt in group:
                            lt_group = group[lt]
                            # pot_fnl Cls are stored per unit fNL (see fctr.py); apply the
                            # fNL amplitude here so the Cls never need recomputing when fNL changes
                            fnl_weight = p.fnl_local if lt == 'pot_fnl' else 1.0
                            # Load each ell from the subgroup
                            for i_ell, ell in enumerate(ell_list_file):
                                ell_key = f'ell_{ell}'
                                if ell_key in lt_group:
                                    Cl_nm_summed[i_ell, :] += fnl_weight * lt_group[ell_key][()]
                                else:
                                    print(f"  Warning: {ell_key} not found in {group_name}/{lt}")
                        else:
                            print(f"  Warning: Subgroup {lt} not found in {group_name}")

                    Cl_nm_list.append(Cl_nm_summed)

                # Apply the combination based on which_for_cls

                # fNL scale-dependent bias: the density Cl factor 3 f H^2 picks up -b_phi/(N D),
                # i.e. C_l^delta = C_l^(0,0) + [3 f H^2 - b_phi/(N D)] C_l^(-2,0). Added to d0d/d1d.
                # b_phi = -2 fNL g_in delta_c (b1-1). Sign must match
                # get_coefficients in fctr.py (Phi = -phi convention): Delta_b1 > 0 for fNL>0, b1>1.
                # d0dd0d excluded for the same reason as b1 above: the b2/2*delta^2 vertex has
                # matter legs, and b_phi is a bias response.
                fnl_corr_on_ra = 0.0
                fnl_dot_corr_on_ra = 0.0
                if fctr.COMPUTE_FNL and p.fnl_local != 0 and p.which != 'd0dd0d' \
                        and 'data' in time_dict and 'b1' in time_dict['data']:
                    deltac = 1.686
                    g_in = time_dict['Da']/time_dict['a'] * 3./5.*(1. + 2./3.*time_dict['fa']/time_dict['Oma'])
                    b1L = UnivariateSpline(time_dict['data']['r'], time_dict['data']['b1']-1., s=0, k=5)(time_dict['ra'])
                    b_phi = - 2.*p.fnl_local*g_in*deltac*b1L
                    N = (2./3./omega_m/H0**2) # N * D
                    fnl_corr_on_ra = -b_phi/ N / time_dict['Da']
                    # dod carries D in its factors, so the object to differentiate is b_phi/N, not
                    # b_phi/(ND). prime = d/dtau = -d/dr, the convention checked on the f-dot bracket.
                    fnl_dot_corr_on_ra = -UnivariateSpline(time_dict['ra'], b_phi, s=0, k=5).derivative(1)(time_dict['ra']) / N

                # b1 multiplies the Newtonian density Cl_nm_list[0] only (see b1_cl above)
                b1_0 = 1.0 if b1_cl is None else b1_cl

                if which_for_cls in ['d0d', 'd1d']:
                    # d0d: Cl = b1*Cl_F2(m=0) + [3*H²*f - b_phi/(N D)] * Cl_F2(m=-2)
                    # d1d: Cl = b1*Cl_d1d + [3*H²*f - b_phi/(N D)] * Cl_d1v
                    factor_on_ra = 3.0 * time_dict['Ha']**2 * time_dict['fa'] + fnl_corr_on_ra
                    factor_spline = UnivariateSpline(time_dict['ra'], factor_on_ra, s=0, k=5)
                    factor = factor_spline(chi_list_file)
                    Cl_combined = b1_0 * Cl_nm_list[0] + factor[None, :] * Cl_nm_list[1]

                elif which_for_cls == 'dod':
                    # dod: Cl = H*D * (f * Cl_F2(m=0) + 3*(f*dotH + H²*(3/2*Om - f)) * Cl_F2(m=-2))

                    # factor1 for Cl_F2(m=0): H*D*f
                    factor1_on_ra = time_dict['Ha'] * time_dict['Da'] * time_dict['fa']

                    # factor2 for Cl_F2(m=-2): H*D * 3*(f*dotH + H²*(3/2*Om - f)) - d(b_phi/N)/dtau
                    factor2_on_ra = time_dict['Ha'] * time_dict['Da'] * 3.0 * (
                        time_dict['fa'] * time_dict['dHa'] +
                        time_dict['Ha']**2 * (1.5 * time_dict['Oma'] - time_dict['fa'])
                    ) - fnl_dot_corr_on_ra

                    # Create splines and evaluate
                    factor1_spline = UnivariateSpline(time_dict['ra'], factor1_on_ra, s=0, k=5)
                    factor2_spline = UnivariateSpline(time_dict['ra'], factor2_on_ra, s=0, k=5)
                    factor1 = factor1_spline(chi_list_file)
                    factor2 = factor2_spline(chi_list_file)

                    # ddot(delta_g) = b1*ddot(delta_N) + b1dot*delta_N + ddot(delta_GR): the
                    # product rule adds b1dot*D to factor1 (D because factor1 = H*D*f carries
                    # one already, and A0 for d1vdod holds only one). factor2 stays unbiased.
                    if b1_cl is not None:
                        Da_cl = UnivariateSpline(time_dict['ra'], time_dict['Da'], s=0, k=5)(chi_list_file)
                        factor1 = b1_cl * factor1 + b1_dot_cl * Da_cl

                    if p.Newton:
                        Cl_combined = factor1[None, :] * Cl_nm_list[0]
                    else:
                        Cl_combined = factor1[None, :] * Cl_nm_list[0] + factor2[None, :] * Cl_nm_list[1]

                # Extract and interpolate the combined result
                # Extract requested ells (they match indices now since we built ell_list_file by scanning)
                valid_ell_indices = [i for i, ell in enumerate(ell_list_file) if ell in ell_list]
                Cl_subset = Cl_combined[valid_ell_indices, :]  # shape (n_valid_ells, n_chi_file)

                if not chi_match:
                    interp_func = interp1d(chi_list_file, Cl_subset, kind='linear',
                                          axis=1, bounds_error=False, fill_value=0.0)
                    Cl_array[valid_mask, :, idx] = interp_func(chi_list)
                else:
                    Cl_array[valid_mask, :, idx] = Cl_subset

    if p.mode=='cl' or p.which in ('local', 'equi', 'ortho'):
        return Cl_array
    else:
        # ========================================================================
        # 2. Compute A0, A2, A4 kernels for all ells directly on chi_list
        # ========================================================================
        # print("  Computing A0, A2, A4 kernels for all ells at once...")
        # Compute for all ells at once (no loop needed!)
        kernels = fctr.get_bispectrum_kernels_analytical(p, ell_list, chi_list, time_dict,
                                                          window_args=window_args, W_derivs_list=W_derivs_list)

        # A0: shape (n_ell, n_components, n_chi)
        # For F2: use analytical A0
        # For G2/dv2: will be filled with f^(0) from file below
        # For quadratic terms: only need A00, extract later
        if p.which == 'F2':
            A0 = kernels['A0']
            n_A0_comp = A0.shape[1]

            # Check if bs_terms are present
            if 'bs_terms' in kernels:
                bs_terms = kernels['bs_terms']  # shape (n_ell, 3, n_chi)
                # print("  Adding b_s corrections to A0, A2, A4 kernels...")

                # A00: base + b_s/6 * f_bs (eq. 20: f^(0)_{0,0} += b_s/6)
                kernels_array[:, :, 0] = A0[:, 0, :] + 1./6.* bs_terms[:, 0, :]

                # A01: base relation -A00/2 + b_s/4 * f_bs (eq. 20: f^(0)_{2,-2} += b_s/4, and f^(0)_{2,-2} = -f^(0)_{0,0}/2 baseline)
                kernels_array[:, :, 1] = -A0[:, 0, :] / 2. + 1./4. * bs_terms[:, 0, :]

                # Copy remaining components (A02, A03 if present)
                for comp in range(2, n_A0_comp):
                    kernels_array[:, :, comp] = A0[:, comp, :]
            else:
                # No bs_terms: use standard formulas
                # Copy A00
                kernels_array[:, :, 0] = A0[:, 0, :]
                # A01 = -A00/2
                kernels_array[:, :, 1] = -kernels_array[:, :, 0] / 2.
                # Copy remaining components (A02, A03 if present)
                for comp in range(2, n_A0_comp):
                    kernels_array[:, :, comp] = A0[:, comp, :]

        # A2 and A4 (only for F2/G2/dv2)
        if p.which in ['F2', 'G2', 'dv2'] and kernels['A2'] is not None and kernels['A4'] is not None:
            A2 = kernels['A2']  # shape (n_ell, 2, n_chi)
            A4 = kernels['A4']  # shape (n_ell, 1, n_chi)

            # Check if bs_terms are present
            if 'bs_terms' in kernels:
                bs_terms = kernels['bs_terms']  # shape (n_ell, 3, n_chi)

                # A2: 2 components with b_s corrections
                # A20: base + (-b_s/2) * D[f_bs] (eq. 20: f^(2)_{0,-2} += -b_s/2)
                kernels_array[:, :, 4] = A2[:, 0, :]  - 1./2. * bs_terms[:, 1, :]
                kernels_array[:, :, 5] = A2[:, 1, :]

                # A4: 1 component with b_s correction
                # A40: base + (b_s/4) * D²[f_bs] (eq. 20: f^(4)_{-2,-2} += b_s/4)
                kernels_array[:, :, 6] = A4[:, 0, :] + 1./4. * bs_terms[:, 2, :]
            else:
                # No bs_terms: use standard formulas
                # A2: 2 components
                for comp in range(2):
                    kernels_array[:, :, 4+comp] = A2[:, comp, :]

                # A4: 1 component
                kernels_array[:, :, 6] = A4[:, 0, :]
        
        # ========================================================================
        # 4. Load f-coefficient terms from HDF5 (F2/G2/dv2 only)
        # ========================================================================
        if p.which in ['F2', 'G2', 'dv2']:
            cls_file = f"{output_dir}Cls.h5"

            # For G2/dv2: Always need f^(0) (goes to A0 slots 0-2)
            # For F2: Only need f^(-2) and f^(-4) if Newton=0 (goes to Am slots 7-11)
            if p.which in ['G2', 'dv2'] or not p.Newton:
                # print("  Loading f-coefficient terms from HDF5...")
                try:
                    with h5py.File(cls_file, 'r') as f:
                        if p.which not in f:
                            raise KeyError(f"Group {p.which} not found")

                        group = f[p.which]
                        chi_list_file = group['chi_list'][()]

                        # Check if chi grids match
                        chi_match = np.array_equal(chi_list_file, chi_list)

                        # NEW STRUCTURE: each multipole is a subgroup with ell_XXX datasets
                        for i, ell in enumerate(ell_list):
                            ell_key = f'ell_{ell}'

                            if p.which in ['G2', 'dv2']:
                                # Load f^(0) directly into A0 slots (0-2)
                                multipole_name = 'f0_newton' if p.Newton else 'f0'

                                if multipole_name not in group:
                                    raise KeyError(f"Multipole {multipole_name} not found in {p.which}")

                                multipole_group = group[multipole_name]

                                if ell_key not in multipole_group:
                                    # Skip missing ells
                                    continue

                                # Load data for this ell: shape is (n_components, n_chi_file)
                                f0_data_ell = multipole_group[ell_key][()]
                                f0_comp1 = f0_data_ell[0, :]  # f_{0,0}
                        
                                if not chi_match:
                                    interp_func = interp1d(chi_list_file, f0_comp1, kind='linear',
                                                          bounds_error=False, fill_value=0.0)
                                    kernels_array[i, :, 0] = interp_func(chi_list)  # A00
                                    kernels_array[i, :, 1] = -kernels_array[i, :, 0] / 2.  # A01 = -f_{0,0}/2

                                else:
                                    kernels_array[i, :, 0] = f0_comp1
                                    kernels_array[i, :, 1] = -f0_comp1 / 2.

                                # If Newton=0, also load f^(-2) into Am slots (7-9, same as F2)
                                if not p.Newton:
                                    f0_comp2 = f0_data_ell[1, :]  # second component

                                    # Load fm2 data for this ell
                                    if 'fm2' not in group:
                                        raise KeyError(f"Multipole fm2 not found in {p.which}")

                                    fm2_group = group['fm2']
                                    if ell_key not in fm2_group:
                                        # Skip missing ells
                                        continue

                                    fm2_data_ell = fm2_group[ell_key][()]
                                    fm2_comp1 = fm2_data_ell[0, :]  # Only one component for G2/dv2

                                    if not chi_match:
                                        interp_func = interp1d(chi_list_file, f0_comp2, kind='linear',
                                                              bounds_error=False, fill_value=0.0)
                                        kernels_array[i, :, 2] = interp_func(chi_list)  # A02

                                        interp_func = interp1d(chi_list_file, fm2_comp1, kind='linear',
                                                              bounds_error=False, fill_value=0.0)
                                        kernels_array[i, :, 7] = interp_func(chi_list)  # Am1
                                        kernels_array[i, :, 8] = -kernels_array[i, :, 7] / 2.  # Am2
                                        # kernels_array[i, :, 9] remains zero (no Am3 for G2/dv2)
                                    else:
                                        kernels_array[i, :, 2] = f0_comp2
                                        kernels_array[i, :, 7] = fm2_comp1
                                        kernels_array[i, :, 8] = -fm2_comp1 / 2.
                                        # kernels_array[i, :, 9] remains zero

                            else:  # F2 with Newton=0
                                # Load f^(-2) into Am slots (7-9)
                                if 'fm2' not in group:
                                    raise KeyError(f"Multipole fm2 not found in {p.which}")

                                fm2_group = group['fm2']
                                if ell_key not in fm2_group:
                                    # Skip missing ells
                                    continue

                                fm2_data_ell = fm2_group[ell_key][()]
                                fm2_comp1 = fm2_data_ell[0, :]
                                fm2_comp2 = fm2_data_ell[1, :]

                                if not chi_match:
                                    interp_func = interp1d(chi_list_file, fm2_comp1, kind='linear',
                                                          bounds_error=False, fill_value=0.0)
                                    kernels_array[i, :, 7] = interp_func(chi_list)  # Am1
                                    kernels_array[i, :, 8] = -kernels_array[i, :, 7] / 2.  # Am2

                                    interp_func = interp1d(chi_list_file, fm2_comp2, kind='linear',
                                                          bounds_error=False, fill_value=0.0)
                                    kernels_array[i, :, 9] = interp_func(chi_list)  # Am3
                                else:
                                    kernels_array[i, :, 7] = fm2_comp1
                                    kernels_array[i, :, 8] = -fm2_comp1 / 2.
                                    kernels_array[i, :, 9] = fm2_comp2

                                # Load f^(-4) into Am slots (10-11)
                                if 'fm4' not in group:
                                    raise KeyError(f"Multipole fm4 not found in {p.which}")

                                fm4_group = group['fm4']
                                if ell_key not in fm4_group:
                                    # Skip missing ells
                                    continue

                                fm4_data_ell = fm4_group[ell_key][()]
                                fm4_comp1 = fm4_data_ell[0, :]

                                if not chi_match:
                                    interp_func = interp1d(chi_list_file, fm4_comp1, kind='linear',
                                                          bounds_error=False, fill_value=0.0)
                                    kernels_array[i, :, 10] = interp_func(chi_list)  # Am4
                                    kernels_array[i, :, 11] = -kernels_array[i, :, 10] / 2.  # Am5
                                else:
                                    kernels_array[i, :, 10] = fm4_comp1
                                    kernels_array[i, :, 11] = -fm4_comp1 / 2.

                except (FileNotFoundError, KeyError) as e:
                    # If we get here, it means the data should have been computed in the section above
                    # but something went wrong. Raise a clear error.
                    raise RuntimeError(
                        f"f-coefficient data not found: {e}\n"
                        f"This should have been computed automatically. Check that tr/Pk/t_grid were provided."
                    ) from e

        # ========================================================================
        # 4b. Load Il terms (radiation) for F2/G2/dv2
        # ========================================================================
        if p.which in ['F2', 'G2', 'dv2'] and not p.Newton and p.rad:
            # print("  Loading Il (radiation) terms from HDF5...")
            try:
                with h5py.File(cls_file, 'r') as f:
                    group = f[p.which]
                    chi_list_file = group['chi_list'][()]

                    # Check if chi grids match
                    chi_match = np.array_equal(chi_list_file, chi_list)

                    # NEW STRUCTURE: radiation terms are in subgroups
                    if 'fm2_rad' not in group:
                        raise KeyError(f"Multipole fm2_rad not found in {p.which}")
                    if 'fm4_rad' not in group:
                        raise KeyError(f"Multipole fm4_rad not found in {p.which}")

                    fm2_rad_group = group['fm2_rad']
                    fm4_rad_group = group['fm4_rad']

                    for i, ell in enumerate(ell_list):
                        ell_key = f'ell_{ell}'

                        if ell_key not in fm2_rad_group or ell_key not in fm4_rad_group:
                            # Skip missing ells
                            continue

                        # Load data for this ell: shape is (n_components, n_chi_file)
                        fm2_rad_ell = fm2_rad_group[ell_key][()]
                        fm4_rad_ell = fm4_rad_group[ell_key][()]

                        # Extract first component
                        fm2_comp1 = fm2_rad_ell[0, :]
                        fm4_comp1 = fm4_rad_ell[0, :]

                        if not chi_match:
                            # Il1 from fm2_rad
                            interp_func = interp1d(chi_list_file, fm2_comp1, kind='linear',
                                                  bounds_error=False, fill_value=0.0)
                            kernels_array[i, :, 12] = interp_func(chi_list)  # Il1
                            kernels_array[i, :, 13] = -kernels_array[i, :, 12] / 2.  # Il2 = -Il1/2

                            # Il3 from fm4_rad
                            interp_func = interp1d(chi_list_file, fm4_comp1, kind='linear',
                                                  bounds_error=False, fill_value=0.0)
                            kernels_array[i, :, 14] = interp_func(chi_list)  # Il3
                            kernels_array[i, :, 15] = -kernels_array[i, :, 14] / 2.  # Il4 = -Il3/2
                        else:
                            kernels_array[i, :, 12] = fm2_comp1
                            kernels_array[i, :, 13] = -fm2_comp1 / 2.
                            kernels_array[i, :, 14] = fm4_comp1
                            kernels_array[i, :, 15] = -fm4_comp1 / 2.

            except (FileNotFoundError, KeyError) as e:
                raise RuntimeError(
                    f"Il (radiation) data not found: {e}\n"
                    f"This should have been computed automatically. Check that tr/Pk/t_grid were provided."
                ) from e

        # ========================================================================
        # 5. Precompute combined coefficients for efficient integration
        # ========================================================================
        # print("  Precomputing angular coefficient combinations...")

        if p.which in ['F2', 'G2', 'dv2']:
            # F2/G2/dv2: 4 angular combinations from the integrand formula
            # coeff_00: for Cl2(0,0)*Cl3(0,0)
            # coeff_m2m2: for Cl2(-2,0)*Cl3(-2,0)
            # coeff_p2m2: for (Cl2(2,0)*Cl3(-2,0) + Cl2(-2,0)*Cl3(2,0))
            # coeff_0m2: for (Cl2(0,0)*Cl3(-2,0) + Cl3(0,0)*Cl2(-2,0))

            coeffs = np.zeros((n_ell, n_chi, 4))

            # Base: A0, A2, A4 contributions (now unified for all cases!)
            coeffs[:, :, 0] = kernels_array[:, :, 0]  # A00 for Cl(0,0)*Cl(0,0)
            coeffs[:, :, 1] = kernels_array[:, :, 3] + kernels_array[:, :, 5] + kernels_array[:, :, 6]  # A03 + A21 + A40 for Cl(-2,0)*Cl(-2,0)
            coeffs[:, :, 2] = kernels_array[:, :, 1]  # A01 for (Cl(2,0)*Cl(-2,0) + Cl(-2,0)*Cl(2,0))
            coeffs[:, :, 3] = kernels_array[:, :, 2] + kernels_array[:, :, 4]  # A02 + A20 for (Cl(0,0)*Cl(-2,0) + Cl(0,0)*Cl(-2,0))

            # Add Am contributions (f^(-2) and f^(-4) terms)
            # Unified formula for F2, G2, dv2!
            # - F2: Am1-Am3 contain f^(-2), Am4-Am5 contain f^(-4)
            # - G2/dv2: Am1-Am2 contain f^(-2), Am3 is zero, Am4-Am5 are zero
            if not p.Newton or (p.Newton and p.which != 'F2'):
                coeffs[:, :, 0] += kernels_array[:, :, 7] + kernels_array[:, :, 10]  # Am1 + Am4 for Cl_0*Cl_0
                coeffs[:, :, 2] += kernels_array[:, :, 8] + kernels_array[:, :, 11]  # Am2 + Am5 for (Cl_p2*Cl_m2 + Cl_m2*Cl_p2)
                coeffs[:, :, 3] += kernels_array[:, :, 9]  # Am3 for (Cl_0*Cl_m2 + Cl_0*Cl_m2)

            # Add Il contributions (radiation terms)
            if not p.Newton and p.rad:
                coeffs[:, :, 0] += kernels_array[:, :, 12] + kernels_array[:, :, 14]  # Il1 + Il3 for Cl_0*Cl_0
                coeffs[:, :, 2] += kernels_array[:, :, 13] + kernels_array[:, :, 15]  # Il2 + Il4 for (Cl_p2*Cl_m2 + Cl_m2*Cl_p2)

        else:
            # Quadratic terms: only need A00
            # Integrand is: A00 * Cl2 * Cl3 (for d2vd2v) or A00 * (Cl2_1*Cl3_2 + Cl3_1*Cl2_2) (for others)
            # Pad to 4 components for compatibility with integration function (last 3 are zeros)
            coeffs = np.zeros((n_ell, n_chi, 1))
            A0 = kernels['A0']
            coeffs[:, :, 0] = A0[:, 0, :]  # A00 (only non-zero component)

        # Return combined coefficients instead of individual kernels
        return Cl_array, coeffs


@njit
def simpson_weights(n):
    """Generate Simpson's rule weights for n points"""
    if n < 3:
        raise ValueError("Need at least 3 points for Simpson's rule")

    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0

    # Handle even n
    if n % 2 == 0:
        w[-2] = 1.0
        w[-1] = 1.0

    return w

@njit(parallel=True)
def compute_bispectrum_quadratic_symmetric(Cl_array, coeffs, chi_list, triplet_list):
    """
    Efficient parallel computation for quadratic terms (d2vd2v, d1vd1d, etc.)

    Parameters:
    -----------
    Cl_array : (n_ell, n_chi, 2) - [Cl_1, Cl_2] for the two parts of quadratic term
    coeffs : (n_ell, n_chi, 4) - Precomputed angular coefficients (only first component used)
    chi_list : (n_chi,) - chi values
    triplet_list : (n_triplets, 3) - indices into ell_array for (ell1, ell2, ell3)

    Returns:
    --------
    results : (n_triplets,) - bispectrum values
    """
    n_triplets = triplet_list.shape[0]
    n_chi = len(chi_list)
    results = np.zeros(n_triplets)

    # Simpson weights
    dchi = chi_list[1] - chi_list[0]
    simp_w = simpson_weights(n_chi)

    for idx in prange(n_triplets):
        i1, i2, i3 = triplet_list[idx, 0], triplet_list[idx, 1], triplet_list[idx, 2]

        # Extract Cls for this triplet
        # d2vd2v case: Cl_array[:, :, 0] == Cl_array[:, :, 1]
        Cl1 = Cl_array[i1, :, 0]
        Cl2 = Cl_array[i2, :, 0]
        Cl3 = Cl_array[i3, :, 0]

        # Extract A00 coefficients
        A00_1 = coeffs[i1, :, 0]
        A00_2 = coeffs[i2, :, 0]
        A00_3 = coeffs[i3, :, 0]

        # Integrand: A00 * Cl * Cl for each permutation
        term1 = A00_1 * Cl2 * Cl3
        term2 = A00_2 * Cl1 * Cl3
        term3 = A00_3 * Cl1 * Cl2

        integrand = term1 + term2 + term3
        # Integrate using Simpson's rule
        integral = np.sum(integrand * simp_w) * dchi / 3.0
        # integral = simpson(integrand, x=chi_list)
        
#        spline = UnivariateSpline(chi_list, integrand, k=5, s=0)
#        integral = quad(spline, chi_list[0], chi_list[-1])[0]
        
        results[idx] = integral

    return 2.*results


@njit(parallel=True)
def compute_bispectrum_quadratic_dav(Cl_array, coeffs, chi_list, triplet_list, Al1l2l3):
    """
    Efficient parallel computation for quadratic terms (d2vd2v, d1vd1d, etc.)

    Parameters:
    -----------
    Cl_array : (n_ell, n_chi, 2) - [Cl_1, Cl_2] for the two parts of quadratic term
    coeffs : (n_ell, n_chi, 4) - Precomputed angular coefficients (only first component used)
    chi_list : (n_chi,) - chi values
    triplet_list : (n_triplets, 3) - indices into ell_array for (ell1, ell2, ell3)

    Returns:
    --------
    results : (n_triplets,) - bispectrum values
    """
    n_triplets = triplet_list.shape[0]
    n_chi = len(chi_list)
    results = np.zeros(n_triplets)

    # Simpson weights
    dchi = chi_list[1] - chi_list[0]
    simp_w = simpson_weights(n_chi)

    for idx in prange(n_triplets):
        i1, i2, i3 = triplet_list[idx, 0], triplet_list[idx, 1], triplet_list[idx, 2]

        # Other quadratic terms: need both Cl components
        Cl1_1, Cl1_2 = Cl_array[i1, :, 0], Cl_array[i1, :, 1]
        Cl2_1, Cl2_2 = Cl_array[i2, :, 0], Cl_array[i2, :, 1]
        Cl3_1, Cl3_2 = Cl_array[i3, :, 0], Cl_array[i3, :, 1]

        # Extract A00 coefficients
        A00_1 = Al1l2l3[idx, 0]*coeffs[i1, :, 0]
        A00_2 = Al1l2l3[idx, 1]*coeffs[i2, :, 0]
        A00_3 = Al1l2l3[idx, 2]*coeffs[i3, :, 0]

        # Integrand: A00 * (Cl_1 * Cl_2 + Cl_2 * Cl_1) for each permutation
        term1 = A00_1 * (Cl2_1 * Cl3_2 + Cl2_2 * Cl3_1)
        term2 = A00_2 * (Cl1_1 * Cl3_2 + Cl1_2 * Cl3_1)
        term3 = A00_3 * (Cl1_1 * Cl2_2 + Cl1_2 * Cl2_1)

        integrand = term1 + term2 + term3

        # Integrate using Simpson's rule
        integral = np.sum(integrand * simp_w) * dchi / 3.0

        results[idx] = integral

    return results


@njit(parallel=True)
def compute_bispectrum_quadratic(Cl_array, coeffs, chi_list, triplet_list):
    """
    Efficient parallel computation for quadratic terms (d2vd2v, d1vd1d, etc.)

    Parameters:
    -----------
    Cl_array : (n_ell, n_chi, 2) - [Cl_1, Cl_2] for the two parts of quadratic term
    coeffs : (n_ell, n_chi, 4) - Precomputed angular coefficients (only first component used)
    chi_list : (n_chi,) - chi values
    triplet_list : (n_triplets, 3) - indices into ell_array for (ell1, ell2, ell3)

    Returns:
    --------
    results : (n_triplets,) - bispectrum values
    """
    n_triplets = triplet_list.shape[0]
    n_chi = len(chi_list)
    results = np.zeros(n_triplets)

    # Simpson weights
    dchi = chi_list[1] - chi_list[0]
    simp_w = simpson_weights(n_chi)

    for idx in prange(n_triplets):
        i1, i2, i3 = triplet_list[idx, 0], triplet_list[idx, 1], triplet_list[idx, 2]

        # Other quadratic terms: need both Cl components
        Cl1_1, Cl1_2 = Cl_array[i1, :, 0], Cl_array[i1, :, 1]
        Cl2_1, Cl2_2 = Cl_array[i2, :, 0], Cl_array[i2, :, 1]
        Cl3_1, Cl3_2 = Cl_array[i3, :, 0], Cl_array[i3, :, 1]

        # Extract A00 coefficients
        A00_1 = coeffs[i1, :, 0]
        A00_2 = coeffs[i2, :, 0]
        A00_3 = coeffs[i3, :, 0]

        # Integrand: A00 * (Cl_1 * Cl_2 + Cl_2 * Cl_1) for each permutation
        term1 = A00_1 * (Cl2_1 * Cl3_2 + Cl2_2 * Cl3_1)
        term2 = A00_2 * (Cl1_1 * Cl3_2 + Cl1_2 * Cl3_1)
        term3 = A00_3 * (Cl1_1 * Cl2_2 + Cl1_2 * Cl2_1)

        integrand = term1 + term2 + term3

        # Integrate using Simpson's rule
        integral = np.sum(integrand * simp_w) * dchi / 3.0
        # integral = simpson(integrand, x=chi_list)
#        spline = UnivariateSpline(chi_list, integrand, k=5, s=0)
#        integral = quad(spline, chi_list[0], chi_list[-1])[0]

        results[idx] = integral

    return results 


@njit(parallel=True)
def compute_bispectrum_parallel_efficient(Cl_array, coeffs, chi_list, triplet_list):
    """
    Efficient parallel computation using precomputed angular coefficients.

    Parameters:
    -----------
    Cl_array : (n_ell, n_chi, 3) - [Cl_m2, Cl_0, Cl_p2] (indices 0,1,2 for nm_pairs [(-2,0), (0,0), (2,0)])
    coeffs : (n_ell, n_chi, 4) - Precomputed angular coefficients [c_00, c_m2m2, c_p2m2, c_0m2]
    chi_list : (n_chi,) - chi values
    triplet_list : (n_triplets, 3) - indices into ell_array for (ell1, ell2, ell3)

    Returns:
    --------
    results : (n_triplets,) - bispectrum values
    """
    n_triplets = triplet_list.shape[0]
    n_chi = len(chi_list)
    results = np.zeros(n_triplets)

    # Simpson weights
    dchi = chi_list[1] - chi_list[0]  # Assume uniform spacing
    simp_w = simpson_weights(n_chi)

    for idx in prange(n_triplets):
        i1, i2, i3 = triplet_list[idx, 0], triplet_list[idx, 1], triplet_list[idx, 2]

        # Extract Cls for this triplet (nm_pairs order: [(-2,0), (0,0), (2,0)])
        Cl1_m2, Cl1_0, Cl1_p2 = Cl_array[i1, :, 0], Cl_array[i1, :, 1], Cl_array[i1, :, 2]
        Cl2_m2, Cl2_0, Cl2_p2 = Cl_array[i2, :, 0], Cl_array[i2, :, 1], Cl_array[i2, :, 2]
        Cl3_m2, Cl3_0, Cl3_p2 = Cl_array[i3, :, 0], Cl_array[i3, :, 1], Cl_array[i3, :, 2]

        # Extract precomputed coefficients (4 components now)
        c1_00, c1_m2m2, c1_p2m2, c1_0m2 = coeffs[i1, :, 0], coeffs[i1, :, 1], coeffs[i1, :, 2], coeffs[i1, :, 3]
        c2_00, c2_m2m2, c2_p2m2, c2_0m2 = coeffs[i2, :, 0], coeffs[i2, :, 1], coeffs[i2, :, 2], coeffs[i2, :, 3]
        c3_00, c3_m2m2, c3_p2m2, c3_0m2 = coeffs[i3, :, 0], coeffs[i3, :, 1], coeffs[i3, :, 2], coeffs[i3, :, 3]

        # Compute integrand for all three permutations (vectorized!)
        # Using formula: A00*Cl_0*Cl_0 + (A03+A21+A40)*Cl_m2*Cl_m2 + A01*(Cl_p2*Cl_m2+Cl_m2*Cl_p2) + (A02+A20)*(Cl_0*Cl_m2+Cl_0*Cl_m2)

        # Permutation 1: kernels from ell1, Cls from ell2 and ell3
        term1 = c1_00 * Cl2_0 * Cl3_0 \
              + c1_m2m2 * Cl2_m2 * Cl3_m2 \
              + c1_p2m2 * (Cl2_p2 * Cl3_m2 + Cl2_m2 * Cl3_p2) \
              + c1_0m2 * (Cl2_0 * Cl3_m2 + Cl3_0 * Cl2_m2)

        # Permutation 2: kernels from ell2, Cls from ell1 and ell3
        term2 = c2_00 * Cl1_0 * Cl3_0 \
              + c2_m2m2 * Cl1_m2 * Cl3_m2 \
              + c2_p2m2 * (Cl1_p2 * Cl3_m2 + Cl1_m2 * Cl3_p2) \
              + c2_0m2 * (Cl1_0 * Cl3_m2 + Cl3_0 * Cl1_m2)

        # Permutation 3: kernels from ell3, Cls from ell1 and ell2
        term3 = c3_00 * Cl1_0 * Cl2_0 \
              + c3_m2m2 * Cl1_m2 * Cl2_m2 \
              + c3_p2m2 * (Cl1_p2 * Cl2_m2 + Cl1_m2 * Cl2_p2) \
              + c3_0m2 * (Cl1_0 * Cl2_m2 + Cl2_0 * Cl1_m2)

        integrand = term1 + term2 + term3

        # Integrate using Simpson's rule
        integral = np.sum(integrand * simp_w) * dchi / 3.0

#        spline = UnivariateSpline(chi_list, integrand, k=5, s=1e-3)
#        integral = quad(spline, chi_list[0], chi_list[-1])[0]

        results[idx] = integral

    return 2.*results


def _init_wigner_worker(max_ell):
    """Initialize pywigxjpf tables once per worker process.

    Takes max(ell) and applies the 2*j conversion internally: wig3jj is always called
    with 2*ell arguments, so the largest two_j is 2*max(ell). BOTH the factorial table
    and the temp array must be sized for that. Undersizing the temp array does NOT
    raise -- pywigxjpf prints "More iterations than allocated" and returns a garbage
    value -- so getting this wrong silently corrupts Al123 (and hence davd1v).

    The temp array must cover the full triangle slack (j1+j2-j3), which reaches
    ~max(ell) for configurations like 'squ2' where ell1 varies against a fixed large
    ell2=ell3 (the case that exposed this).
    """
    two_j_max = 2 * max_ell
    try:
        # Factorials table (shared, only needs to be done once per process)
        wig_table_init(two_j_max, 3)
        # Temp array for this worker (same sizing -- must cover the largest two_j)
        wig_temp_init(two_j_max)
    except Exception as e:
        print(f'Warning: pywigxjpf initialization failed: {e}')


def _compute_Al123_single(ell1, ell2, ell3, wigner_000):
    """
    Compute Al123 for a single (ell1, ell2, ell3) triplet using precomputed wigner_000.

    Al123 = (wigner_3j(l1,l2,l3,0,1,-1) + wigner_3j(l1,l2,l3,0,-1,1)) / wigner_3j(l1,l2,l3,0,0,0)
    """
    if ell1 == ell2 == ell3:
        return -1.0

    try:
        # Use pywigxjpf (2*j convention, m values are also doubled)
        w_01m1 = wig3jj(2*ell1, 2*ell2, 2*ell3, 0, 2, -2)
        w_0m11 = wig3jj(2*ell1, 2*ell2, 2*ell3, 0, -2, 2)
    except Exception:
        # Fallback to sympy
        w_01m1 = float(wigner_3j(ell1, ell2, ell3, 0, 1, -1))
        w_0m11 = float(wigner_3j(ell1, ell2, ell3, 0, -1, 1))

    return (w_01m1 + w_0m11) / wigner_000


def _compute_wigner_wrapper(args):
    """Helper function for parallel Wigner 3j and Al123 computation (must be at module level for pickling)"""
    triplet, ell_list = args
    i1, i2, i3 = triplet
    ell1, ell2, ell3 = int(ell_list[i1]), int(ell_list[i2]), int(ell_list[i3])

    # Use fast pywigxjpf if available, otherwise fall back to sympy
    try:
        # wig3jj expects 2*j values (uses half-integer convention)
        # wig3jj(two_j1, two_j2, two_j3, two_m1, two_m2, two_m3)
        # Note: tables are already initialized by _init_wigner_worker
        wigner_val = wig3jj(2*ell1, 2*ell2, 2*ell3, 0, 0, 0)
    except (ImportError, Exception) as e:
        # Fallback to sympy
        print(f'pywigxjpf failed ({e}), try sympy (slower)')
        wigner_val = float(wigner_3j(ell1, ell2, ell3, 0, 0, 0))

    if wigner_val == 0:
        return None

    # Compute Al123 for all 3 permutations (reusing wigner_000 as denominator)
    A1 = _compute_Al123_single(ell1, ell2, ell3, wigner_val) * np.sqrt(ell2*(ell2+1.) * ell3*(ell3+1.))
    A2 = _compute_Al123_single(ell2, ell1, ell3, wigner_val) * np.sqrt(ell1*(ell1+1.) * ell3*(ell3+1.))
    A3 = _compute_Al123_single(ell3, ell2, ell1, wigner_val) * np.sqrt(ell2*(ell2+1.) * ell1*(ell1+1.))

    return (triplet, wigner_val, A1, A2, A3)


def ell_configurations(p, ell_list):
    """
    Generate triplet configurations, compute Wigner 3j symbols and Al123 coefficients.

    This function computes purely geometric quantities that are independent of cosmology.

    Parameters:
    -----------
    p : parameter object
    ell_list : array of multipoles

    Returns:
    --------
    triplets : array of shape (n_triplets, 3) - indices into ell_list
    wigner_values : array of shape (n_triplets,)
    config_name : string
    Al1l2l3_values : array of shape (n_triplets, 3) - Al123 coefficients for davd1v term
    """
    # Create ell to index mapping
    ell_to_idx = {ell: i for i, ell in enumerate(ell_list)}

    # ========================================================================
    # Cache filename based on ell_list and configuration
    # ========================================================================
    config = p.configuration

    # Create descriptive filename from ell_list properties
    ell_min = int(ell_list[0])
    ell_max = int(ell_list[-1])
    n_ells = len(ell_list)

    # Detect spacing type (linear, log, custom)
    if n_ells > 1:
        diffs = np.diff(ell_list)
        if np.allclose(diffs, diffs[0], rtol=0.01):
            spacing = f'lin{int(diffs[0])}'
        else:
            spacing = 'custom'
    else:
        spacing = 'single'

    ell_descriptor = f'ell{ell_min}to{ell_max}_n{n_ells}_{spacing}'

    if config == 'equi':
        cache_file = f'{p.output_dir}triplets_cache_equi_{ell_descriptor}.npz'
        config_name = 'equilateral'
    elif config == 'squ':
        cache_file = f'{p.output_dir}triplets_cache_squ_ell{p.ell}_{ell_descriptor}.npz'
        config_name = f'squeezed_ell{p.ell}'
    elif config == 'squ2':
        cache_file = f'{p.output_dir}triplets_cache_squ2_ell{p.ellmax}_{ell_descriptor}.npz'
        config_name = f'squeezed2_ell{p.ellmax}'
    elif config == 'folded':
        cache_file = f'{p.output_dir}triplets_cache_folded_ell{p.ellmax}_{ell_descriptor}.npz'
        config_name = f'folded_ell{p.ellmax}'
    else:
        cache_file = f'{p.output_dir}triplets_cache_all_{ell_descriptor}.npz'
        config_name = ''

    # Try to load from cache
    if os.path.exists(cache_file):
        # print(f"Loading triplets from cache: {cache_file}")
        data = np.load(cache_file)
        triplets = data['triplets']
        wigner_values = data['wigner_values']
        # Load Al1l2l3 if available in cache
        Al1l2l3_values = data['Al1l2l3_values'] if 'Al1l2l3_values' in data else None
        # print(f"  Loaded {len(triplets)} triplets from cache")
        # if Al1l2l3_values is not None:
        #     print(f"  Loaded Al1l2l3 coefficients for {len(triplets)} triplets")

        return triplets, wigner_values, config_name, Al1l2l3_values

    # ========================================================================
    # 3. Generate triplet list and compute Wigner 3j symbols
    # ========================================================================
    # print(f"Generating triplets for {len(ell_list)} ells (configuration: {config})...")
    triplets = []
    wigner_values = []

    # Configuration-based triplet filtering
    Al1l2l3_values = []

    # Initialize Wigner symbol library for Al123 computation
    max_ell = int(np.max(ell_list))
    _init_wigner_worker(max_ell)

    if config == 'equi':
        # Equilateral: ell1 = ell2 = ell3
        for i, ell in enumerate(ell_list):
            ell_int = int(ell)
            wigner_test = wig3jj(2*ell_int, 2*ell_int, 2*ell_int, 0, 0, 0)  # fast pywigxjpf (tables set up above)
            if wigner_test != 0:
                triplets.append([i, i, i])
                wigner_values.append(wigner_test)
                # For equilateral, Al123 = -1.0 (special case)
                A_val = -1.0 * np.sqrt(ell_int*(ell_int+1.) * ell_int*(ell_int+1.))
                Al1l2l3_values.append([A_val, A_val, A_val])
        config_name = 'equilateral'

    elif config == 'squ':
        # Squeezed: ell1 fixed, ell2 = ell3 varying
        ell1_fixed = p.ell
        if ell1_fixed not in ell_list:
            print(f"Warning: ell1={ell1_fixed} not in ell_list, no triplets generated")
        else:
            i1 = ell_to_idx[ell1_fixed]
            ell1_int = int(ell1_fixed)
            for i23, ell23 in enumerate(ell_list):
                ell23_int = int(ell23)
                # Triangle inequality
                if ell23 < abs(ell1_fixed - ell23) or ell23 > ell1_fixed + ell23:
                    continue
                wigner_test = wig3jj(2*ell1_int, 2*ell23_int, 2*ell23_int, 0, 0, 0)  # fast pywigxjpf
                if wigner_test != 0:
                    triplets.append([i1, i23, i23])
                    wigner_values.append(wigner_test)
                    # Compute Al123 for all 3 permutations
                    A1 = _compute_Al123_single(ell1_int, ell23_int, ell23_int, wigner_test) * np.sqrt(ell23_int*(ell23_int+1.) * ell23_int*(ell23_int+1.))
                    A2 = _compute_Al123_single(ell23_int, ell1_int, ell23_int, wigner_test) * np.sqrt(ell1_int*(ell1_int+1.) * ell23_int*(ell23_int+1.))
                    A3 = _compute_Al123_single(ell23_int, ell23_int, ell1_int, wigner_test) * np.sqrt(ell23_int*(ell23_int+1.) * ell1_int*(ell1_int+1.))
                    Al1l2l3_values.append([A1, A2, A3])
        config_name = f'squeezed_ell{ell1_fixed}'

    elif config == 'squ2':
        # Squeezed-2: ell2 = ell3 = ellmax fixed, ell1 (the long mode) varying.
        # This is the configuration that actually probes k_L -> 0, i.e. the limit
        # the squeezed-limit consistency relation is about.
        ell23_fixed = p.ellmax
        if ell23_fixed not in ell_list:
            print(f"Warning: ellmax={ell23_fixed} not in ell_list, no triplets generated")
        else:
            i23 = ell_to_idx[ell23_fixed]
            ell23_int = int(ell23_fixed)
            for i1, ell1 in enumerate(ell_list):
                ell1_int = int(ell1)
                # Triangle inequality: |ell2-ell3| <= ell1 <= ell2+ell3  ->  0 <= ell1 <= 2*ell23
                if ell1_int > 2*ell23_int:
                    continue
                wigner_test = wig3jj(2*ell1_int, 2*ell23_int, 2*ell23_int, 0, 0, 0)  # fast pywigxjpf
                if wigner_test != 0:
                    triplets.append([i1, i23, i23])
                    wigner_values.append(wigner_test)
                    # Same permutation structure as 'squ': triplet is (ell1, ell23, ell23)
                    A1 = _compute_Al123_single(ell1_int, ell23_int, ell23_int, wigner_test) * np.sqrt(ell23_int*(ell23_int+1.) * ell23_int*(ell23_int+1.))
                    A2 = _compute_Al123_single(ell23_int, ell1_int, ell23_int, wigner_test) * np.sqrt(ell1_int*(ell1_int+1.) * ell23_int*(ell23_int+1.))
                    A3 = _compute_Al123_single(ell23_int, ell23_int, ell1_int, wigner_test) * np.sqrt(ell23_int*(ell23_int+1.) * ell1_int*(ell1_int+1.))
                    Al1l2l3_values.append([A1, A2, A3])
        config_name = f'squeezed2_ell{ell23_fixed}'

    elif config == 'folded':
        # Folded: ell1 = ellmax, ell2 = ell3 varying
        ell1_fixed = p.ellmax
        if ell1_fixed not in ell_list:
            print(f"Warning: ellmax={ell1_fixed} not in ell_list, no triplets generated")
        else:
            i1 = ell_to_idx[ell1_fixed]
            ell1_int = int(ell1_fixed)
            for i23, ell23 in enumerate(ell_list):
                ell23_int = int(ell23)
                # Triangle inequality
                if ell23 < abs(ell1_fixed - ell23) or ell23 > ell1_fixed + ell23:
                    continue
                wigner_test = wig3jj(2*ell1_int, 2*ell23_int, 2*ell23_int, 0, 0, 0)  # fast pywigxjpf
                if wigner_test != 0:
                    triplets.append([i1, i23, i23])
                    wigner_values.append(wigner_test)
                    # Compute Al123 for all 3 permutations
                    A1 = _compute_Al123_single(ell1_int, ell23_int, ell23_int, wigner_test) * np.sqrt(ell23_int*(ell23_int+1.) * ell23_int*(ell23_int+1.))
                    A2 = _compute_Al123_single(ell23_int, ell1_int, ell23_int, wigner_test) * np.sqrt(ell1_int*(ell1_int+1.) * ell23_int*(ell23_int+1.))
                    A3 = _compute_Al123_single(ell23_int, ell23_int, ell1_int, wigner_test) * np.sqrt(ell23_int*(ell23_int+1.) * ell1_int*(ell1_int+1.))
                    Al1l2l3_values.append([A1, A2, A3])
        config_name = f'folded_ell{ell1_fixed}'

    else:
        # All configurations: generate all valid triplets
        # First, generate candidate triplets (fast)
        candidates = []
        for i1 in range(len(ell_list)):
            if i1%50==0: print(f'     {i1+1}/{len(ell_list)}')
            for i2 in range(i1, len(ell_list)):
                for i3 in range(i2, len(ell_list)):
                    # Triangle inequality: |ell1 - ell2| <= ell3 <= ell1 + ell2
                    if ell_list[i3] < abs(ell_list[i1] - ell_list[i2]) or ell_list[i3] > ell_list[i1] + ell_list[i2]:
                        continue

                    # Parity rule: ell1 + ell2 + ell3 must be even (major optimization!)
                    if (ell_list[i1] + ell_list[i2] + ell_list[i3]) % 2 != 0:
                        continue

                    candidates.append((i1, i2, i3))

        # print(f'  Generated {len(candidates)} candidate triplets, computing Wigner 3j and Al123...')

        # Compute Wigner 3j and Al123 in parallel
        from multiprocessing import Pool, cpu_count
        n_cores = min(cpu_count(), 16)

        # print(f'  Using {n_cores} cores for parallel computation...')
        # Prepare arguments: each candidate needs access to ell_list
        args_list = [(cand, ell_list) for cand in candidates]

        # Pass max(ell); _init_wigner_worker applies the 2*j conversion itself
        max_ell = int(max(ell_list))

        # Use initializer to set up pywigxjpf once per worker (huge speedup!)
        from functools import partial
        initializer = partial(_init_wigner_worker, max_ell)

        with Pool(n_cores, initializer=initializer) as pool:
            # Use imap to get results as they complete (allows progress tracking)
            results_iter = pool.imap(_compute_wigner_wrapper, args_list, chunksize=1000)

            # Process results and show progress
            total = len(candidates)
            processed = 0
            for result in results_iter:
                processed += 1
                if processed % 50000 == 0 or processed == total:
                    print(f'    Progress: {processed}/{total} ({100*processed/total:.1f}%)')

                if result is not None:
                    triplet, wigner_val, A1, A2, A3 = result
                    triplets.append(list(triplet))
                    wigner_values.append(wigner_val)
                    Al1l2l3_values.append([A1, A2, A3])

        # print(f'  Kept {len(triplets)} triplets with non-zero Wigner 3j')

        config_name = ''

    # Convert to arrays
    triplets = np.array(triplets, dtype=np.int64)
    wigner_values = np.array(wigner_values)
    Al1l2l3_values = np.array(Al1l2l3_values)

    # Save to cache for future use
    # print(f"Saving triplets to cache: {cache_file}")
    np.savez(cache_file, triplets=triplets, wigner_values=wigner_values, Al1l2l3_values=Al1l2l3_values)
    print(f"  Saved {len(triplets)} triplets with Al1l2l3 to cache")

    return triplets, wigner_values, config_name, Al1l2l3_values


def compute_variance_for_triplets(p, ell_list, triplet_array):
    """
    Compute variance V_ell1ell2ell3 = C_ell1 * C_ell2 * C_ell3 for all triplets.

    This is cosmology-dependent and should be computed separately from geometric quantities.

    Parameters:
    -----------
    p : parameter object
    ell_list : array of multipoles
    triplet_array : array of shape (n_triplets, 3) - indices into ell_list

    Returns:
    --------
    variance_values : array of shape (n_triplets,) if C_ell file exists, else None
    """
    cl_file = os.path.join(p.output_dir, f'Cl_{p.lterm}.h5')
    if not os.path.exists(cl_file):
        print(f"    C_ell file not found at {cl_file}, skipping variance computation")
        return None

    # print(f"  Loading power spectrum from {cl_file}...")
    with h5py.File(cl_file, 'r') as f:
        ell_file = f['ell'][:]
        C_ell_file = f['C_ell'][:]

    # Interpolate to match ell_list if needed
    if np.array_equal(ell_file, ell_list):
        C_ell = C_ell_file
    else:
        raise ValueError(f"ell_file must match ell_list")

    # print(f"  Loaded C_ell for {len(C_ell)} multipoles")

    # Compute variance
    # print("  Computing variance V_ell1ell2ell3 = C_ell1 * C_ell2 * C_ell3...")
    variance_values = np.zeros(len(triplet_array))
    for i, (i1, i2, i3) in enumerate(triplet_array):
        variance_values[i] = C_ell[i1] * C_ell[i2] * C_ell[i3]
    # print(f"  Computed variance for {len(triplet_array)} triplets")

    return variance_values


def get_all_primordial_shapes(p, ell_list, chi_list, time_dict, window_args, lterm_list,
                               W_derivs_list=None, W_lens_derivs_list=None, tr=None, Pk=None, t_grid=None):
    """
    Compute and combine all primordial shapes: local, equilateral, orthogonal.

    The shapes are combined as follows:
    - local: computed directly with which='local'
    - B_1_13_23: computed with which='equi'
    - B_23_23_23: computed with which='ortho'
    - equilateral = -3*local + 6*B_1_13_23 - 12*B_23_23_23
    - orthogonal = 3*equilateral - 12*B_23_23_23

    Parameters:
    -----------
    p : Param object with mode='primordial'
    ell_list : list - ell values
    chi_list : array - chi values for integration
    time_dict : dict - cosmological functions
    window_args : tuple - (r0, ddr, normW)
    lterm_list : list - lterm values to sum over
    W_derivs_list : optional precomputed window derivatives
    tr : dict - transfer functions
    Pk : array - power spectrum
    t_grid : array - t values for FFTLog
    """

    # print(f"="*70)
    # print(f"Computing all primordial shapes: local, equilateral, orthogonal")
    # print(f"="*70)

    # Store original which
    original_which = p.which

    # Determine config_name and file_path (shared file for all primordial shapes)
    config_name = 'equilateral' if p.configuration == 'equi' else \
                 (f'squeezed_ell{p.ell}' if p.configuration == 'squ' else \
                  f'squeezed2_ell{p.ellmax}' if p.configuration == 'squ2' else \
                  f'folded_ell{p.ellmax}' if p.configuration == 'folded' else 'all')
    file_path = f"{p.output_dir}bl_{p.lterm}.h5"

    # Step 1: Compute local bispectrum
    # print(f"\n{'='*70}")
    # print(f"  Step 1/3: Computing LOCAL bispectrum")
    # print(f"{'='*70}\n")
    p.which = 'local'
    compute_all_bispectra_efficient(p, ell_list, chi_list, time_dict, window_args, lterm_list,
                                     W_derivs_list=W_derivs_list, W_lens_derivs_list=W_lens_derivs_list, tr=tr, Pk=Pk, t_grid=t_grid)

    # Step 2: Compute B_1_13_23 (using which='equi')
    # print(f"\n{'='*70}")
    # print(f"  Step 2/3: Computing B_1_13_23 (intermediate for equilateral)")
    # print(f"{'='*70}\n")
    p.which = 'equi'
    compute_all_bispectra_efficient(p, ell_list, chi_list, time_dict, window_args, lterm_list,
                                     W_derivs_list=W_derivs_list, W_lens_derivs_list=W_lens_derivs_list, tr=tr, Pk=Pk, t_grid=t_grid)

    # Step 3: Compute B_23_23_23 (using which='ortho')
    # print(f"\n{'='*70}")
    # print(f"  Step 3/3: Computing B_23_23_23 (intermediate for equilateral & orthogonal)")
    # print(f"{'='*70}\n")
    p.which = 'ortho'
    compute_all_bispectra_efficient(p, ell_list, chi_list, time_dict, window_args, lterm_list,
                                     W_derivs_list=W_derivs_list, W_lens_derivs_list=W_lens_derivs_list, tr=tr, Pk=Pk, t_grid=t_grid)

    # Step 4: Load results and combine to get final equilateral and orthogonal
    # print(f"\n{'='*70}")
    # print(f"  Combining results to get final shapes...")
    # print(f"{'='*70}\n")

    with h5py.File(file_path, 'r') as f:
        grp = f[config_name]
        B_local = grp['bl_local'][:]
        B_1_13_23 = grp['bl_equi'][:]
        B_23_23_23 = grp['bl_ortho'][:]

    B_equilateral = -3.*B_local + 6.*B_1_13_23 - 12.*B_23_23_23
    B_orthogonal = 3.*B_equilateral - 12.*B_23_23_23

    # Step 5: Save combined results to the same file
    # print(f"  Saving combined primordial shapes to {file_path}...")

    with h5py.File(file_path, "a") as f:
        grp = f[config_name]

        # Save equilateral (combined)
        if 'bl_equilateral' in grp:
            del grp['bl_equilateral']
        grp.create_dataset('bl_equilateral', data=B_equilateral)

        # Save orthogonal (combined)
        if 'bl_orthogonal' in grp:
            del grp['bl_orthogonal']
        grp.create_dataset('bl_orthogonal', data=B_orthogonal)

        f.flush()

    # Restore original which
    p.which = original_which

#    print(f"\n{'='*70}")
#    print(f"All primordial shapes computed and saved successfully!")
#    print(f"  File: {file_path}")
#    print(f"  Datasets: bl_local, bl_equi, bl_ortho, bl_equilateral, bl_orthogonal")
#    print(f"{'='*70}\n")


def compute_all_bispectra_efficient(p, ell_list, chi_list, time_dict, window_args, lterm_list,
                                     W_derivs_list=None, W_lens_derivs_list=None, tr=None, Pk=None, t_grid=None):
    """
    Efficient computation of all bispectra for ell_list.

    This function:
    1. Loads all required data once (Cls, Am, Il, A0, A2, A4)
    2. Generates all valid triplets
    3. Parallelizes computation over triplets using numba
    4. Saves results to HDF5

    Parameters:
    -----------
    p : Param object with .which, .Newton, .rad
    ell_list : list - ell values
    time_dict : dict - cosmological functions
    chi_list : array - chi values for integration
    window_args : tuple - (r0, ddr, normW)
    W_derivs_list : list of arrays, optional
        Precomputed window derivatives. If None, will be computed from window_args.
    """

    # print(f"="*70)
    # print(f"Computing bispectrum for ell_list={ell_list[0]}-{ell_list[-1]}, which={p.which}")
    # print(f"="*70)

    # ========================================================================
    # 2. Load all data once
    # ========================================================================
    start_time = time.time()
    if p.which in ('local', 'equi', 'ortho'):
        Cl_array = load_and_compute_all_terms(
                        p, ell_list, chi_list, time_dict, window_args, lterm_list,
                        W_derivs_list=W_derivs_list, W_lens_derivs_list=W_lens_derivs_list, tr=tr, Pk=Pk, t_grid=t_grid)
        coeffs = (Cl_array[:, :, -1] * chi_list**2)[:, :, None]
    else:
        Cl_array, coeffs = load_and_compute_all_terms(
                        p, ell_list, chi_list, time_dict, window_args, lterm_list,
                        W_derivs_list=W_derivs_list, W_lens_derivs_list=W_lens_derivs_list, tr=tr, Pk=Pk, t_grid=t_grid)
    # print(f"Data loading completed in {time.time()-start_time:.2f} seconds")

    # get all ell triplets, wigner values, and Al1l2l3 coefficients (geometry only, cosmology-independent)
    triplet_array, wigner_array, config_name, Al1l2l3_array = ell_configurations(p, ell_list)

    n_triplets = len(triplet_array)
    # print(f"Valid triplets (non-zero Wigner): {n_triplets}")

    if n_triplets == 0:
        print("No valid triplets, exiting...")
        return

    # Initialize variance (will be computed on-demand if needed and not already in HDF5)
    var_array = None

    # ========================================================================
    # 4. Compute bispectra in parallel
    # ========================================================================
    # print(f"Computing bispectra in parallel...")
    start_time = time.time()

    if p.which in ['F2', 'G2', 'dv2']:
        # F2/G2/dv2: use full integration with 4 angular coefficients
        bl_results = compute_bispectrum_parallel_efficient(
            Cl_array, coeffs,
            chi_list, triplet_array
        )
    elif p.which == 'davd1v':
        # Use precomputed Al1l2l3 from ell_configurations (now parallelized and cached)
        bl_results = compute_bispectrum_quadratic_dav(
            Cl_array, coeffs,
            chi_list, triplet_array, Al1l2l3_array
            )
    else:
        # Quadratic terms: use simpler integration
        is_symmetric = (p.which[:3] == p.which[3:] or p.which in ['local', 'ortho'])  # True for d2vd2v, False for d1vd1d, etc.

        if is_symmetric:
            bl_results = compute_bispectrum_quadratic_symmetric(
                Cl_array, coeffs,
                chi_list, triplet_array
                )
        else:
            bl_results = compute_bispectrum_quadratic(
                Cl_array, coeffs,
                chi_list, triplet_array
                )

        # For ortho: divide by 6 (3 cyclic permutations * 2 from return statement)
        if p.which == 'ortho':
            bl_results = bl_results / 6.0

    # print(f"Computation completed in {time.time()-start_time:.2f} seconds")

    # ========================================================================
    # 5. Save results to HDF5
    # ========================================================================
    # Construct output file name (shared across all 'which' values)
    if p.rad and p.which in ['F2', 'G2', 'dv2']:
        name_suffix = '_rad'
    elif p.Newton:
        name_suffix = '_newton'
    else:
        name_suffix = ''

    # fNL is orthogonal to rad/Newton (pot_fnl is in the Newton lterm list too, and the rad
    # bl uses the linear factors), so append it rather than replace. fnl=0 keeps the bare
    # name, matching how rad/Newton treat their default: bl_all.h5, bl_all_fnl1.h5,
    # bl_all_rad_fnl1.h5, ...
    if p.fnl_local != 0:
        name_suffix += f'_fnl{p.fnl_local:g}'

    # Single file for all 'which' values
    file_path = f"{p.output_dir}bl_{p.lterm}{name_suffix}.h5"

    # print(f"Saving results to {file_path}...")

    with h5py.File(file_path, "a") as f:
        # Determine group name based on configuration
        group_name = config_name if config_name else 'all'

        # Create group if it doesn't exist
        if group_name not in f:
            grp = f.create_group(group_name)
        else:
            grp = f[group_name]

        # Check if variance already exists (for 'all' config only)
        variance_exists = (p.configuration not in ['equi', 'squ', 'folded']) and ('variance' in grp)

        # If variance doesn't exist and we need it, compute it now
        if (not variance_exists or p.force) and p.configuration not in ['equi', 'squ', 'folded', 'squ2']:
            if var_array is None:
                var_array = compute_variance_for_triplets(p, ell_list, triplet_array)

        # Prepare shared data based on configuration type
        if p.configuration == 'equi':
            # Equilateral: ell1=ell2=ell3, just store the unique ell values
            ell_array = np.array([ell_list[triplet_array[i][0]] for i in range(len(triplet_array))])
            shared_data = {
                'ell': ell_array,
                'wigner': wigner_array
            }
        elif p.configuration in ['squ', 'folded']:
            # Squeezed/Folded: ell1 fixed, ell2=ell3 varying
            ell1_fixed = ell_list[triplet_array[0][0]]
            ell23_array = np.array([ell_list[triplet_array[i][1]] for i in range(len(triplet_array))])
            shared_data = {
                'ell1': ell1_fixed,
                'ell23': ell23_array,
                'wigner': wigner_array
            }
        elif p.configuration == 'squ2':
            # Squeezed-2: ell2=ell3 fixed, ell1 (the long mode) varying -> ell1 is the x-axis
            ell23_fixed = ell_list[triplet_array[0][1]]
            ell1_array = np.array([ell_list[triplet_array[i][0]] for i in range(len(triplet_array))])
            shared_data = {
                'ell23': ell23_fixed,
                'ell1': ell1_array,
                'wigner': wigner_array
            }
        else:
            # All configurations: need full triplet specification
            ell1_array = np.array([ell_list[triplet_array[i][0]] for i in range(len(triplet_array))])
            ell2_array = np.array([ell_list[triplet_array[i][1]] for i in range(len(triplet_array))])
            ell3_array = np.array([ell_list[triplet_array[i][2]] for i in range(len(triplet_array))])
            shared_data = {
                'ell1': ell1_array,
                'ell2': ell2_array,
                'ell3': ell3_array,
                'wigner': wigner_array,
                'variance': var_array
            }

        # Check and save shared data
        for key, new_data in shared_data.items():
            if new_data is None:
                continue
            if key in grp:
                # Data exists - check consistency
                existing_data = grp[key][()]
                # Handle scalar vs array comparison
                if np.isscalar(new_data) or np.isscalar(existing_data):
                    is_consistent = np.isclose(existing_data, new_data)
                else:
                    is_consistent = np.array_equal(existing_data, new_data)

                if not is_consistent:
                    raise ValueError(
                        f"Inconsistent data for '{key}' in group '{group_name}'!\n"
                        f"Existing data shape: {np.shape(existing_data)}, new data shape: {np.shape(new_data)}\n"
                        f"This could indicate a mismatch in ell_list or configuration. "
                        f"Delete the file {file_path} and rerun if you want to use different parameters."
                    )
                # Data exists and is consistent - skip saving
            else:
                # Data doesn't exist - save it
                grp.create_dataset(key, data=new_data)

        # Save bispectrum with 'which'-specific name
        bl_dataset_name = f'bl_{p.which}'
        if bl_dataset_name in grp:
            del grp[bl_dataset_name]
        grp.create_dataset(bl_dataset_name, data=np.array(bl_results))

        # # Print summary
        # if p.configuration == 'equi':
        #     print(f"  Saved {len(triplet_array)} equilateral triplets as '{bl_dataset_name}' in group '{group_name}'")
        # elif p.configuration in ['squ', 'folded']:
        #     ell1_fixed = ell_list[triplet_array[0][0]]
        #     print(f"  Saved {len(triplet_array)} {p.configuration} triplets with ell1={ell1_fixed} as '{bl_dataset_name}' in group '{group_name}'")
        # else:
        #     print(f"  Saved {len(triplet_array)} triplets as '{bl_dataset_name}' in group '{group_name}'")

        f.flush()

    print(f"    Done! Results {p.which} saved to {file_path}")

