import numpy as np
from numba import njit, prange
import cubature, time, h5py
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from sympy.physics.wigner import wigner_3j
from filelock import FileLock
import os

from fftlog import *
from mathematica import *
from lincosmo import *
from param_used import *
import general_ps
import fctr

################################################################################ New efficient bispectrum computation
def load_and_compute_all_terms(p, ell_list, chi_list, time_dict, window_args, lterm_list,
                               W_derivs_list=None, tr=None, Pk=None, t_grid=None):
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
    print("  Loading power spectra from Cls.h5...")

    cls_file = f"{output_dir}Cls.h5"

    # For F2, G2, dv2: use 'FG2' to get nm_pairs
    # For other quadratic terms: concatenate nm_pairs from both parts
    if p.which in ['F2', 'G2', 'dv2']:
        nm_pairs = general_ps.get_nm_values('FG2')
        which_for_cls = 'FG2'
    else:
        # Quadratic term: concatenate nm_pairs from which[:3] and which[3:]
        nm_pairs = general_ps.get_nm_values(p.which[:3]) + general_ps.get_nm_values(p.which[3:])
        which_for_cls = p.which[:3]  # For now just compute first part

    # Single Cl_array for all cases
    Cl_array = np.zeros((n_ell, n_chi, len(nm_pairs)))

    kernels_array = np.zeros((n_ell, n_chi, 15))

    # Check if Cls.h5 exists and has required data
    need_to_compute_cls = False
    need_to_compute_am = False
    missing_cls_ells = []
    missing_am_ells = []

    if not os.path.exists(cls_file):
        need_to_compute_cls = True
        missing_cls_ells = list(ell_list)
        if p.which in ['F2', 'G2', 'dv2']:
            need_to_compute_am = True
            missing_am_ells = list(ell_list)
    else:
        # Check Cls and Am data separately
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
                    else:
                        missing_am_ells = list(ell_list)

                    if missing_am_ells:
                        need_to_compute_am = True

        except Exception as e:
            need_to_compute_cls = True
            missing_cls_ells = list(ell_list)
            if p.which in ['F2', 'G2', 'dv2']:
                need_to_compute_am = True
                missing_am_ells = list(ell_list)

    # Compute only what's missing
    if need_to_compute_cls or need_to_compute_am:
        print(f"\n{'='*70}")
        if need_to_compute_cls:
            print(f"  Missing Cls for ells: {missing_cls_ells}")
        if need_to_compute_am:
            print(f"  Missing Am/f-coefficients for ells: {missing_am_ells}")
        print(f"{'='*70}\n")

        if tr is None or Pk is None or t_grid is None:
            missing = missing_cls_ells if need_to_compute_cls else missing_am_ells
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
            fctr_dict = fctr.fct_of_r_analytical(p, missing_cls_ells, r_list, time_dict,
                                                  window_args, lterm_list)
            cp_dict = apply_fftlog_dict(tr['k'], tr['dTdk'] if p.rad else Pk, p, lterm_list)
            general_ps.compute_integral_generalized(p, missing_cls_ells, chi_list, r_list, t_grid, cp_dict, fctr_dict)
            print(f"  Cls computed and saved")

        # Step 2: Compute Am/f-coefficient terms if needed (only for F2/G2/dv2)
        if need_to_compute_am:
            p.which = original_which
            print(f"  Computing Am/f-coefficients ({p.which})...")
            fctr_dict = fctr.fct_of_r_analytical(p, missing_am_ells, r_list, time_dict,
                                                  window_args, lterm_list)
            cp_dict = apply_fftlog_dict(tr['k'], tr['dTdk'] if p.rad else Pk, p, lterm_list)
            general_ps.compute_integral_generalized(p, missing_am_ells, chi_list, r_list, t_grid, cp_dict, fctr_dict)
            print(f"  Am/f-coefficients computed and saved")

        # Restore original which
        p.which = original_which
        print(f"\n  All required data computed and saved to {cls_file}\n")

    with h5py.File(cls_file, 'r') as f:
        # Get ell_list and chi_list from file
        # They should be in each group, let's get from first group
        first_group_name = f'n_{nm_pairs[0][0]}_m_{nm_pairs[0][1]}'
        if first_group_name not in f:
            raise ValueError(f"Group {first_group_name} not found in {cls_file}")

        ell_list_file = f[first_group_name]['ell_list'][()]
        chi_list_file = f[first_group_name]['chi_list'][()]

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

        # Precompute ell indices once (outside the loop)
        ell_indices = []
        for ell in ell_list:
            idx = np.where(ell_list_file == ell)[0]
            if len(idx) == 0:
                print(f"  Warning: ell={ell} not found in file")
                ell_indices.append(-1)
            else:
                ell_indices.append(idx[0])
        ell_indices = np.array(ell_indices)
        valid_mask = ell_indices >= 0

        # For each (n,m) pair, sum over lterms
        for cl_idx, (n, m) in enumerate(nm_pairs):
            group_name = f'n_{n}_m_{m}'

            if group_name not in f:
                print(f"  Warning: Group {group_name} not found, skipping")
                continue

            group = f[group_name]

            # Sum over requested lterms
            Cl_nm_summed = np.zeros((len(ell_list_file), len(chi_list_file)))

            for lt in lterm_list:
                if lt in group:
                    Cl_nm_summed += group[lt][()]
                else:
                    print(f"  Warning: Dataset {lt} not found in {group_name}")

            # Extract all requested ells at once using fancy indexing
            Cl_subset = Cl_nm_summed[ell_indices[valid_mask], :]  # shape (n_valid_ells, n_chi_file)

            # Interpolate all ells at once if needed
            if not chi_match:
                # Vectorized interpolation: interpolate all ells simultaneously
                interp_func = interp1d(chi_list_file, Cl_subset, kind='linear',
                                      axis=1, bounds_error=False, fill_value=0.0)
                Cl_array[valid_mask, :, cl_idx] = interp_func(chi_list)
            else:
                Cl_array[valid_mask, :, cl_idx] = Cl_subset

    # ========================================================================
    # Apply stuff normalization for quadratic terms with d2p or d0p
    # ========================================================================
    if p.which not in ['F2', 'G2', 'dv2']:
        # Extract first and second parts
        which_parts = [p.which[:3], p.which[3:]]  # e.g., ['d2p', 'd2p'] or ['d1v', 'd1d']

        # Get number of nm_pairs for each part to know the Cl_array slicing
        nm_pairs_counts = [len(general_ps.get_nm_values(wp)) for wp in which_parts]
        cl_slices = [
            slice(0, nm_pairs_counts[0]),
            slice(nm_pairs_counts[0], nm_pairs_counts[0] + nm_pairs_counts[1])
        ]

        # Apply stuff normalization to each part separately
        for part_idx, which_part in enumerate(which_parts):
            if which_part in ['d2p', 'd0p']:
                stuff = 2.0 / (3.0 * omega_m * H0**2)
                print(f"  Applying stuff normalization to {which_part}: {stuff:.6e}")
                cl_slice = cl_slices[part_idx]
                Cl_array[:, :, cl_slice] /= stuff

    # ========================================================================
    # 2. Compute A0, A2, A4 kernels for all ells directly on chi_list
    # ========================================================================
    print("  Computing A0, A2, A4 kernels for all ells at once...")
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
            print("  Loading f-coefficient terms from HDF5...")
            try:
                with h5py.File(cls_file, 'r') as f:
                    if p.which not in f:
                        raise KeyError(f"Group {p.which} not found")

                    group = f[p.which]
                    ell_list_file = group['ell_list'][()]
                    chi_list_file = group['chi_list'][()]

                    # Check which ells are missing
                    missing_ells = [ell for ell in ell_list if ell not in ell_list_file]
                    if missing_ells:
                        raise KeyError(f"ells {missing_ells} not found")

                    # Extract for each ell and interpolate to chi_list
                    chi_match = np.array_equal(chi_list_file, chi_list)

                    for i, ell in enumerate(ell_list):
                        ell_idx = np.where(ell_list_file == ell)[0][0]

                        if p.which in ['G2', 'dv2']:
                            # Load f^(0) directly into A0 slots (0-2)
                            f0_data = group['f0'][()]  # shape: (n_components, n_ell_file, n_chi_file)
                            f0_comp1 = f0_data[0, ell_idx, :]  # f_{0,0}
                            f0_comp2 = f0_data[1, ell_idx, :]  # second component

                            if not chi_match:
                                interp_func = interp1d(chi_list_file, f0_comp1, kind='linear',
                                                      bounds_error=False, fill_value=0.0)
                                kernels_array[i, :, 0] = interp_func(chi_list)  # A00
                                kernels_array[i, :, 1] = -kernels_array[i, :, 0] / 2.  # A01 = -f_{0,0}/2

                                interp_func = interp1d(chi_list_file, f0_comp2, kind='linear',
                                                      bounds_error=False, fill_value=0.0)
                                kernels_array[i, :, 2] = interp_func(chi_list)  # A02
                            else:
                                kernels_array[i, :, 0] = f0_comp1
                                kernels_array[i, :, 1] = -f0_comp1 / 2.
                                kernels_array[i, :, 2] = f0_comp2

                            # If Newton=0, also load f^(-2) into Am slots (7-9, same as F2)
                            if not p.Newton:
                                fm2_data = group['fm2'][()]
                                fm2_comp1 = fm2_data[0, ell_idx, :]  # Only one component for G2/dv2

                                if not chi_match:
                                    interp_func = interp1d(chi_list_file, fm2_comp1, kind='linear',
                                                          bounds_error=False, fill_value=0.0)
                                    kernels_array[i, :, 7] = interp_func(chi_list)  # Am1
                                    kernels_array[i, :, 8] = -kernels_array[i, :, 7] / 2.  # Am2
                                    # kernels_array[i, :, 9] remains zero (no Am3 for G2/dv2)
                                else:
                                    kernels_array[i, :, 7] = fm2_comp1
                                    kernels_array[i, :, 8] = -fm2_comp1 / 2.
                                    # kernels_array[i, :, 9] remains zero

                        else:  # F2 with Newton=0
                            # Load f^(-2) into Am slots (7-9)
                            fm2_data = group['fm2'][()]
                            fm2_comp1 = fm2_data[0, ell_idx, :]
                            fm2_comp2 = fm2_data[1, ell_idx, :]

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
                            fm4_data = group['fm4'][()]
                            fm4_comp1 = fm4_data[0, ell_idx, :]

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
    # 5. Precompute combined coefficients for efficient integration
    # ========================================================================
    print("  Precomputing angular coefficient combinations...")

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

    else:
        # Quadratic terms: only need A00
        # Integrand is: A00 * Cl2 * Cl3 (for d2vd2v) or A00 * (Cl2_1*Cl3_2 + Cl3_1*Cl2_2) (for others)
        # Pad to 4 components for compatibility with integration function (last 3 are zeros)
        coeffs = np.zeros((n_ell, n_chi, 4))
        A0 = kernels['A0']
        coeffs[:, :, 0] = A0[:, 0, :]  # A00 (only non-zero component)

    # Return combined coefficients instead of individual kernels
    return Cl_array, coeffs

    #    # Load Il terms (radiation)
    #    if not p.Newton and p.rad:
    #        print("  Loading Il terms from HDF5...")
    #        try:
    #            with h5py.File(cls_file, 'r') as f:
    #                group = f[p.which]
    #                ell_list_file = group['ell_list'][()]
    #                chi_list_file = group['chi_list'][()]

    #                missing_ells = [ell for ell in ell_list if ell not in ell_list_file]
    #                if missing_ells:
    #                    raise KeyError(f"ells {missing_ells} not found")

    #                fm2_rad = group['fm2_rad'][()]
    #                fm4_rad = group['fm4_rad'][()]

    #                for i, ell in enumerate(ell_list):
    #                    ell_idx = np.where(ell_list_file == ell)[0][0]

    #                    Il_tab = np.zeros((len(chi_list_file), 5))
    #                    Il_tab[:, 0] = chi_list_file
    #                    Il_tab[:, 1] = fm2_rad[0, ell_idx, :]
    #                    Il_tab[:, 2] = -Il_tab[:, 1] / 2.
    #                    Il_tab[:, 3] = fm4_rad[0, ell_idx, :]
    #                    Il_tab[:, 4] = -Il_tab[:, 3] / 2.

    #                    # Interpolate Il terms onto chi_list
    #                    for col in range(1, 5):
    #                        interp_func = interp1d(Il_tab[:, 0], Il_tab[:, col], kind='linear',
    #                                               bounds_error=False, fill_value=0.0)
    #                        kernels_array[i, :, 11+col] = interp_func(chi_list)  # Slots 12-15

    #        except (FileNotFoundError, KeyError) as e:
    #            print(f"  Il data not found ({e}), computing...")
    #            general_ps.compute_integral_F2_G2_dv2(p, ell_list, chi_list,
    #                                                  time_dict['r_list'], time_dict, {}, {})
    #            # Retry loading
    #            with h5py.File(cls_file, 'r') as f:
    #                group = f[p.which]
    #                ell_list_file = group['ell_list'][()]
    #                chi_list_file = group['chi_list'][()]

    #                fm2_rad = group['fm2_rad'][()]
    #                fm4_rad = group['fm4_rad'][()]

    #                for i, ell in enumerate(ell_list):
    #                    ell_idx = np.where(ell_list_file == ell)[0][0]

    #                    Il_tab = np.zeros((len(chi_list_file), 5))
    #                    Il_tab[:, 0] = chi_list_file
    #                    Il_tab[:, 1] = fm2_rad[0, ell_idx, :]
    #                    Il_tab[:, 2] = -Il_tab[:, 1] / 2.
    #                    Il_tab[:, 3] = fm4_rad[0, ell_idx, :]
    #                    Il_tab[:, 4] = -Il_tab[:, 3] / 2.

    #                    for col in range(1, 5):
    #                        interp_func = interp1d(Il_tab[:, 0], Il_tab[:, col], kind='linear',
    #                                               bounds_error=False, fill_value=0.0)
    #                        kernels_array[i, :, 11+col] = interp_func(chi_list)

    return Cl_array, kernels_array


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
def compute_bispectrum_quadratic(ell_array, Cl_array, coeffs,
                                chi_list, triplet_list, apply_sign, is_symmetric):
    """
    Efficient parallel computation for quadratic terms (d2vd2v, d1vd1d, etc.)

    Parameters:
    -----------
    ell_array : (n_ell,) - ell values
    Cl_array : (n_ell, n_chi, 2) - [Cl_1, Cl_2] for the two parts of quadratic term
    coeffs : (n_ell, n_chi, 4) - Precomputed angular coefficients (only first component used)
    chi_list : (n_chi,) - chi values
    triplet_list : (n_triplets, 3) - indices into ell_array for (ell1, ell2, ell3)
    apply_sign : bool - if True, multiply result by -1
    is_symmetric : bool - True for d2vd2v where both parts are the same

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
        if is_symmetric:
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
        else:
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

        # Apply normalization and optional sign
        result = integral * 8.0 / (np.pi**2)
        if apply_sign:
            result *= -1.0

        results[idx] = result

    return results


@njit(parallel=True)
def compute_bispectrum_parallel_efficient(ell_array, Cl_array, coeffs,
                                         chi_list, triplet_list, apply_sign):
    """
    Efficient parallel computation using precomputed angular coefficients.

    Parameters:
    -----------
    ell_array : (n_ell,) - ell values
    Cl_array : (n_ell, n_chi, 3) - [Cl_m2, Cl_0, Cl_p2] (indices 0,1,2 for nm_pairs [(-2,0), (0,0), (2,0)])
    coeffs : (n_ell, n_chi, 4) - Precomputed angular coefficients [c_00, c_m2m2, c_p2m2, c_0m2]
    chi_list : (n_chi,) - chi values
    triplet_list : (n_triplets, 3) - indices into ell_array for (ell1, ell2, ell3)
    apply_sign : bool - if True, multiply result by -1

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

        # Apply normalization and optional sign
        result = integral * 8.0 / (np.pi**2)
        if apply_sign:
            result *= -1.0

        results[idx] = result

    return results


def compute_all_bispectra_efficient(p, ell_list, chi_list, time_dict, window_args, lterm_list,
                                     W_derivs_list=None, tr=None, Pk=None, t_grid=None):
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

    print(f"="*70)
    print(f"Computing bispectrum for ell_list={ell_list}, which={p.which}")
    print(f"="*70)

    # Determine which_code for numba
    which_codes = {'F2': 0, 'G2': 1, 'dv2': 2}
    if p.which in which_codes:
        which_code = which_codes[p.which]
    else:
        which_code = -1

    # Determine if sign flip needed
    apply_sign = p.which in ['G2', 'd2vd0d', 'd1vd1d', 'd1vd2v', 'd1vdod', 'd0pd3v', 'davd1v']

    # ========================================================================
    # 2. Load all data once
    # ========================================================================
    start_time = time.time()
    Cl_array, coeffs = load_and_compute_all_terms(
                        p, ell_list, chi_list, time_dict, window_args, lterm_list,
                        W_derivs_list=W_derivs_list, tr=tr, Pk=Pk, t_grid=t_grid)
    print(f"Data loading completed in {time.time()-start_time:.2f} seconds")

    # Create ell to index mapping
    ell_to_idx = {ell: i for i, ell in enumerate(ell_list)}

    # ========================================================================
    # 3. Generate triplet list and compute Wigner 3j symbols
    # ========================================================================
    config = p.configuration
    print(f"Generating triplets for {len(ell_list)} ells (configuration: {config})...")
    triplets = []
    wigner_values = []

    # Configuration-based triplet filtering
    if config == 'equi':
        # Equilateral: ell1 = ell2 = ell3
        for i, ell in enumerate(ell_list):
            wigner_test = float(wigner_3j(int(ell), int(ell), int(ell), 0, 0, 0))
            if wigner_test != 0:
                triplets.append([i, i, i])
                wigner_values.append(wigner_test)
        config_name = 'equilateral'

    elif config == 'squ':
        # Squeezed: ell1 fixed, ell2 = ell3 varying
        ell1_fixed = p.ell
        if ell1_fixed not in ell_list:
            print(f"Warning: ell1={ell1_fixed} not in ell_list, no triplets generated")
        else:
            i1 = ell_to_idx[ell1_fixed]
            for i23, ell23 in enumerate(ell_list):
                # Triangle inequality
                if ell23 < abs(ell1_fixed - ell23) or ell23 > ell1_fixed + ell23:
                    continue
                wigner_test = float(wigner_3j(int(ell1_fixed), int(ell23), int(ell23), 0, 0, 0))
                if wigner_test != 0:
                    triplets.append([i1, i23, i23])
                    wigner_values.append(wigner_test)
        config_name = f'squeezed_ell{ell1_fixed}'

    elif config == 'folded':
        # Folded: ell1 = ellmax, ell2 = ell3 varying
        ell1_fixed = p.ellmax
        if ell1_fixed not in ell_list:
            print(f"Warning: ellmax={ell1_fixed} not in ell_list, no triplets generated")
        else:
            i1 = ell_to_idx[ell1_fixed]
            for i23, ell23 in enumerate(ell_list):
                # Triangle inequality
                if ell23 < abs(ell1_fixed - ell23) or ell23 > ell1_fixed + ell23:
                    continue
                wigner_test = float(wigner_3j(int(ell1_fixed), int(ell23), int(ell23), 0, 0, 0))
                if wigner_test != 0:
                    triplets.append([i1, i23, i23])
                    wigner_values.append(wigner_test)
        config_name = f'folded_ell{ell1_fixed}'

    else:
        # All configurations: generate all valid triplets
        for i1, ell1 in enumerate(ell_list):
            for i2, ell2 in enumerate(ell_list):
                if ell2 < ell1:  # Only compute ell2 >= ell1 to avoid duplicates
                    continue
                for i3, ell3 in enumerate(ell_list):
                    if ell3 < ell2:  # Only compute ell3 >= ell2
                        continue
                    # Triangle inequality: |ell1 - ell2| <= ell3 <= ell1 + ell2
                    if ell3 < abs(ell1 - ell2) or ell3 > ell1 + ell2:
                        continue

                    # Convert to Python int for wigner_3j (which doesn't accept numpy integers)
                    wigner_test = float(wigner_3j(int(ell1), int(ell2), int(ell3), 0, 0, 0))
                    if wigner_test != 0:
                        triplets.append([i1, i2, i3])
                        wigner_values.append(wigner_test)
        config_name = ''

    n_triplets = len(triplets)
    print(f"Valid triplets (non-zero Wigner): {n_triplets}")

    if n_triplets == 0:
        print("No valid triplets, exiting...")
        return

    triplet_array = np.array(triplets, dtype=np.int64)
    wigner_array = np.array(wigner_values)

    # ========================================================================
    # 4. Compute bispectra in parallel
    # ========================================================================
    print(f"Computing bispectra in parallel...")
    start_time = time.time()

    if p.which in ['F2', 'G2', 'dv2']:
        # F2/G2/dv2: use full integration with 4 angular coefficients
        bl_results = compute_bispectrum_parallel_efficient(
            ell_list, Cl_array, coeffs,
            chi_list, triplet_array, apply_sign
        )
    else:
        # Quadratic terms: use simpler integration
        is_symmetric = (p.which[:3] == p.which[3:])  # True for d2vd2v, False for d1vd1d, etc.
        bl_results = compute_bispectrum_quadratic(
            ell_list, Cl_array, coeffs,
            chi_list, triplet_array, apply_sign, is_symmetric
        )

    print(f"Computation completed in {time.time()-start_time:.2f} seconds")

    # ========================================================================
    # 5. Save results to HDF5
    # ========================================================================
    # Construct output file name
    if p.rad and p.which in ['F2', 'G2', 'dv2']:
        name_suffix = '_rad'
    elif p.Newton:
        name_suffix = '_newton'
    else:
        name_suffix = ''

    # Use same file regardless of configuration
    file_path = f"{p.output_dir}bl/bl_{p.lterm}_{p.which}{name_suffix}.h5"
    lock_path = f"{file_path}.lock"

    print(f"Saving results to {file_path}...")

    with FileLock(lock_path):
        with h5py.File(file_path, "a") as f:
            # Determine group name based on configuration
            group_name = config_name if config_name else 'all'

            # Delete existing group if it exists
            if group_name in f:
                del f[group_name]

            # Create group
            grp = f.create_group(group_name)

            # Save based on configuration type
            if config == 'equi':
                # Equilateral: ell1=ell2=ell3, just store the unique ell values
                ell_array = [ell_list[triplets[i][0]] for i in range(len(triplets))]
                grp.create_dataset('ell', data=np.array(ell_array))
                grp.create_dataset('bl', data=np.array(bl_results))
                grp.create_dataset('wigner', data=np.array(wigner_values))
                print(f"  Saved {len(triplets)} equilateral triplets in group '{group_name}'")

            elif config in ['squ', 'folded']:
                # Squeezed/Folded: ell1 fixed, ell2=ell3 varying
                ell1_fixed = ell_list[triplets[0][0]]
                ell23_array = [ell_list[triplets[i][1]] for i in range(len(triplets))]
                grp.create_dataset('ell1', data=ell1_fixed)
                grp.create_dataset('ell23', data=np.array(ell23_array))
                grp.create_dataset('bl', data=np.array(bl_results))
                grp.create_dataset('wigner', data=np.array(wigner_values))
                print(f"  Saved {len(triplets)} {config} triplets with ell1={ell1_fixed} in group '{group_name}'")

            else:
                # All configurations: need full triplet specification
                ell1_array = [ell_list[triplets[i][0]] for i in range(len(triplets))]
                ell2_array = [ell_list[triplets[i][1]] for i in range(len(triplets))]
                ell3_array = [ell_list[triplets[i][2]] for i in range(len(triplets))]
                grp.create_dataset('ell1', data=np.array(ell1_array))
                grp.create_dataset('ell2', data=np.array(ell2_array))
                grp.create_dataset('ell3', data=np.array(ell3_array))
                grp.create_dataset('bl', data=np.array(bl_results))
                grp.create_dataset('wigner', data=np.array(wigner_values))
                print(f"  Saved {len(triplets)} triplets in group '{group_name}'")

            f.flush()

    print(f"Done! Results saved to {file_path}")

