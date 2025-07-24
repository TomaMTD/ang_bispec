import numpy as np
import os, sys
import time
from numba import njit, prange
import h5py
import threading

from param_used import *
from mathematica import *

# Global lock for HDF5 file access
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



def save_to_hdf5(filename, group_path, data, metadata=None):
    """Thread-safe HDF5 saving function"""
    with hdf5_lock:
        with h5py.File(filename, 'a') as f:
            # Check if group exists and handle force parameter
            if group_path in f and not force:
                print(f'    Group {group_path} already exists, skipping (use force=True to overwrite)')
                return False
            
            # Create group (remove existing if force=True)
            if group_path in f and force:
                print(f'    Group {group_path} exists, removing and recreating (force=True)')
                del f[group_path]
            
            group = f.create_group(group_path)
            
            # Save data
            for key, value in data.items():
                group.create_dataset(key, data=value)
            
            # Save metadata if provided
            if metadata:
                for key, value in metadata.items():
                    group.attrs[key] = value
            return True


def compute_integral_generalized(p, ell_list, chi_list, r_list, t_grid, cp, fctr):
    """
    Generalized computation function
    
    Parameters:
    -----------
    p : parameter object with attributes omega_m, H0, etc.
    cp : computation parameters dictionary
    chi_list, ell_list, r_list, t_grid : coordinate arrays
    fctr : factor array
    output_filename : HDF5 filename for output
    which_list : list of 'which' values to compute (if None, use p.which)
    lterm_list : list of 'lterm' values to compute (if None, use p.lterm)
    """
    print('----------------------------------------------------')
    
    def get_nm_values(which):
        """Get the (n,m) pairs for different 'which' cases"""
        nm_mapping = {
            'FG2': [(-2, 0), (0, 0), (2, 0)],
            'd1v': [(-1, 1)],
            'd2v': [(0, 2)],
            'd3v': [(1, 3)],
            'd1d': [(1, 1)]
        }
        return nm_mapping.get(which, [(0, 0)]) 


    def check_computation_exists(filename, n, m, lterm):
        """Check if a specific computation already exists"""
        try:
            with h5py.File(filename, 'r') as f:
                group_path = f'n_{n}_m_{m}'
                exists = group_path in f and lterm in f[group_path]
                return exists
        except (OSError, KeyError, IOError):
            return False

    middle = len(cp['eta_p']) // 2
    
    output_filename = f'{output_dir}Cls.h5'
    # Initialize HDF5 file
    try:
        # Try to open in read mode first to check if file exists
        with h5py.File(output_filename, 'r') as f:
            pass  # File exists, do nothing
    except (OSError, IOError):
        # File doesn't exist, create it
        with h5py.File(output_filename, 'w') as f:
            f.attrs['chi_list'] = chi_list
            f.attrs['ell_list'] = ell_list

    
    # Main computation loop
    nm_pairs = get_nm_values(p.which)
    kpow = 2 if p.lterm == 'density' and p.which in ['FG2', 'F2', 'G2'] else 0 

    
    # Compute factors that depend on lterm and which
    if p.lterm not in ['pot', 'dpot']:
        stuff2 = (2./3./omega_m/H0**2)**2
    elif p.lterm in ['pot', 'dpot']:
        stuff2 = (2./3./omega_m/H0**2)
    else:
        stuff2 = 1.0

    print(f'  Computing for lterm = {p.lterm}')
    for n, m in nm_pairs:
        # for m==0, the n values are taken into account in cp
        n_eff = n if m == 0 else 0

        if n_eff + kpow == 2:
            power_reduction = 1
        elif n_eff + kpow == 4:
            power_reduction = 2
        else:
            power_reduction = 0
        
        print(f'    Computing for (n,m) = ({n},{m}) with power_reduction = {power_reduction}')

        # Check if computation already exists
        if not force and check_computation_exists(output_filename, n, m, p.lterm):
            print(f'    Results for (n,m)=({n},{m}), lterm={p.lterm} already exist, skipping (use force=True to overwrite)')
            continue


        # Initialize result for this (which, lterm, n) combination
        result = np.zeros((len(ell_list),len(chi_list)), dtype=np.float64)
        
        # Loop over qterms
        for qt_ind, qt in enumerate(cp['qterm_list']):
            if len(cp['qterm_list'])>1: print(f'      Computing qterm: {qt}/{len(cp["qterm_list"])}')
            
            # Compute nu_p
            nu_p = 1 + cp[qt]['b'] + 1j*cp['eta_p'] + n_eff + kpow - 2*power_reduction
            
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
            result += stuff2 * integral_result
            print(f'        Integral computation done in {time.time()-start_time:.2f} seconds')
        
        group_path = f'n_{n}_m_{m}'
        data_to_save = {
            p.lterm: result,
            'ell_list': ell_list,  # Add ell_list to each group
            'chi_list': chi_list   # Add chi_list to each group
        }
        metadata = {
            'n': n,
            'm': m,
            'which': p.which,
            'lterm': p.lterm,
        }
        
        save_to_hdf5(output_filename, group_path, data_to_save, metadata)
        #print(f'    Saved results for (n,m)=({n},{m}) lterm={p.lterm}')




def get_all_Clnv1(p, ell_list, chi_list, r_list, t_grid, cp, fctr):
    '''
    Main function to compute the generalised power spectrum
    Computes the generalised power spectrum for all chi's and n's given ell. The result is normalised 
    by "stuff2=\mathcal N^2"


    returns \mathcal N^2 * \int dr f(r) \sum_p \int dk k^{nu_p-1} jl(k chi) jl(k r)
    '''
    if p.lterm not in ['pot', 'dpot']:
        stuff2=(2./3./p.omega_m/p.H0**2)**2
    elif p.lterm in ['pot', 'dpot']:
        stuff2=(2./3./p.omega_m/p.H0**2)
    else:
        stuff2=1.



    if p.lterm=='density' and p.which in ['FG2', 'F2', 'G2']: kpow=2.
    else: kpow=0.

    n=0
    
    middle = len(cp['eta_p']) // 2
    res = {'chi_list': chi_list, 'ell_list': ell_list}

    res[p.which] = np.zeros(len(chi_list), dtype=np.float64)
    for qt_ind, qt in enumerate(cp['qterm_list']):
        print('Computing qterm: {}/{}'.format(qt, len(cp['qterm_list'])))

        nu_p = 1 + cp[qt]['b'] + 1j*cp['eta_p'] + n + kpow
        a=time.time()
        F12 = compute_hyp21_grid_numba(t_grid, nu_p[:middle+1], ell_list)
        print('2F1 precomputation done in {:.2f} seconds'.format(time.time()-a))

        a=time.time()
        res[p.which] += stuff2*compute_integral_precompute(ell_list, chi_list, r_list,\
                                                    t_grid, nu_p, cp[qt]['cp'],\
                                                    F12, fctr[0,qt_ind])

        print('Integral performed in {:.2f} seconds'.format(time.time()-a))

    return 0












############################################################################## integrand for Cl
@njit
def theintegrand(rvar, chi, nu_p, ell, r_list, f_of_r):
    '''
    This function computes the integrand for a single frequency p: 2pi^2/r^2 * I_me(nu_p, r, chi)*f(r)
    '''

    result = np.zeros((len(rvar)), dtype=np.complex128)
    tvar=rvar/chi

    if ell>=5: t1min = tmin_fct(ell, nu_p)
    else: t1min=0
    
    for ind, t in enumerate(tvar[:,0]):

        res=myhyp21(nu_p, t, chi, ell, t1min) * f_of_r[ind] 
        result[ind] = res
    return result 

@njit
def theintegrand_sum(rvar, chi, ell, n, r_list, cp, f_of_r, N, kmax, kmin, kpow, b):
    '''
    Function summing theintegrand over the frequency before integration: sum_p theintegrand_p
    '''
    res = np.zeros((len(rvar[:,0])), dtype=np.complex128)
    for p in range(-N//2, N//2+1):
        eta_p = 2.*np.pi*p/np.log(kmax/kmin)
        nu_p = 1.+b+n+1j*eta_p + kpow
    
        val = cp[p+N//2]*theintegrand(rvar, chi, nu_p, ell, r_list, f_of_r)
        res+=val
    return res.real


@njit
def theintegrand_sum_quadratic(rvar, chi, ell, n, r_list, cp, f_of_r, N, kmax, kmin, kpow, b_list):
    '''
    Function summing theintegrand_quadratic over the frequency before integration: sum_p theintegrand_p
    '''

    res = np.zeros((len(rvar[:,0])), dtype=np.float64)
    for ind, b in enumerate(b_list):
        res+=theintegrand_sum(rvar, chi, ell, n, r_list, cp[:,ind], f_of_r[:,ind], N, kmax, kmin, kpow, b)

    return res

def get_Cl_sum(integrand, chi, ell, n, r_list, cp, fctr, rmin, rmax, N, kmax, kmin, kpow, b):
    '''
    Computes the generalised power spectrum for a given chi, ell and n

    integration of integrand which can be theintegrand_sum or theintegrand_sum_quadratic and division by 4pi
    returns  pi/2 \int dr/r^2 I_me(nu_p, r, chi) * f(r)
            = \int dr f(r) \sum_p \int dk k^{nu_p-1} jl(k chi) jl(k r)
    '''
    #print('  n={}'.format(n))
    if n+kpow==2:
        f_of_r=fctr[1]
        n-=2
    elif n+kpow==4:
        f_of_r=fctr[2]
        n-=4
    else:
        #f_of_r=fctr[1]
        #n-=2
        f_of_r=fctr[0]

    evaluation=integrand(r_list[:,None], chi, ell, n, r_list, cp, f_of_r, N, kmax, kmin, kpow, b)
    #np.save(output_dir+'check{:.0f}_n{}'.format(chi, nn), np.vstack([r_list, evaluation]))
    val=simpson(evaluation, x=r_list)

    return val/4./np.pi

def get_all_Clnold(which, qterm, lterm, Newton, chi_list, ell, r_list, cp, fctr, rmin, rmax, N, kmax, kmin, \
        kpow, b):
    '''
    Main function to compute the generalised power spectrum
    Computes the generalised power spectrum for all chi's and n's given ell. The result is normalised 
    by "stuff2=\mathcal N^2"


    returns \mathcal N^2 * \int dr f(r) \sum_p \int dk k^{nu_p-1} jl(k chi) jl(k r)
    '''
    if lterm not in ['pot', 'dpot']:
        stuff2=(2./3./omega_m/H0**2)**2
    elif lterm in ['pot', 'dpot']:
        stuff2=(2./3./omega_m/H0**2)
    else:
        stuff2=1.

    if lterm in ['pot', 'all'] and Newton: key_pot='_newton'
    else: key_pot=''

    if which in ['FG2', 'F2', 'G2', 'all']:
        res=np.zeros((len(chi_list), 4))
        cl_name = output_dir+'cln/Cln_{}{}_ell{}{}.txt'.format(lterm, key_pot, int(ell))
        integrand=theintegrand_sum
    else:
        if qterm==0:
            res=np.zeros((len(chi_list), 2))
            cl_name = output_dir+'cln/Cln_{}_{}{}_ell{}.txt'.format(which, lterm, key_pot, int(ell))
            integrand=theintegrand_sum_quadratic
        else:
            res=np.zeros((len(chi_list), 2))
            cl_name = output_dir+'cln/Cln_{}_qterm{}_{}{}_ell{}.txt'.format(which, qterm, lterm, key_pot, int(ell))
            integrand=theintegrand_sum 

    print(' ') 
    print('integration {}'.format(cl_name)) 
    if os.path.isfile(cl_name) and not force:
        res_test=np.loadtxt(cl_name)
        if len(res_test)==0: res[:,0]=chi_list
        else: res=res_test
    else:
        res[:,0]=chi_list

    a=time.time()
    for ind_chi, chi in enumerate(chi_list):

        if res[ind_chi,1]!=0 and not force: 
            print('     already computed -> jump')
            continue
        res[ind_chi,1]=stuff2*get_Cl_sum(integrand, chi, ell, 0, r_list, cp, fctr, rmin, rmax, N, kmax, kmin, kpow, b)

        if which in ['FG2', 'F2', 'G2']:
            if res[ind_chi,2]!=0 and not force: 
                print('     already computed -> jump')
                continue
            res[ind_chi,2]=stuff2*get_Cl_sum(integrand, chi, ell, -2, r_list, cp, fctr, rmin, rmax, N, kmax, kmin, kpow, b)

            if res[ind_chi,3]!=0 and not force: 
                print('     already computed -> jump')
                continue
            res[ind_chi,3]=stuff2*get_Cl_sum(integrand, chi, ell, 2, r_list, cp, fctr, rmin, rmax, N, kmax, kmin, kpow, b)
        
        if len(chi_list)==1:
            os.system("awk -i inplace '{{if (NR=={}) $2=\"{:.18e}\"; print $0}}' {}".format(ind_chi+1, res[ind_chi,1], cl_name))
            if which in ['FG2', 'F2', 'G2']:
                os.system("awk -i inplace '{{if (NR=={}) $3=\"{:.18e}\"; print $0}}' {}".format(ind_chi+1, res[ind_chi,2], cl_name))
                os.system("awk -i inplace '{{if (NR=={}) $4=\"{:.18e}\"; print $0}}' {}".format(ind_chi+1, res[ind_chi,3], cl_name))
        else:
            np.savetxt(cl_name, res) 
        print('  {}/{} chi={:.2f}, time {:.2f}'.format(ind_chi, len(res[:,0]), chi, time.time()-a))
    else:
        res=np.loadtxt(cl_name)
    return res
