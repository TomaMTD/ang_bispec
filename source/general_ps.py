import numpy as np
import os, sys
import time
from numba import njit
import cubature
from scipy.integrate import simpson

from param_used import *
from mathematica import *


############################################################################# integrand for Cl
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
    res = np.zeros((len(rvar[:,0])), dtype=np.float64)
    for ind, b in enumerate(b_list):
        res+=theintegrand_sum(rvar, chi, ell, n, r_list, cp[:,ind], f_of_r[:,ind], N, kmax, kmin, kpow, b)

    return res

def get_Cl_sum(integrand, chi, ell, n, r_list, cp, fctr, rmin, rmax, N, kmax, kmin, kpow, b, Limber=False):
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
        f_of_r=fctr[0]

    if not Limber:
        evaluation=integrand(r_list[:,None], chi, ell, n, r_list, cp, f_of_r, N, kmax, kmin, kpow, b)
        #np.save(output_dir+'check{:.0f}_n{}'.format(chi, nn), np.vstack([r_list, evaluation]))
        val=simpson(evaluation, x=r_list)
    else:
        #Pk = np.load(output_dir+'fct_k.npy')
        val=2*np.pi**2 * (ell/chi)**(n+kpow-2) * np.interp(np.log(ell/chi), np.log(cp[:,0]), cp[:,1]) 
        * np.interp(chi, r_list, f_of_r.real) / chi**2

    return val/4./np.pi

def get_all_Cln(which, qterm, lterm, Newton, chi_list, ell, r_list, cp, fctr, rmin, rmax, N, kmax, kmin, kpow, b, Limber=False):
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
        cl_name = output_dir+'cln/Cln_{}{}_ell{}.txt'.format(lterm, key_pot, int(ell))
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
        res[ind_chi,1]=stuff2*get_Cl_sum(integrand, chi, ell, 0, r_list, cp, fctr, rmin, rmax, N, kmax, kmin, kpow, b, Limber)

        if which in ['FG2', 'F2', 'G2']:
            if res[ind_chi,2]!=0 and not force: 
                print('     already computed -> jump')
                continue
            res[ind_chi,2]=stuff2*get_Cl_sum(integrand, chi, ell, -2, r_list, cp, fctr, rmin, rmax, N, kmax, kmin, kpow, b, Limber)

            if res[ind_chi,3]!=0 and not force: 
                print('     already computed -> jump')
                continue
            res[ind_chi,3]=stuff2*get_Cl_sum(integrand, chi, ell, 2, r_list, cp, fctr, rmin, rmax, N, kmax, kmin, kpow, b, Limber)
        
        if len(chi_list)==1:
            os.system("awk -i inplace '{{if (NR=={}) $2=\"{:.18e}\"; print $0}}' {}".format(ind_chi+1, res[ind_chi,1], cl_name))
            if which in ['FG2', 'F2', 'G2']:
                os.system("awk -i inplace '{{if (NR=={}) $3=\"{:.18e}\"; print $0}}' {}".format(ind_chi+1, res[ind_chi,2], cl_name))
                os.system("awk -i inplace '{{if (NR=={}) $4=\"{:.18e}\"; print $0}}' {}".format(ind_chi+1, res[ind_chi,3], cl_name))
        else:
            np.savetxt(cl_name, res) 
        if not Limber: print('  {}/{} chi={:.2f}, time {:.2f}'.format(ind_chi, len(res[:,0]), chi, time.time()-a))
    else:
        res=np.loadtxt(cl_name)
    return res




# debug
def plot_integrand(ell, n, r_list, y, y1, rmin, rmax, N, kmax, kmin, kpow, b):
    #rvar_list=np.loadtxt('qterms4_L10_r3200.dat')[:,0] #np.linspace(rmin, rmax, 403)
    rvar_list=np.linspace(rmin, rmax, 1000)
    chi =  3200 
    print(ell, n, chi)
    a=time.time()
    res = theintegrand_sum(rvar_list[:,None], chi, ell, n, r_list, y, N, kmax, kmin, kpow, b)
    np.savetxt('integrand.txt', np.array([rvar_list, res]).T)
    print(time.time()-a)




























