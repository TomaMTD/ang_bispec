import numpy as np
from numba import njit
import cubature, time, h5py
from scipy.integrate import simpson
from sympy.physics.wigner import wigner_3j
from filelock import FileLock

from fftlog import *
from mathematica import *
from lincosmo import *
from param_used import *

def sum_qterm_and_linear_term(which, Newton, Limber, lterm, ell, r_list=0, Hr=0, fr=0, Dr=0, ar=0):
    '''
    Loading the generalised power spectra and summing over qterm and lterm
    '''
    if Limber and ell>200: Limber='_Limber'
    else: Limber=''

    if which in ['d2p', 'd0p']:
        stuff=2./3./omega_m/H0**2
        if which=='d2p': which='d2v'
    else:
        stuff=1.

    if not isinstance(lterm, list):
        if lterm=='all':
            lterm=['density', 'rsd', 'pot', 'doppler', 'dpot']
        elif lterm=='nopot':
            lterm=['density', 'rsd', 'doppler'] 
        elif lterm=='noproj':
            lterm=['density', 'rsd'] #, 'pot', 'dpot']
        else:
            lterm=[lterm]

    if which in ['F2', 'G2', 'dv2']:
        for ind, lt in enumerate(lterm):
            if lt=='pot':  lt='pot_newton'
            if ind==0:
                Cl2_chi = np.loadtxt(output_dir+'cln/Cln_{}_ell{}{}.txt'.format(lt, int(ell), Limber))
            else:
                Cl2_chi[:,1:] += np.loadtxt(output_dir+'cln/Cln_{}_ell{}{}.txt'.format(lt, int(ell), Limber))[:,1:]

            if lt=='density' and not Newton: 
                Cl2_chi_R = np.loadtxt(output_dir+'cln/Cln_pot_ell{}{}.txt'.format(int(ell), Limber))
                Cl2_chi_N = np.loadtxt(output_dir+'cln/Cln_pot_newton_ell{}{}.txt'.format(int(ell), Limber))
                Cl2_chi[:,1:] += 2./3./omega_m/H0**2*(Cl2_chi_R[:,1:]-Cl2_chi_N[:,1:])

    elif which=='d0d':
        Cl2_chi=sum_qterm_and_linear_term('F2', Newton, Limber, lterm, ell)
        if not Newton:
            Cl2_chi[:,1]+=3.*np.interp(Cl2_chi[:,0], r_list, Hr)**2*np.interp(Cl2_chi[:,0], r_list, fr)\
                                    *Cl2_chi[:,2]

    elif which in ['d0p', 'dav']:
        Cl2_chi=sum_qterm_and_linear_term('F2', Newton, Limber, lterm, ell)
        Cl2_chi[:,1]=Cl2_chi[:,2]

    elif which=='dod':
        Cl2_chi=sum_qterm_and_linear_term('F2', Newton, Limber, lterm, ell)
        ff=np.interp(Cl2_chi[:,0], r_list, fr)
        HH=np.interp(Cl2_chi[:,0], r_list, Hr)
        if not Newton:
            Cl2_chi[:,1]=HH*np.interp(Cl2_chi[:,0], r_list, Dr) * \
                        (ff * Cl2_chi[:,1] + 3.*(ff*np.interp(Cl2_chi[:,0], r_list, dotH_(1./ar-1.))+\
                        HH**2*(3./2.*np.interp(Cl2_chi[:,0], r_list, Om_(1./ar-1.))-ff))*Cl2_chi[:,2])
        else:
            Cl2_chi[:,1]=HH*np.interp(Cl2_chi[:,0], r_list, Dr) * ff * Cl2_chi[:,1]

    else:
        ind=0
        for lt in lterm:
            #try: 
            #    if lt=='pot' and Newton: lt='pot_newton'
            #    #                print(output_dir+'cln/Cln_{}_{}_ell{}.txt not found, try qterm...'.format(which, lt, int(ell)))
            #    if which[:2]=='d2':
            #        qlist=[1, 2, 3]
            #    elif which[:2]=='d1':
            #        qlist=[1, 2]
            #    elif which[:2]=='d3': 
            #        qlist=[1, 2, 3, 4]
            #    else: 
            #        print('{} not recognised'.format(which))
            #        exit()
 
            #    for qt in qlist:
            #        if ind==0:
            #            Cl2_chi = np.loadtxt(output_dir+'cln/Cln_{}_qterm{}_{}_ell{}.txt'.format(which, qt, lt, int(ell)))
            #        else:
            #            Cl2_chi[:,1] += np.loadtxt(output_dir+'cln/Cln_{}_qterm{}_{}_ell{}.txt'.format(which, qt, lt, int(ell)))[:,1]


            #except FileNotFoundError:
            if lt=='pot': lt='pot_newton'
            if ind==0:
                Cl2_chi = np.loadtxt(output_dir+'cln/Cln_{}_{}_ell{}{}.txt'.format(which, lt, int(ell), Limber))
            else:
                Cl2_chi[:,1] += np.loadtxt(output_dir+'cln/Cln_{}_{}_ell{}{}.txt'.format(which, lt, int(ell), Limber))[:,1]

            if lt=='density' and not Newton: 
                Cl2_chi_R = np.loadtxt(output_dir+'cln/Cln_{}_pot_ell{}{}.txt'.format(which, int(ell), Limber))
                Cl2_chi_N = np.loadtxt(output_dir+'cln/Cln_{}_pot_newton_ell{}{}.txt'.format(which, int(ell), Limber))
                Cl2_chi[:,1:] += 2./3./omega_m/H0**2*(Cl2_chi_R[:,1:]-Cl2_chi_N[:,1:])
            ind+=1
        
        if which=='d1d' and not Newton:
            Cl2_chi[:,1]+=3.*np.interp(Cl2_chi[:,0], r_list, Hr)**2*np.interp(Cl2_chi[:,0], r_list, fr)\
                            *sum_qterm_and_linear_term('d1v', Newton, Limber, lterm, ell)[:,1]

    #np.savetxt(output_dir+'cl_{}_{}_ell{}.txt'.format(which, lterm[0], int(ell)), Cl2_chi)
    Cl2_chi[:,1:]/=stuff
    return Cl2_chi


############################################################################# definition of alpha, beta gamma as fct of chi
@njit
def alpha_chi(r, index, which, Dr, fr, vr, wr, Omr):
    if which=='F2':
        if index==0:
            res=(7.-3.*vr)/14.
        elif index==1:
            res=4.*fr + 3./2.*Omr - 9./7.*wr
        elif index==2:
            res=(18.*fr**2 + 9.*fr**2*Omr - 9./2.*fr*Omr)
        else:
            res=np.zeros((len(r)), dtype=np.float64)
    else:
        if index==0:
            res=fr-3./7.*wr
        elif index==1:
            res=-0.5*15.*Omr*fr
        else:
            res=np.zeros((len(r)), dtype=np.float64)
    
    return  res

@njit
def beta_chi(r, index, which, Dr, fr, vr, wr, Omr):
    if which=='F2':
        if index==0:
            res=np.ones((len(r)), dtype=np.float64)
        elif index==1:
            res=-2*fr**2 + 6*fr - 9./2.*Omr
        elif index==2:
            res=(36.*fr**2 + 18.*fr**2*Omr )
        else:
            res=np.zeros((len(r)), dtype=np.float64)
    else:
        if index==0:
            res=fr
        elif index==1:
            res=-12.*Omr*fr
        else:
            res=np.zeros((len(r)), dtype=np.float64)
    return  res

@njit
def gamma_chi(r, index, which, Dr, fr, vr, wr, Omr):
    if which=='F2':
        if index==0:
            res=np.zeros((len(r)), dtype=np.float64)
        elif index==1:
            res=1./2.*(-fr**2+fr  - 3.*Omr)
        elif index==2:
            res=1./4.*(18*fr**2+9.*(fr**2-fr)*Omr)
        else:
            res=np.zeros((len(r)), dtype=np.float64)
    else:
        res=-9./4.*Omr*fr #np.zeros((len(r)), dtype=np.float64)
    return  res

############################################################################# definition of fnm as function of chi
#@njit
def f0_nm(r, which, Dr, fr, vr, wr, Omr, Hr):
    res=np.zeros((4, len(r)), dtype=np.float64)
    res[0]=(beta_chi(r, 0, which, Dr, fr, vr, wr, Omr) - alpha_chi(r, 0, which, Dr, fr, vr, wr, Omr))/2.
    res[1]=(alpha_chi(r, 0, which, Dr, fr, vr, wr, Omr) - beta_chi(r, 0, which, Dr, fr, vr, wr, Omr))/4.
    res[2]=Hr**2/2.*(beta_chi(r, 1, which, Dr, fr, vr, wr, Omr)/2. - alpha_chi(r, 1, which, Dr, fr, vr, wr, Omr))
    res[3]=Hr**4/4.*alpha_chi(r, 2, which, Dr, fr, vr, wr, Omr)
    return res

#@njit
def f2_nm(r, which, Dr, fr, vr, wr, Omr, Hr):
    res=np.zeros((2, len(r)))
    res[0]=1./2.*(beta_chi(r, 0, which, Dr, fr, vr, wr, Omr)/2. - alpha_chi(r, 0, which, Dr, fr, vr, wr, Omr))
    res[1]=Hr**2/4.*alpha_chi(r, 1, which, Dr, fr, vr, wr, Omr)
    return res

#@njit
def f4_nm(r, which, Dr, fr, vr, wr, Omr, Hr):
    return 1./4.*alpha_chi(r, 0, which, Dr, fr, vr, wr, Omr)

############################################################################# integral over chi
#@njit
def A0_chi_F2(r, which, Dr, fr, vr, wr, Omr, Hr, Wr):
    #if which=='F2':
    return f0_nm(r, which, Dr, fr, vr, wr, Omr, Hr)*Dr**2* Wr
    #else:
    #    return 0

#@njit
def A2_chi(r, which, ell, Dr, fr, vr, wr, Omr, Hr, Wr, mathcalR):
    y=f2_nm(r, which, Dr, fr, vr, wr, Omr, Hr)*Dr**2*Wr 

    if which == 'F2':
        out = mathcalD(r, y, ell)
    elif which=='G2':
        out = np.gradient(np.gradient(y, r, axis=1), r, axis=1)
    else:
        y*=mathcalR
        out = np.gradient(y, r, axis=1)
    return out

#@njit
def A4_chi(r, which, ell, Dr, fr, vr, wr, Omr, Hr, Wr, mathcalR):
    y=f4_nm(r, which, Dr, fr, vr, wr, Omr, Hr)*Dr**2*Wr

    if which == 'F2':
        y=mathcalD(r, y[None,:], ell)
    elif which == 'G2':
        y=np.gradient(np.gradient(y[None, :], r, axis=1), r, axis=1)
    else:
        y*=mathcalR
        y=np.gradient(y[None, :], r, axis=1)
    return  mathcalD(r, y, ell)



############################################################################# RE
@njit
def fm2_nm(r, which, Dr, fr, vr, wr, Omr, Hr):
    H2 = Hr**2
    res=np.zeros((3, len(r)))
    res[0]=(beta_chi(r, 1, which, Dr, fr, vr, wr, Omr) - alpha_chi(r, 1, which, Dr, fr, vr, wr, Omr))/2.-2.*gamma_chi(r, 1, which, Dr, fr, vr, wr, Omr)
    res[1]=(alpha_chi(r, 1, which, Dr, fr, vr, wr, Omr) - beta_chi(r, 1, which, Dr, fr, vr, wr, Omr))/4.\
            +gamma_chi(r, 1, which, Dr, fr, vr, wr, Omr)
    res[2]=H2/2.*(beta_chi(r, 2, which, Dr, fr, vr, wr, Omr)/2. - alpha_chi(r, 2, which, Dr, fr, vr, wr, Omr))
    return res*H2

@njit
def fm4_nm(r, which, Dr, fr, vr, wr, Omr, Hr):
    res=np.zeros((2, len(r)))
    res[0]= (beta_chi(r, 2, which, Dr, fr, vr, wr, Omr) - alpha_chi(r, 2, which, Dr, fr, vr, wr, Omr))/2.-2.*gamma_chi(r, 2, which, Dr, fr, vr, wr, Omr)
    res[1]=(alpha_chi(r, 2, which, Dr, fr, vr, wr, Omr) - beta_chi(r, 2, which, Dr, fr, vr, wr, Omr))/4.+gamma_chi(r, 2, which, Dr, fr, vr, wr, Omr)
    return res*Hr**4

################################################################################ radiation
@njit
def fm2R_nm(which, Dr, fr, vr, wr, Omr, Hr, ar):
    res=np.zeros((2, len(fr)))
    res[0]=fr+3.*Omr/2.
    res[1]=-res[0]/2.
    return res*Hr**2 * Dr/ar

@njit
def fm4R_nm(which, Dr, fr, vr, wr, Omr, Hr, ar):
    res=np.zeros((2, len(fr)))
    if which=='F2':
        res[0]= 3.*fr*(fr+3.*Omr/2.)
    else:
        res[0]= 3.*(fr-1)*(fr+3.*Omr/2.)
    res[1]=-res[0]/2.
    return res*Hr**4 * Dr/ar

@njit
def nb_gradient(f, x):
    out = np.empty_like(f, np.float64)
    out[1:-1] = (f[2:] - f[:-2]) / (x[2:] - x[:-2])
    out[0] = (f[1] - f[0]) / (x[1] - x[0]) 
    out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
    return out

@njit
def integrand_Am(r, chi, ell, which, r_list, \
        ar, Dr, fr, vr, wr, Omr, Hr, mathcalR, r0, ddr, normW, \
        fm2_tab, fm4_tab,\
        cp_tr, bphi, Nphi, eta):
    '''
    For all multiplet n and m, this function computes the integrand of the pure relativistic \
            terms coming from the k1^-2 and k1^-4 terms of the kernel
    
    returns Am2=2/pi * D^2(r)W_tilde(r) f^{(-2)}_{nm}(r) \int dk jl(k chi)*jl(k r)
            Am4=2/pi * D^2(r)W_tilde(r) f^{(-4)}_{nm}(r) \int dk k^{-2} jl(k chi)*jl(k r)
            
    '''
    t_list=r[:,0]/chi
    Am4=np.zeros(len(t_list), dtype=np.complex128)
    out=np.zeros((5, len(t_list)), dtype=np.float64)
    
    #Drr=np.interp(r[:,0], r_list, Dr)
    #frr=np.interp(r[:,0], r_list, fr)
    #vrr=np.interp(r[:,0], r_list, vr)
    #wrr=np.interp(r[:,0], r_list, wr)
    #Omrr=np.interp(r[:,0], r_list, Omr)
    #Hrr=np.interp(r[:,0], r_list, Hr)

    for ind,t in enumerate(t_list):
        if t>1:
            fact=t
            t=1./t
        else:
            fact=1.
        Am4[ind]=chi*fact*Il(-1+0.j, t+0.j, ell) 
    
    out[0]=np.interp(r[:,0], r_list, fm2_tab[0]) 
    out[1]=np.interp(r[:,0], r_list, fm2_tab[1]) 
    out[2]=np.interp(r[:,0], r_list, fm2_tab[2]) 
    out[3]=np.interp(r[:,0], r_list, fm4_tab[0])
    out[4]=np.interp(r[:,0], r_list, fm4_tab[1])

    return (out*Am4.real).T/2/np.pi**2


@njit
def integrand_Il_F2(r, chi, ell, which, r_list, \
        ar, Dr, fr, vr, wr, Omr, Hr, mathcalR, r0, ddr, normW, \
        f0_tab, fm2_tab,\
        cp_tr, bphi, Nphi, eta):
    '''
    For all multiplet n and m, this function computes the integrand of the radiation term \
            \partial \log T_{phi_0} / \partial \log k = \sum_p c_p k^{b+i\eta_p}
    
    returns 2/pi * D^2(r)W_tilde(r) f^{(R)}_{nm}(r) \sum_p c_p * \int dk k**(nu_p-1) jl(k*chi)jl(k*r)
    '''

    t_list=r[:,0]/chi
    out=np.zeros((4, len(t_list)), dtype=np.float64)
    
    arr=np.interp(r[:,0],  r_list, ar)
    Drr=np.interp(r[:,0],  r_list, Dr)
    frr=np.interp( r[:,0], r_list, fr)
    vrr=np.interp( r[:,0], r_list, vr)
    wrr=np.interp( r[:,0], r_list, wr)
    Omrr=np.interp(r[:,0], r_list, Omr)
    Hrr=np.interp( r[:,0], r_list, Hr)
    Wrr=W_tilde(r[:,0], r0, ddr, r_list, Hr, ar, normW)

    fm20=np.interp(r[:,0], r_list, fm2_tab[0]) 
    fm21=np.interp(r[:,0], r_list, fm2_tab[1]) 
    fm4=fm4R_nm(which, Drr, frr, vrr, wrr, Omrr, Hrr, arr)

    Ilm4=np.zeros(len(t_list), dtype=np.complex128)
    for p in range(-Nphi//2, Nphi//2+1):
        nu=1.+bphi+1j*p*eta
        if ell>=5: 
            t1min = tmin_fct(ell, nu)
        else: 
            t1min=0

        for ind, t in enumerate(t_list):
            Ilm4[ind]+=cp_tr[p+Nphi//2]*myhyp21(nu-2., t, chi, ell, t1min)

    # here we have computed 
    # 2pi^2/r^2 \sum c_p I_me(1.+bphi+1j*p*eta, r1, chi)
    WDr=Drr**2*Wrr
    out[0]=fm20*Ilm4.real
    out[1]=fm21*Ilm4.real
    out[2]=fm4[0]*Ilm4.real*WDr
    out[3]=fm4[1]*Ilm4.real*WDr
    return out.T/2/np.pi**2 

@njit
def integrand_Il_G2(r, chi, ell, which, r_list, \
        ar, Dr, fr, vr, wr, Omr, Hr, mathcalR, r0, ddr, normW, \
        f0_tab, fm2_tab,\
        cp_tr, bphi, Nphi, eta):
    t_list=r[:,0]/chi
    out=np.zeros((4, len(t_list)), dtype=np.float64)
 
    Ilm2=np.zeros(len(t_list), dtype=np.complex128)
    Ilm4=np.zeros(len(t_list), dtype=np.complex128)
    for p in range(-Nphi//2, Nphi//2+1):
        nu=-1.+bphi+1j*p*eta
        if ell>=5: 
            t1min = tmin_fct(ell, nu)
        else: 
            t1min=0
        
        for ind, t in enumerate(t_list):
            Ilm2[ind]+=cp_tr[p+Nphi//2]*myhyp21(nu, t, chi, ell, t1min)
            Ilm4[ind]+=cp_tr[p+Nphi//2]*myhyp21(nu-2., t, chi, ell, t1min)
    
    out[0]=np.interp(r[:,0], r_list, fm2_tab[0])*Ilm2.real
    out[1]=np.interp(r[:,0], r_list, fm2_tab[1])*Ilm2.real
    out[2]=np.interp(r[:,0], r_list, f0_tab[0])*Ilm4.real
    out[3]=np.interp(r[:,0], r_list, f0_tab[1])*Ilm4.real

    return out.T/2./np.pi**2


def get_Am_and_Il(chi_list, ell1, which, Newton, rad, time_dict, r0, ddr, normW, rmin, rmax, \
        cp_tr, bphi, Nphi, kmax, kmin, save=True):
    '''
    Computes the r and k integrals of the pure relativistic terms coming from the k1^-2 and k1^-4 terms of the kernel. 
    ell1 is fixed. All multiplet of n, m are computed as function of chi
    
    if rad==False: 
        computes the relativistic terms. Those terms lead to a function of chi that can be \
        expressed as an integral over r1:

        returns integration of the function integrand_Am over r for all chi_list values and multiplication by chi^2:
            2/pi* chi^2 * \int dr D^2(r)W_tilde(r) f^{(-2)}_{nm}(r) \int dk jl(k chi)*jl(k r)
            and         
            2/pi* chi^2 * \int dr D^2(r)W_tilde(r) f^{(-4)}_{nm}(r) \int dk k^{-2} jl(k chi)*jl(k r)

    else radiation is on:
        computes the radiation term. Those terms lead to a function of chi that can be \
        expressed as an integral over r1:

        returns integration of the function integrand_Il_F2 over r for all chi_list values and multiplication by chi^2:
            2/pi* chi^2 * \int dr D^2(r)W_tilde(r) f^{(R)}_{nm}(r) \sum_p c_p * \int dk k**(nu_p-1) jl(k*chi)jl(k*r)
    
    '''


    eta=2.*np.pi/np.log(kmax/kmin)
    fm2_tab, f0_tab=0, 0
    
    if rad: print('     Computing I_ell={}'.format(int(ell1)))
    else: print('     Computing A_ell={}'.format(int(ell1)))

    if rad:
        fdim=4
        fn = 'Il/Il_{}_ell{}.txt'
        
        fm2_tab=fm2R_nm(which, time_dict['Dr'], time_dict['fr'], time_dict['vr'], time_dict['wr'],\
                                time_dict['Omr'], time_dict['Hr'], time_dict['ar'])\
                                *time_dict['Dr']**2*time_dict['Wr']
        f0_tab=fm4R_nm(which, time_dict['Dr'], time_dict['fr'], time_dict['vr'], time_dict['wr'],\
                                time_dict['Omr'], time_dict['Hr'], time_dict['ar'])\
                                *time_dict['Dr']**2*time_dict['Wr']
        if which=='F2':
            fm2_tab = mathcalD(time_dict['r_list'], fm2_tab, ell1, axis=1)
            integ = integrand_Il_F2
        elif which=='G2':
            integ = integrand_Il_G2
            fm2_tab = np.gradient(np.gradient(fm2_tab, \
                            time_dict['r_list'], axis=1), time_dict['r_list'], axis=1)
            f0_tab = np.gradient(np.gradient(f0_tab, \
                            time_dict['r_list'], axis=1), time_dict['r_list'], axis=1)
        elif which=='dv2':
            integ = integrand_Il_G2
            fm2_tab = np.gradient(fm2_tab*\
                        time_dict['mathcalR']*time_dict['Hr'], time_dict['r_list'], axis=1)
            f0_tab = np.gradient(f0_tab*\
                        time_dict['mathcalR']*time_dict['Hr'], time_dict['r_list'], axis=1)

    else:
        fdim=5
        if not Newton: fn = 'Am/Am_{}_ell{}.txt'
        else: fn = 'Am/Am_{}_ell{}_newton.txt'
        if Newton: Hr=0
        else: Hr=time_dict['Hr']
        integ = integrand_Am


        if which=='F2': 
            if Newton: return 0
            else:
                f0_tab=fm2_nm(time_dict['r_list'], which, time_dict['Dr'],\
                        time_dict['fr'], time_dict['vr'], time_dict['wr'],\
                        time_dict['Omr'], Hr)*time_dict['Dr']**2*time_dict['Wr']
    
                fm2_tab=fm4_nm(time_dict['r_list'], which, time_dict['Dr'],\
                            time_dict['fr'], time_dict['vr'], time_dict['wr'],\
                            time_dict['Omr'], Hr)*time_dict['Dr']**2*time_dict['Wr']
 
        else:
            if Newton: Hr=0
            else: Hr=time_dict['Hr']
    
            f0_tab=f0_nm(time_dict['r_list'], which, time_dict['Dr'],\
                    time_dict['fr'], time_dict['vr'], time_dict['wr'],\
                    time_dict['Omr'], Hr)*time_dict['Dr']**2*time_dict['Wr']

            fm2_tab=fm2_nm(time_dict['r_list'], which, time_dict['Dr'],\
                            time_dict['fr'], time_dict['vr'], time_dict['wr'],\
                            time_dict['Omr'], Hr)*time_dict['Dr']**2*time_dict['Wr']

            if which=='G2':
                f0_tab =np.gradient(np.gradient(f0_tab, time_dict['r_list'], axis=1), time_dict['r_list'], axis=1)
                fm2_tab=np.gradient(np.gradient(fm2_tab, time_dict['r_list'], axis=1), time_dict['r_list'], axis=1)
            else:
                f0_tab *=time_dict['mathcalR']*time_dict['Hr']
                fm2_tab*=time_dict['mathcalR']*time_dict['Hr']
                f0_tab =np.gradient(f0_tab, time_dict['r_list'], axis=1)
                fm2_tab=np.gradient(fm2_tab, time_dict['r_list'], axis=1)
    
        f0_tab = mathcalD(time_dict['r_list'], f0_tab, ell1, axis=1)

    try : 
        res=(np.loadtxt(output_dir+fn.format(which, int(ell1)))[:,1:]).T
    except (OSError, FileNotFoundError, EOFError, ValueError): 
        res=np.zeros((fdim, len(chi_list)))
    #if ell1<100:

    for ind, chi in enumerate(chi_list):
        if ind%10==0 and rad: print('   {}/{}'.format(ind, len(chi_list)))
        
        if res[0,ind]!=0 and not force: 
            print('     already computed -> jump chi_ind='+str(ind))
            continue
        else:


            evaluation=integ(time_dict['r_list'][:,None], chi, ell1, which, time_dict['r_list'],\
                                            time_dict['ar'], time_dict['Dr'], time_dict['fr'],\
                                            time_dict['vr'], time_dict['wr'], time_dict['Omr'],\
                                            time_dict['Hr'], time_dict['mathcalR'], r0, ddr, normW,\
                                            f0_tab, fm2_tab,\
                                            cp_tr, bphi, Nphi, eta) 
            val=simpson(evaluation.T, x=time_dict['r_list'])

            #val, err = cubature.cubature(integ, ndim=1, fdim=fdim, xmin=[rmin], xmax=[rmax],\
            #                             args=(chi, ell1, which, time_dict['r_list'],\
            #                                time_dict['ar'], time_dict['Dr'], time_dict['fr'],\
            #                                time_dict['vr'], time_dict['wr'], time_dict['Omr'],\
            #                                time_dict['Hr'], time_dict['mathcalR'], r0, ddr, normW,\
            #                                f0_tab, fm2_tab,\
            #                                cp_tr, bphi, Nphi, eta), relerr=1e-5,\
            #                                     maxEval=1e6, abserr=0, vectorized=True)
            
            res[:,ind]=val*chi**2

            if save: 
                np.savetxt(output_dir+fn.format(which, int(ell1)), np.vstack([chi_list, res]).T) 

    # Limber seems not to work ...
    #else:
    #    ar=np.interp( chi_list, time_dict['r_list'], time_dict['ar'])
    #    Dr=np.interp( chi_list, time_dict['r_list'], time_dict['Dr'])
    #    fr=np.interp( chi_list, time_dict['r_list'], time_dict['fr'])
    #    vr=np.interp( chi_list, time_dict['r_list'], time_dict['vr'])
    #    wr=np.interp( chi_list, time_dict['r_list'], time_dict['wr'])
    #    Omr=np.interp(chi_list, time_dict['r_list'], time_dict['Omr'])
    #    Hr=np.interp( chi_list, time_dict['r_list'], time_dict['Hr'])
    #    Wr=W_tilde(chi_list, r0, ddr, normW)
    #
    #    fm2=fm2R_nm(which, Dr, fr, vr, wr, Omr, Hr, ar)
    #    fm4=fm4R_nm(which, Dr, fr, vr, wr, Omr, Hr, ar)
    #
    #    p_range=np.arange(-Nphi//2, Nphi//2+1)
    #    nu=bphi+1j*p_range*eta
    #        
    #    limber=(cp_tr*(ell1/chi_list[:,None])**(nu-3.)).sum(1)
    #    WDr=Dr**2*Wr
    #    res[0]=fm2[0]*limber.real*WDr
    #    res[1]=fm2[1]*limber.real*WDr
    #    res[2]=fm4[0]*limber.real*WDr
    #    res[3]=fm4[1]*limber.real*WDr
    
    return np.vstack([chi_list, res]).T


################################################################################ spherical bispectrum
@njit
def final_integrand(chi, which, Newton, rad, Cl2n1_chi, Cl3n1_chi, Cl2n2_chi, Cl3n2_chi,\
                        r_list, A0_tab, A2_tab, A4_tab, Am_tab, Il_tab):
    '''
    The sum over n and m is now explicitely computed. The result is a function of chi which will be integrated over.

    returns 2/pi* chi^2 * \sum_{mn} C_{\ell2}^{(n)}(chi) C_{\ell3}^{(m)}(chi) 
                
                \int dr D^2(r)W_tilde(r) f^{(X)}_{nm}(r) \int dk k^(X)*jl(k chi)*jl(k r) where X is an integer
            or
                \int dr D^2(r)W_tilde(r) f^{(R)}_{nm}(r) \sum_p c_p * \int dk k**(nu_p-1) jl(k*chi)jl(k*r) for radiation
    '''

    chi=chi[:,0]
    Cl2_0 = np.interp(chi, Cl2n1_chi[:,0], Cl2n1_chi[:,1])
    Cl3_0 = np.interp(chi, Cl3n1_chi[:,0], Cl3n1_chi[:,1])

    if which in ['F2', 'G2', 'dv2']:

        A20 = np.interp(chi, r_list, A2_tab[0])
        A21 = np.interp(chi, r_list, A2_tab[1])
        A40 = np.interp(chi, r_list, A4_tab[0])

        Cl2_p2 = np.interp(chi, Cl2n1_chi[:,0], Cl2n1_chi[:,3])
        Cl2_m2 = np.interp(chi, Cl2n1_chi[:,0], Cl2n1_chi[:,2])
        Cl3_m2 = np.interp(chi, Cl3n1_chi[:,0], Cl3n1_chi[:,2])
        Cl3_p2 = np.interp(chi, Cl3n1_chi[:,0], Cl3n1_chi[:,3])

        if which == 'F2':
            A00 = np.interp(chi, r_list, A0_tab[0])
            A01 = np.interp(chi, r_list, A0_tab[1])
            A02 = np.interp(chi, r_list, A0_tab[2])
            A03 = np.interp(chi, r_list, A0_tab[3])
    
            out = A00*Cl2_0*Cl3_0 \
              +(A03+A21+A40)*Cl2_m2*Cl3_m2 \
              +A01*(Cl3_p2*Cl2_m2+Cl3_m2*Cl2_p2)\
              +(A02+A20)*(Cl2_0*Cl3_m2+Cl3_0*Cl2_m2)

        else :
            out = (A21+A40)*Cl2_m2*Cl3_m2 \
              +(A20)*(Cl2_0*Cl3_m2+Cl3_0*Cl2_m2)

        if not Newton or (Newton and which!='F2'):
            out += np.interp(chi, Am_tab[:,0], Am_tab[:,1])*Cl2_0*Cl3_0+\
                 np.interp(chi, Am_tab[:,0], Am_tab[:,2])*(Cl3_p2*Cl2_m2+Cl3_m2*Cl2_p2)+\
                 np.interp(chi, Am_tab[:,0], Am_tab[:,3])*(Cl3_0*Cl2_m2+Cl3_m2*Cl2_0)+\
                 np.interp(chi, Am_tab[:,0], Am_tab[:,4])*Cl2_0*Cl3_0+\
                 np.interp(chi, Am_tab[:,0], Am_tab[:,5])*(Cl3_p2*Cl2_m2+Cl3_m2*Cl2_p2)

        if rad and not Newton:
            out += np.interp(chi, Il_tab[:,0], Il_tab[:,1])*Cl2_0*Cl3_0+\
                       np.interp(chi, Il_tab[:,0], Il_tab[:,2])*(Cl3_p2*Cl2_m2+Cl3_m2*Cl2_p2)+\
                       np.interp(chi, Il_tab[:,0], Il_tab[:,3])*Cl2_0*Cl3_0+\
                       np.interp(chi, Il_tab[:,0], Il_tab[:,4])*(Cl3_p2*Cl2_m2+Cl3_m2*Cl2_p2)

    else:
        A00 = np.interp(chi, r_list, A0_tab[0])
        if which=='d2vd2v':
            out = A00*Cl2_0*Cl3_0 
        else:
            Cl2_0_2 = np.interp(chi, Cl2n2_chi[:,0], Cl2n2_chi[:,1])
            Cl3_0_2 = np.interp(chi, Cl3n2_chi[:,0], Cl3n2_chi[:,1])
            out = A00*(Cl2_0*Cl3_0_2+Cl3_0*Cl2_0_2)
    return out


################################################################################ spherical bispectrum
def get_kernels(ell1, Newton, rad, which, chi_list, time_dict, r0, ddr, normW, rmin,\
                                rmax, cp_tr, b, Nk, kmax, kmin):

    A0_tab, A2_tab, A4_tab, Am_tab, Il_tab = np.zeros((5, 5)), np.zeros((5, 5)), \
                                             np.zeros((5, 5)), np.zeros((5, 5)), np.zeros((5, 5)) 
    
    if which in ['F2', 'G2', 'dv2']:
        Il_fn='Il/Il_{}_ell{}.txt'
        if not Newton and rad:
            try:
                Il_tab = np.loadtxt(output_dir+Il_fn.format(which, int(ell1)))
            except FileNotFoundError:
                Il_tab = get_Am_and_Il(chi_list, ell1, which, Newton, True, time_dict, r0, ddr, normW, rmin,\
                                    rmax, cp_tr, b, Nk, kmax, kmin, True)
        
        Am_fn='Am/Am_{}_ell{}{}.txt'
        if not Newton:
            Am_new=''
            Hr = time_dict['Hr']
        else:
            if which=='F2': Am_new=''
            else: Am_new='_newton'
            Hr = 0

        if not Newton or (Newton and which!='F2'):
            try:
                Am_tab = np.loadtxt(output_dir+Am_fn.format(which, \
                    int(ell1), Am_new))
            except (ValueError, FileNotFoundError):
                Am_tab = get_Am_and_Il(chi_list, ell1, which, Newton, False, time_dict, r0, ddr, normW, rmin,\
                                    rmax, cp_tr, b, Nk, kmax, kmin, True)

        A0_tab = A0_chi_F2(time_dict['r_list'], which,       time_dict['Dr'], time_dict['fr'], \
                time_dict['vr'], time_dict['wr'], time_dict['Omr'], Hr, time_dict['Wr'])
        A2_tab = A2_chi(time_dict['r_list'], which, ell1, time_dict['Dr'], time_dict['fr'], \
                time_dict['vr'], time_dict['wr'], time_dict['Omr'], Hr, time_dict['Wr'], \
                time_dict['mathcalR'])
        A4_tab = A4_chi(time_dict['r_list'], which, ell1, time_dict['Dr'], time_dict['fr'], \
                time_dict['vr'], time_dict['wr'], time_dict['Omr'], Hr, time_dict['Wr'], 
                time_dict['mathcalR'])
        
        if which=='dv2':
            A2_tab*=time_dict['Hr']
            A4_tab*=time_dict['Hr']


    else:
        if which in ['d2vd0d', 'd1vd1d', 'd1vd0d']:
            A0_tab = time_dict['Dr']**2*time_dict['fr']*time_dict['Wr'] 
            if which in ['d1vd0d']:
                A0_tab*=time_dict['Hr']*time_dict['mathcalR'] 

        elif which in ['d1vdod']:
            A0_tab = time_dict['Dr']*time_dict['fr']*time_dict['Wr'] 
        elif which in ['d0pd3v', 'd0pd1d', 'd1vd2p']:
            A0_tab = time_dict['Dr']**2*time_dict['Wr']/time_dict['Hr']/time_dict['ar']
            if 'v' in which:
                A0_tab*=time_dict['fr']
        else:
            A0_tab = time_dict['Dr']**2*time_dict['fr']**2*time_dict['Wr'] 
            if which in ['d1vd2v']:
                A0_tab *= time_dict['Hr'] * \
                        (1.+ 3.*time_dict['dHr']/time_dict['Hr']**2 + 4./time_dict['Hr']/time_dict['r_list'])
            elif which in ['davd1v']:
                A0_tab *= time_dict['Hr'] #*Al123(ell1, ell2, ell3)*np.sqrt(ell2*(ell2+1.)*ell3*(ell3+1.))


        try:
            A0_tab/=time_dict['r_list']**(int(which[1]) + int(which[4]))
        except ValueError:
            if which=='davd1v':
                A0_tab/=time_dict['r_list']**2

            try: 
                A0_tab/=time_dict['r_list']**(int(which[1]))
            except ValueError:
                A0_tab/=time_dict['r_list']**(int(which[4]))

        if which != 'd2vd2v': A0_tab=A0_tab[None,:]/2.
        else: A0_tab=A0_tab[None,:]

    return A0_tab, A2_tab, A4_tab, Am_tab, Il_tab


def get_cl(which, Newton, Limber, lterm, ell1, time_dict):
    Cl2_chi=np.zeros((5, 5))
    if which in ['F2', 'G2', 'dv2']:
        Cl1_chi = sum_qterm_and_linear_term(which, Newton, Limber, lterm, ell1)
    elif which == 'd2vd2v':
        Cl1_chi = sum_qterm_and_linear_term('d2v', Newton, Limber, lterm, ell1)
    else:
        Cl1_chi = sum_qterm_and_linear_term(which[:3], Newton, Limber, lterm, ell1, time_dict['r_list'], \
                time_dict['Hr'], time_dict['fr'], time_dict['Dr'], time_dict['ar'])

        Cl2_chi = sum_qterm_and_linear_term(which[3:], Newton, Limber, lterm, ell1, time_dict['r_list'], \
                time_dict['Hr'], time_dict['fr'], time_dict['Dr'], time_dict['ar'])
    return Cl1_chi, Cl2_chi


def write_all_configuration(ell1, ellmax, which, lterm, name, Limber, rad, Newton, time_dict, chi_list,\
                                    r0, ddr, normW, rmin, rmax, cp_tr, b, Nk, kmax, kmin):
    '''
        '''

    file_path = f"{name}.h5"
    lock_path = f"{file_path}.lock"
    length = int ((ellmax - ell1) * (ell1+1)) 

    with FileLock(lock_path): 
        # Create or open the HDF5 file
        with h5py.File(file_path, "a") as f:

            # Initialize or load `bl` and `wigner` datasets
            if f"bl_ell{ell1}" not in f:
                bl = f.create_dataset(f"bl_ell{ell1}", (length,), dtype='f8', fillvalue=-999)[:]
                wigner = f.create_dataset(f"wigner_ell{ell1}", (length,), dtype='f8', fillvalue=-999)[:]
            else:
                print(' resume from previous run!')
                bl = f[f"bl_ell{ell1}"][:]
                wigner = f[f"wigner_ell{ell1}"][:]


    Cl1_1_chi, Cl1_2_chi = get_cl(which, Newton, Limber, lterm, ell1, time_dict)
    A0_tab_ell1, A2_tab_ell1, A4_tab_ell1, Am_tab_ell1, Il_tab_ell1 \
        = get_kernels(ell1, Newton, rad, which, chi_list, time_dict, r0, ddr, normW, rmin,\
                            rmax, cp_tr, b, Nk, kmax, kmin)
    ind = 0
    for ell2 in range(ell1, ellmax):
        print(f'     ell2={ell2}/{ellmax}')
        a=time.time()
        Cl2_1_chi, Cl2_2_chi = get_cl(which, Newton, Limber, lterm, ell2, time_dict)
        A0_tab_ell2, A2_tab_ell2, A4_tab_ell2, Am_tab_ell2, Il_tab_ell2 \
            = get_kernels(ell2, Newton, rad, which, chi_list, time_dict, r0, ddr, normW, rmin,\
                            rmax, cp_tr, b, Nk, kmax, kmin)
    
        for ell3 in range(ell2, min(ell2+ell1+1, ellmax)):
            wigner_test = float(wigner_3j(ell1, ell2, ell3, 0,0,0))
    
            if wigner_test==0:
                bl[ind] = 0
                wigner[ind] = 0
            else:
                a=time.time()
                if ell3==ell2: Cl3_1_chi, Cl3_2_chi  = Cl2_1_chi, Cl2_2_chi
                else: Cl3_1_chi, Cl3_2_chi  = get_cl(which, Newton, Limber, lterm, ell3, time_dict)

                A0_tab_ell3, A2_tab_ell3, A4_tab_ell3, Am_tab_ell3, Il_tab_ell3 \
                    = get_kernels(ell3, Newton, rad, which, chi_list, time_dict, r0, ddr, normW, rmin,\
                            rmax, cp_tr, b, Nk, kmax, kmin)

                if 'dav' in which:
                    A0_tab_ell1_arg = A0_tab_ell1*Al123(ell1, ell2, ell3)*np.sqrt(ell2*(ell2+1.)*ell3*(ell3+1.))
                    A0_tab_ell2_arg = A0_tab_ell2*Al123(ell2, ell1, ell3)*np.sqrt(ell1*(ell1+1.)*ell3*(ell3+1.))
                    A0_tab_ell3_arg = A0_tab_ell3*Al123(ell3, ell2, ell1)*np.sqrt(ell2*(ell2+1.)*ell1*(ell1+1.))
                else:
                    A0_tab_ell1_arg = A0_tab_ell1
                    A0_tab_ell2_arg = A0_tab_ell2
                    A0_tab_ell3_arg = A0_tab_ell3

                # Skip if already computed
                if bl[ind] != -999:
                    ind += 1
                    continue
    
                evaluation=final_integrand(time_dict['r_list'][:,None], which, Newton, rad, Cl2_1_chi, Cl3_1_chi, Cl2_2_chi, Cl3_2_chi,\
                                time_dict['r_list'], A0_tab_ell1_arg, A2_tab_ell1, A4_tab_ell1, Am_tab_ell1, Il_tab_ell1)\
                          +final_integrand(time_dict['r_list'][:,None], which, Newton, rad, Cl1_1_chi, Cl3_1_chi, Cl1_2_chi, Cl3_2_chi,\
                                time_dict['r_list'], A0_tab_ell2_arg, A2_tab_ell2, A4_tab_ell2, Am_tab_ell2, Il_tab_ell2)\
                          +final_integrand(time_dict['r_list'][:,None], which, Newton, rad, Cl2_1_chi, Cl1_1_chi, Cl2_2_chi, Cl1_2_chi,\
                                time_dict['r_list'], A0_tab_ell3_arg, A2_tab_ell3, A4_tab_ell3, Am_tab_ell3, Il_tab_ell3)
    
                val=simpson(evaluation.T, x=time_dict['r_list'])

                #if which in ['G2', 'd2vd0d', 'd1vd1d', 'd1vd2v', 'd0pd1d', 'd1vd2p', 'davd1v']:
                if which in ['G2', 'd2vd0d', 'd1vd1d', 'd1vd2v', 'd1vdod', 'd0pd3v', 'davd1v']:
                    val*=-1

                # The factor 2/pi in the def of Cl is not included in general_ps. We also add here the factor 2


    
                # Store results directly into the HDF5 datasets
                bl[ind] = val*8./np.pi**2
                wigner[ind] = wigner_test
    
                #if ind % 1000 == 0:
                #    a=time.time()
                #    # Save every 1000 iterations for persistence
                #    with FileLock(lock_path): 
                #        with h5py.File(file_path, "a") as f:

                #            f[f"bl_ell{ell1}"][:]=bl
                #            f[f"wigner_ell{ell1}"][:]=wigner
                #            f.flush()  # Ensure data is written to disk
                #    print('save: ', time.time()-a)
        
            ind += 1

    with FileLock(lock_path): 
        with h5py.File(file_path, "a") as f:

            f[f"bl_ell{ell1}"][:]=bl
            f[f"wigner_ell{ell1}"][:]=wigner
 
            # Ensure all data is saved at the end
            f.flush()


def spherical_bispectrum(which, Newton, rad, Limber, lterm, ell1, ell2, ell3, time_dict, r0, ddr, normW, rmax, rmin, chi_list, cp_tr, b, Nk, kmax, kmin):
    '''
        '''

    wigner_test = float(wigner_3j(ell1, ell2, ell3, 0,0,0))

    if wigner_test==0:
        return 0., 0.
    else:

        Cl1_1_chi, Cl1_2_chi = get_cl(which, Newton, Limber, lterm, ell1, time_dict)
        Cl2_1_chi, Cl2_2_chi = get_cl(which, Newton, Limber, lterm, ell2, time_dict)
        Cl3_1_chi, Cl3_2_chi = get_cl(which, Newton, Limber, lterm, ell3, time_dict)

        A0_tab_ell1, A2_tab_ell1, A4_tab_ell1, Am_tab_ell1, Il_tab_ell1 \
            = get_kernels(ell1, Newton, rad, which, chi_list, time_dict, r0, ddr, normW, rmin,\
                            rmax, cp_tr, b, Nk, kmax, kmin)

        A0_tab_ell2, A2_tab_ell2, A4_tab_ell2, Am_tab_ell2, Il_tab_ell2 \
            = get_kernels(ell2, Newton, rad, which, chi_list, time_dict, r0, ddr, normW, rmin,\
                            rmax, cp_tr, b, Nk, kmax, kmin)

        A0_tab_ell3, A2_tab_ell3, A4_tab_ell3, Am_tab_ell3, Il_tab_ell3 \
            = get_kernels(ell3, Newton, rad, which, chi_list, time_dict, r0, ddr, normW, rmin,\
                            rmax, cp_tr, b, Nk, kmax, kmin)
        if 'dav' in which:
            A0_tab_ell1_arg = A0_tab_ell1*Al123(ell1, ell2, ell3)*np.sqrt(ell2*(ell2+1.)*ell3*(ell3+1.))
            A0_tab_ell2_arg = A0_tab_ell2*Al123(ell2, ell1, ell3)*np.sqrt(ell1*(ell1+1.)*ell3*(ell3+1.))
            A0_tab_ell3_arg = A0_tab_ell3*Al123(ell3, ell2, ell1)*np.sqrt(ell2*(ell2+1.)*ell1*(ell1+1.))
        else:
            A0_tab_ell1_arg = A0_tab_ell1
            A0_tab_ell2_arg = A0_tab_ell2
            A0_tab_ell3_arg = A0_tab_ell3

        evaluation=final_integrand(time_dict['r_list'][:,None], which, Newton, rad, Cl2_1_chi, Cl3_1_chi, Cl2_2_chi, Cl3_2_chi,\
                                time_dict['r_list'], A0_tab_ell1_arg, A2_tab_ell1, A4_tab_ell1, Am_tab_ell1, Il_tab_ell1)\
                          +final_integrand(time_dict['r_list'][:,None], which, Newton, rad, Cl1_1_chi, Cl3_1_chi, Cl1_2_chi, Cl3_2_chi,\
                                time_dict['r_list'], A0_tab_ell2_arg, A2_tab_ell2, A4_tab_ell2, Am_tab_ell2, Il_tab_ell2)\
                          +final_integrand(time_dict['r_list'][:,None], which, Newton, rad, Cl2_1_chi, Cl1_1_chi, Cl2_2_chi, Cl1_2_chi,\
                                time_dict['r_list'], A0_tab_ell3_arg, A2_tab_ell3, A4_tab_ell3, Am_tab_ell3, Il_tab_ell3)
        val=simpson(evaluation.T, x=time_dict['r_list'])

        #if which in ['G2', 'd2vd0d', 'd1vd1d', 'd1vd2v', 'd0pd1d', 'd1vd2p', 'davd1v']:
        if which in ['G2', 'd2vd0d', 'd1vd1d', 'd1vd2v', 'd1vdod', 'd0pd3v', 'davd1v']:
            val*=-1

        # The factor 2/pi in the def of Cl is not included in general_ps. We also add here the factor 2
        return val*8./np.pi**2, wigner_test
