import numpy as np
from numba import njit
import cubature

from param import *
from fftlog import *
from mathematica import *
from lincosmo import *

def sum_qterm_and_linear_term(which, Newton, lterm, ell, r_list=0, Hr=0, fr=0, Dr=0, ar=0):
    '''
    Loading the generalised power spectra and summing over qterm and lterm
    '''
    if which in ['d2p', 'd0p']:
        stuff=2./3./omega_m/H0**2
        if which=='d2p': which='d2v'
    else:
        stuff=1.

    if not isinstance(lterm, list):
        if lterm=='all':
            lterm=['density', 'rsd', 'pot', 'doppler']
            #print(' ONLY DENSITY AND RSD AS LINEAR TERM')
        elif lterm=='nopot':
            lterm=['density', 'rsd', 'doppler'] 
        else:
            lterm=[lterm]

    if which in ['F2', 'G2', 'dv2']:
        for ind, lt in enumerate(lterm):
            if lt=='pot': lt='pot_newton'
            if ind==0:
                Cl2_chi = np.loadtxt(output_dir+'cln/Cln_{}_ell{}.txt'.format(lt, int(ell)))
            else:
                Cl2_chi[:,1:] += np.loadtxt(output_dir+'cln/Cln_{}_ell{}.txt'.format(lt, int(ell)))[:,1:]

    elif which=='d0d':
        Cl2_chi=sum_qterm_and_linear_term('F2', Newton, lterm, ell)
        if not Newton:
            Cl2_chi[:,1]+=3.*np.interp(Cl2_chi[:,0], r_list, Hr)**2*np.interp(Cl2_chi[:,0], r_list, fr)\
                                    *Cl2_chi[:,2]

    elif which in ['d0p', 'dav']:
        Cl2_chi=sum_qterm_and_linear_term('F2', Newton, lterm, ell)
        Cl2_chi[:,1]=Cl2_chi[:,2]

    elif which=='dod':
        Cl2_chi=sum_qterm_and_linear_term('F2', Newton, lterm, ell)
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
            if lt=='pot': lt='pot_newton'
            try: 
                #                print(output_dir+'cln/Cln_{}_{}_ell{}.txt not found, try qterm...'.format(which, lt, int(ell)))
                if which[:2]=='d2':
                    qlist=[1, 2, 3]
                elif which[:2]=='d1':
                    qlist=[1, 2]
                elif which[:2]=='d3': 
                    qlist=[1, 2, 3, 4]
                else: 
                    print('{} not recognised'.format(which))
                    exit()
 
                for qt in qlist:
                    if ind==0:
                        Cl2_chi = np.loadtxt(output_dir+'cln/Cln_{}_qterm{}_{}_ell{}.txt'.format(which, qt, lt, int(ell)))
                    else:
                        Cl2_chi[:,1] += np.loadtxt(output_dir+'cln/Cln_{}_qterm{}_{}_ell{}.txt'.format(which, qt, lt, int(ell)))[:,1]


            except FileNotFoundError:
                if ind==0:
                    Cl2_chi = np.loadtxt(output_dir+'cln/Cln_{}_{}_ell{}.txt'.format(which, lt, int(ell)))
                else:
                    Cl2_chi[:,1] += np.loadtxt(output_dir+'cln/Cln_{}_{}_ell{}.txt'.format(which, lt, int(ell)))[:,1]
            ind+=1
        
        if which=='d1d' and not Newton:
            Cl2_chi[:,1]+=3.*np.interp(Cl2_chi[:,0], r_list, Hr)**2*np.interp(Cl2_chi[:,0], r_list, fr)\
                            *sum_qterm_and_linear_term('d1v', Newton, lterm, ell)[:,1]

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
            res=-0.5*(2.*fr-6./7.*wr)
        elif index==1:
            res=0.5*15.*Omr*fr
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
            res=-fr
        elif index==1:
            res=3.*Omr*fr
        else:
            res=np.zeros((len(r)), dtype=np.float64)
    return  res

@njit
def gamma_chi(r, index, which, Dr, fr, vr, wr, Omr):
    if which=='F2':
        if index==0:
            res=np.zeros((len(r)), dtype=np.float64)
        elif index==1:
            res=-1./2.*(-fr**2+fr  - 3.*Omr)
        elif index==1:
            res=1./4.*(18*fr**2+9.*(fr**2-fr)*Omr)
        else:
            res=np.zeros((len(r)), dtype=np.float64)
    else:
        res=np.zeros((len(r)), dtype=np.float64)
    return  res

@njit
def epsilon_chi(r, index, which, Dr, fr, vr, wr, Omr):
    if which=='F2':
        res=np.zeros((len(r)), dtype=np.float64)
    else:
        res=9./4.*Omr*fr
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
def A0_chi(r, which, Dr, fr, vr, wr, Omr, Hr, Wr):
    if which=='F2':
        return f0_nm(r, which, Dr, fr, vr, wr, Omr, Hr)*Dr**2* Wr
    else:
        return 0

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
    res[0]=(beta_chi(r, 1, which, Dr, fr, vr, wr, Omr) - alpha_chi(r, 1, which, Dr, fr, vr, wr, Omr))/2.-2.*gamma_chi(r, 1, which, Dr, fr, vr, wr, Omr)\
            + 2.*epsilon_chi(r, 1, which, Dr, fr, vr, wr, Omr)
    res[1]=(alpha_chi(r, 1, which, Dr, fr, vr, wr, Omr) - beta_chi(r, 1, which, Dr, fr, vr, wr, Omr) + 4.*epsilon_chi(r, 1, which, Dr, fr, vr, wr, Omr))/4.\
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
def integrand_Am_F2(r, chi, ell, which, r_list, \
        ar, Dr, fr, vr, wr, Omr, Hr, mathcalR, r0, ddr, normW, \
        f0_tab, fm2_tab,\
        cp_tr, bphi, Nphi, eta):
    '''
    For all multiplet n and m, this function computes the integrand of the pure relativistic \
            terms coming from the k1^-2 and k1^-4 terms of the kernel
    
    returns Am2=2/pi * D^2(r)W(r) f^{(-2)}_{nm}(r) \int dk jl(k chi)*jl(k r)
            Am4=2/pi * D^2(r)W(r) f^{(-4)}_{nm}(r) \int dk k^{-2} jl(k chi)*jl(k r)
            
    '''
    t_list=r[:,0]/chi
    Am4=np.zeros(len(t_list), dtype=np.complex128)
    Am2=np.zeros(len(t_list), dtype=np.float64)
    out=np.zeros((5, len(t_list)), dtype=np.float64)
    
    Drr=np.interp(r[:,0], r_list, Dr)
    frr=np.interp(r[:,0], r_list, fr)
    vrr=np.interp(r[:,0], r_list, vr)
    wrr=np.interp(r[:,0], r_list, wr)
    Omrr=np.interp(r[:,0], r_list, Omr)
    Hrr=np.interp(r[:,0], r_list, Hr)

    #if ell>=5: 
    #    t1min = tmin_fct(ell, -1.)
    #else: 
    #    t1min=0

    for ind,t in enumerate(t_list):
        if t>1:
            Am2[ind]=t**(-ell+1.)
            fact=t
            t=1./t
        else:
            fact=1.
            Am2[ind]=t**(ell+2.)
        Am4[ind]=chi*fact*Il(-1+0.j, t+0.j, ell) #myhyp21(-1.+0.j, t, chi, ell, t1min) #
    
    fm2=fm2_nm(r[:,0], which, Drr, frr, vrr, wrr, Omrr, Hrr)
    fm4=fm4_nm(r[:,0], which, Drr, frr, vrr, wrr, Omrr, Hrr)
    Am2=Am2/(1.+2.*ell)/r[:,0]**2
    WDr=Drr**2*W(r[:,0], r0, ddr, normW) 

    out[0]=chi*Am2*WDr*fm2[0] 
    out[1]=chi*Am2*WDr*fm2[1] 
    out[2]=chi*Am2*WDr*fm2[2] 
    out[3]=Am4.real*fm4[0]*WDr/2/np.pi**2
    out[4]=Am4.real*fm4[1]*WDr/2/np.pi**2
#    out=chi*np.array([fm2[0]*Am2, fm2[1]*Am2, fm2[2]*Am2, fm4[0]*Am4, fm4[1]*Am4])\
        #        *Drr**2*W(r[:,0], r0, ddr, normW) 
    return out.T


@njit
def integrand_Am_G2(r, chi, ell, which, r_list, \
        ar, Dr, fr, vr, wr, Omr, Hr, mathcalR, r0, ddr, normW, \
        f0_tab, fm2_tab,\
        cp_tr, bphi, Nphi, eta):
    t_list=r[:,0]/chi
    Am4=np.zeros(len(t_list), dtype=np.complex128)
    Am2=np.zeros(len(t_list), dtype=np.float64)
    out=np.zeros((5, len(t_list)), dtype=np.float64)
    
    for ind,t in enumerate(t_list):
        if t>1:
            Am2[ind]=t**(-ell+1.)
            fact=t
            t=1./t
        else:
            fact=1.
            Am2[ind]=t**(ell+2.)
        Am4[ind]=fact*(Il(-1+0.j, t+0.j, ell))

    f00=np.interp(r[:,0], r_list, f0_tab[0])
    f01=np.interp(r[:,0], r_list, f0_tab[1])
    f02=np.interp(r[:,0], r_list, f0_tab[2])

    fm20=np.interp(r[:,0], r_list, fm2_tab[0])
    fm21=np.interp(r[:,0], r_list, fm2_tab[1])

    Am2=Am2/(1.+2.*ell)/ r[:,0]**2

    out[0]=chi*f00*Am2
    out[1]=chi*f01*Am2
    out[2]=chi*f02*Am2
    out[3]=chi*fm20*Am4.real/2/np.pi**2
    out[4]=chi*fm21*Am4.real/2/np.pi**2

    #out=chi*np.array([f00*Am2, f01*Am2, f02*Am2, fm20*Am4, fm21*Am4])
    return out.T

@njit
def integrand_Il_F2(r, chi, ell, which, r_list, \
        ar, Dr, fr, vr, wr, Omr, Hr, mathcalR, r0, ddr, normW, \
        f0_tab, fm2_tab,\
        cp_tr, bphi, Nphi, eta):
    '''
    For all multiplet n and m, this function computes the integrand of the radiation term \
            \partial \log T_{phi_0} / \partial \log k = \sum_p c_p k^{b+i\eta_p}
    
    returns 2/pi * D^2(r)W(r) f^{(R)}_{nm}(r) \sum_p c_p * \int dk k**(nu_p-1) jl(k*chi)jl(k*r)
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
    Wrr=W(r[:,0], r0, ddr, normW)

    fm2=fm2R_nm(which, Drr, frr, vrr, wrr, Omrr, Hrr, arr)
    fm4=fm4R_nm(which, Drr, frr, vrr, wrr, Omrr, Hrr, arr)

    Ilm2=np.zeros(len(t_list), dtype=np.complex128)
    Ilm4=np.zeros(len(t_list), dtype=np.complex128)
    for p in range(-Nphi//2, Nphi//2+1):
        nu=1.+bphi+1j*p*eta
        if ell>=5: 
            t1min = tmin_fct(ell, nu)
        else: 
            t1min=0

        for ind, t in enumerate(t_list):
            Ilm2[ind]+=cp_tr[p+Nphi//2]*myhyp21(nu, t, chi, ell, t1min)
            Ilm4[ind]+=cp_tr[p+Nphi//2]*myhyp21(nu-2., t, chi, ell, t1min)
    # here we have computed 
    # 2pi^2/r^2 \sum c_p I_me(1.+bphi+1j*p*eta, r1, chi)

    WDr=Drr**2*Wrr
    out[0]=fm2[0]*Ilm2.real*WDr
    out[1]=fm2[1]*Ilm2.real*WDr
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
        nu=1.+bphi+1j*p*eta
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


def get_Am_and_Il(chi_list, ell1, lterm, which, Newton, rad, time_dict, r0, ddr, normW, rmin, rmax, \
        cp_tr, bphi, Nphi, kmax, kmin, save=True):
    '''
    Computes the r and k integrals of the pure relativistic terms coming from the k1^-2 and k1^-4 terms of the kernel. 
    ell1 is fixed. All multiplet of n, m are computed as function of chi
    
    if rad==False: 
        computes the relativistic terms. Those terms lead to a function of chi that can be \
        expressed as an integral over r1:

        returns integration of the function integrand_Am_F2 over r for all chi_list values and multiplication by chi^2:
            2/pi* chi^2 * \int dr D^2(r)W(r) f^{(-2)}_{nm}(r) \int dk jl(k chi)*jl(k r)
            and         
            2/pi* chi^2 * \int dr D^2(r)W(r) f^{(-4)}_{nm}(r) \int dk k^{-2} jl(k chi)*jl(k r)

    else radiation is on:
        computes the radiation term. Those terms lead to a function of chi that can be \
        expressed as an integral over r1:

        returns integration of the function integrand_Il_F2 over r for all chi_list values and multiplication by chi^2:
            2/pi* chi^2 * \int dr D^2(r)W(r) f^{(R)}_{nm}(r) \sum_p c_p * \int dk k**(nu_p-1) jl(k*chi)jl(k*r)
    
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
            integ = integrand_Il_F2
        elif which=='G2':
            integ = integrand_Il_G2
            fm2_tab = -np.gradient(np.gradient(fm2_tab, \
                            time_dict['r_list'], axis=1), time_dict['r_list'], axis=1)
            f0_tab = -np.gradient(np.gradient(f0_tab, \
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
 
        if which=='F2' :
            if not Newton:
                integ = integrand_Am_F2
            else:
                return 0
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
    
            integ = integrand_Am_G2

    res=np.zeros((fdim, len(chi_list)))
    #if ell1<100:

    for ind, chi in enumerate(chi_list):
        if ind%10==0 and rad: print('   {}/{}'.format(ind, len(chi_list)))

        val, err = cubature.cubature(integ, ndim=1, fdim=fdim, xmin=[rmin], xmax=[rmax],\
                                     args=(chi, ell1, which, time_dict['r_list'],\
                                        time_dict['ar'], time_dict['Dr'], time_dict['fr'],\
                                        time_dict['vr'], time_dict['wr'], time_dict['Omr'],\
                                        time_dict['Hr'], time_dict['mathcalR'], r0, ddr, normW,\
                                        f0_tab, fm2_tab,\
                                        cp_tr, bphi, Nphi, eta), relerr=relerr,\
                                             maxEval=1e5, abserr=0, vectorized=True)
        
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
    #    Wr=W(chi_list, r0, ddr, normW)
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
def final_integrand(chi, which, Newton, rad, Cl2n1_chi, Cl3n1_chi, Cl2n2_chi=0, Cl3n2_chi=0,\
                        r_list=0, A0_tab=0, A2_tab=0, A4_tab=0, Am_tab=0, Il_tab=0):
    '''
    The sum over n and m is now explicitely computed. The result is a function of chi which will be integrated over.

    returns 2/pi* chi^2 * \sum_{mn} C_{\ell2}^{(n)}(chi) C_{\ell3}^{(m)}(chi) 
                
                \int dr D^2(r)W(r) f^{(X)}_{nm}(r) \int dk k^(X)*jl(k chi)*jl(k r) where X is an integer
            or
                \int dr D^2(r)W(r) f^{(R)}_{nm}(r) \sum_p c_p * \int dk k**(nu_p-1) jl(k*chi)jl(k*r) for radiation
    '''

    chi=chi[:,0]
    Cl2_0 = np.interp(chi, Cl2n1_chi[:,0], Cl2n1_chi[:,1])
    Cl3_0 = np.interp(chi, Cl3n1_chi[:,0], Cl3n1_chi[:,1])

    if which in ['F2', 'G2', 'dv2']:
        if which == 'F2':
            A00 = np.interp(chi, r_list, A0_tab[0])
            A01 = np.interp(chi, r_list, A0_tab[1])
            A02 = np.interp(chi, r_list, A0_tab[2])
            A03 = np.interp(chi, r_list, A0_tab[3])
        else :
            A00, A01, A02, A03 = 0, 0, 0, 0

        Cl2_p2 = np.interp(chi, Cl2n1_chi[:,0], Cl2n1_chi[:,3])
        Cl2_m2 = np.interp(chi, Cl2n1_chi[:,0], Cl2n1_chi[:,2])
        Cl3_m2 = np.interp(chi, Cl3n1_chi[:,0], Cl3n1_chi[:,2])
        Cl3_p2 = np.interp(chi, Cl3n1_chi[:,0], Cl3n1_chi[:,3])

        if not Newton or (Newton and which!='F2'):
            Am = np.interp(chi, Am_tab[:,0], Am_tab[:,1])*Cl2_0*Cl3_0+\
                 np.interp(chi, Am_tab[:,0], Am_tab[:,2])*(Cl3_p2*Cl2_m2+Cl3_m2*Cl2_p2)+\
                 np.interp(chi, Am_tab[:,0], Am_tab[:,3])*(Cl3_0*Cl2_m2+Cl3_m2*Cl2_0)+\
                 np.interp(chi, Am_tab[:,0], Am_tab[:,4])*Cl2_0*Cl3_0 +\
                 np.interp(chi, Am_tab[:,0], Am_tab[:,5])*(Cl3_p2*Cl2_m2+Cl3_m2*Cl2_p2)
        else:
            Am=0

        A20 = np.interp(chi, r_list, A2_tab[0])
        A21 = np.interp(chi, r_list, A2_tab[1])
        A40 = np.interp(chi, r_list, A4_tab[0])

        if rad and not Newton:
            #if which=='F2':
            Il_res=np.interp(chi, Il_tab[:,0], Il_tab[:,1])*Cl2_0*Cl3_0+\
                       np.interp(chi, Il_tab[:,0], Il_tab[:,2])*(Cl3_p2*Cl2_m2+Cl3_m2*Cl2_p2)+\
                       np.interp(chi, Il_tab[:,0], Il_tab[:,3])*Cl2_0*Cl3_0+\
                       np.interp(chi, Il_tab[:,0], Il_tab[:,4])*(Cl3_p2*Cl2_m2+Cl3_m2*Cl2_p2)
            #else:
    
            #    Il_res=np.interp(chi, Il_tab[:,0], Il_tab[:,1])*Cl2_0*Cl3_0+\
            #           np.interp(chi, Il_tab[:,0], Il_tab[:,2])*(Cl3_p2*Cl2_m2+Cl3_m2*Cl2_p2)
        else:
            Il_res=0

        out = A00*Cl2_0*Cl3_0 + (A03+A21+A40)*Cl2_m2*Cl3_m2 + A01*(Cl3_p2*Cl2_m2+Cl3_m2*Cl2_p2)\
                          + (A02+A20)*(Cl2_0*Cl3_m2+Cl3_0*Cl2_m2) + Am + Il_res

    else:
        A00 = np.interp(chi, r_list, A0_tab)
        if which=='d2vd2v':
            out = A00*Cl2_0*Cl3_0 
        else:
            Cl2_0_2 = np.interp(chi, Cl2n2_chi[:,0], Cl2n2_chi[:,1])
            Cl3_0_2 = np.interp(chi, Cl3n2_chi[:,0], Cl3n2_chi[:,1])
            out = A00*(Cl2_0*Cl3_0_2+Cl3_0*Cl2_0_2)
    return out


################################################################################ spherical bispectrum

def spherical_bispectrum_perm1(which, Newton, rad, lterm, ell1, ell2, ell3, time_dict, r0, ddr, normW, rmax, rmin, chi_list, cp_tr, b, k, kmax, kmin):
    '''
    Main function to compute de bispectrum assuming the generalised power spectra have already been computed.
    Power spectra are loaded with the function sum_qterm_and_linear_term

    Relativistic effects (including radiation) are either loaded (if already computed) or computed 
    thanks to the function get_Am_and_Il 
    ...
    '''

    if which in ['F2', 'G2', 'dv2']:
        Cl2n_chi = sum_qterm_and_linear_term(which, Newton, lterm, ell2)
        Cl3n_chi = sum_qterm_and_linear_term(which, Newton, lterm, ell3)

        Il_fn='Il/Il_{}_ell{}.txt'
        if not Newton and rad:
            try:
                Il_tab = np.loadtxt(output_dir+Il_fn.format(which, int(ell1)))
            except FileNotFoundError:
                Il_tab = get_Am_and_Il(chi_list, ell1, lterm, which, Newton, True, time_dict, r0, ddr, normW, rmin,\
                                    rmax, cp_tr, b, k, kmax, kmin, True)
        else:
            Il_tab = 0 
        
        Am_fn='Am/Am_{}_ell{}{}.txt'
        if not Newton:
            Am_new=''
            Hr = time_dict['Hr']
        else:
            Am_new='_newton'
            Hr = 0
        
        if not Newton or (Newton and which!='F2'):
            try:
                Am_tab = np.loadtxt(output_dir+Am_fn.format(which, \
                    int(ell1), Am_new))
            except (ValueError, FileNotFoundError):
                Am_tab = get_Am_and_Il(chi_list, ell1, lterm, which, Newton, False, time_dict, r0, ddr, normW, rmin,\
                                    rmax, cp_tr, b, k, kmax, kmin, True)
        else:
            Am_tab = 0

        A0_tab = A0_chi(time_dict['r_list'], which,       time_dict['Dr'], time_dict['fr'], \
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

        val, err = cubature.cubature(final_integrand, ndim=1, fdim=1, xmin=[rmin], xmax=[rmax],\
                                     args=(which, Newton, rad, Cl2n_chi, Cl3n_chi, 0, 0, \
                                     time_dict['r_list'],\
                                     A0_tab, A2_tab, A4_tab, Am_tab, Il_tab), \
                                     relerr=relerr, maxEval=0, abserr=0, vectorized=True)
    else:

        if which=='d2vd2v':
            A0_tab = time_dict['Dr']**2*time_dict['fr']**2*time_dict['Wr']/time_dict['r_list']**4

            Cl2_chi = sum_qterm_and_linear_term('d2v', Newton, lterm, ell2)
            Cl3_chi = sum_qterm_and_linear_term('d2v', Newton, lterm, ell3)

            val, err = cubature.cubature(final_integrand, ndim=1, fdim=1, xmin=[rmin], xmax=[rmax],\
                                     args=(which, Newton, rad, Cl2_chi, Cl3_chi, 0, 0, time_dict['r_list'], A0_tab),\
                                     relerr=relerr, maxEval=0, abserr=0, vectorized=True)
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
                    A0_tab *= time_dict['Hr']*Al123(ell1, ell2, ell3)*np.sqrt(ell2*(ell2+1.)*ell3*(ell3+1.))

            if which in ['d2vd0d', 'd1vd1d', 'd1vd2v', 'd0pd1d', 'd1vd2p', 'davd1v']:
                A0_tab*=-1

            try:
                A0_tab/=time_dict['r_list']**(int(which[1]) + int(which[4]))
            except ValueError:
                if which=='davd1v':
                    A0_tab/=time_dict['r_list']**2

                try: 
                    A0_tab/=time_dict['r_list']**(int(which[1]))
                except ValueError:
                    A0_tab/=time_dict['r_list']**(int(which[4]))

            Cl2_1_chi = sum_qterm_and_linear_term(which[:3], Newton, lterm, ell2, time_dict['r_list'], \
                    time_dict['Hr'], time_dict['fr'], time_dict['Dr'], time_dict['ar'])

            Cl3_1_chi = sum_qterm_and_linear_term(which[:3], Newton, lterm, ell3, time_dict['r_list'], \
                    time_dict['Hr'], time_dict['fr'], time_dict['Dr'], time_dict['ar'])
                                                                                                                                                             
            Cl2_2_chi = sum_qterm_and_linear_term(which[3:], Newton, lterm, ell2, time_dict['r_list'], \
                    time_dict['Hr'], time_dict['fr'], time_dict['Dr'], time_dict['ar'])

            Cl3_2_chi = sum_qterm_and_linear_term(which[3:], Newton, lterm, ell3, time_dict['r_list'], \
                    time_dict['Hr'], time_dict['fr'], time_dict['Dr'], time_dict['ar'])

            val, err = cubature.cubature(final_integrand, ndim=1, fdim=1, xmin=[rmin], xmax=[rmax],\
                                     args=(which, Newton, rad, Cl2_1_chi, Cl3_1_chi, Cl2_2_chi, Cl3_2_chi, time_dict['r_list'], A0_tab),\
                                     relerr=relerr, maxEval=0, abserr=0, vectorized=True)
            val/=2.

    # The factor 2/pi in the def of Cl is not included in general_ps. We also add here the factor 2
    return val[0]*8./np.pi**2

def spherical_bispectrum(which, Newton, rad, lterm, ell1, ell2, ell3, time_dict, r0, ddr, normW, rmax, rmin, chi_list, cp_tr, b, k, kmax, kmin):
    '''
    Function computing the permutations
    '''
    #print(which, lterm, ell1, ell2, ell3)
    return   spherical_bispectrum_perm1(which, Newton, rad, lterm, ell1, ell2, ell3, time_dict,\
                r0, ddr, normW, rmax, rmin, chi_list, cp_tr, b, k, kmax, kmin)\
            +spherical_bispectrum_perm1(which, Newton, rad, lterm, ell2, ell1, ell3, time_dict,\
                r0, ddr, normW, rmax, rmin, chi_list, cp_tr, b, k, kmax, kmin)\
            +spherical_bispectrum_perm1(which, Newton, rad, lterm, ell3, ell2, ell1, time_dict,\
                r0, ddr, normW, rmax, rmin, chi_list, cp_tr, b, k, kmax, kmin)
