import numpy as np
from numba import njit
import cubature

from param import *
from fftlog import *
from mathematica import *
from lincosmo import *

def sum_qterm_and_linear_term(which, lterm, ell, r_list=0, Hr=0, fr=0, Dr=0, ar=0):
    if which in ['d2p', 'd0p']:
        stuff=2./3./omega_m/H0**2
        if which=='d2p': which='d2v'
    else:
        stuff=1.

    if not isinstance(lterm, list):
        if lterm=='all':
            lterm=['density', 'rsd', 'doppler', 'pot']
            #print(' ONLY DENSITY AND RSD AS LINEAR TERM')
        else:
            lterm=[lterm]

    if which in ['F2', 'G2', 'dv2']:
        for ind, lt in enumerate(lterm):
            if ind==0:
                Cl2_chi = np.loadtxt(output_dir+'cln/Cln_{}_ell{}.txt'.format(lt, int(ell)))
            else:
                Cl2_chi[:,1:] += np.loadtxt(output_dir+'cln/Cln_{}_ell{}.txt'.format(lt, int(ell)))[:,1:]

    elif which=='d0d':
        Cl2_chi=sum_qterm_and_linear_term('F2', lterm, ell)
        Cl2_chi[:,1]+=3.*np.interp(Cl2_chi[:,0], r_list, Hr)**2*np.interp(Cl2_chi[:,0], r_list, fr)\
                            *Cl2_chi[:,2]

    elif which in ['d0p', 'dav']:
        Cl2_chi=sum_qterm_and_linear_term('F2', lterm, ell)
        Cl2_chi[:,1]=Cl2_chi[:,2]

    elif which=='dod':
        Cl2_chi=sum_qterm_and_linear_term('F2', lterm, ell)
        ff=np.interp(Cl2_chi[:,0], r_list, fr)
        HH=np.interp(Cl2_chi[:,0], r_list, Hr)
        Cl2_chi[:,1]=HH*np.interp(Cl2_chi[:,0], r_list, Dr) * \
                        (ff * Cl2_chi[:,1] + 3.*(ff*np.interp(Cl2_chi[:,0], r_list, dotH_(1./ar-1.))+\
                        HH**2*(3./2.*np.interp(Cl2_chi[:,0], r_list, Om_(1./ar-1.))-ff))*Cl2_chi[:,2])

    else:
        ind=0
        for lt in lterm:
            try: 
                if ind==0:
                    Cl2_chi = np.loadtxt(output_dir+'cln/Cln_{}_{}_ell{}.txt'.format(which, lt, int(ell)))
                else:
                    Cl2_chi[:,1] += np.loadtxt(output_dir+'cln/Cln_{}_{}_ell{}.txt'.format(which, lt, int(ell)))[:,1]
            except FileNotFoundError:
                print(output_dir+'cln/Cln_{}_{}_ell{}.txt not found, try qterm...'.format(which, lt, int(ell)))
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
            ind+=1
        
        if which=='d1d':
            Cl2_chi[:,1]+=3.*np.interp(Cl2_chi[:,0], r_list, Hr)**2*np.interp(Cl2_chi[:,0], r_list, fr)\
                            *sum_qterm_and_linear_term('d1v', lterm, ell)[:,1]

    #np.savetxt(output_dir+'cl_{}_{}_ell{}.txt'.format(which, lterm[0], int(ell)), Cl2_chi)
    return Cl2_chi/stuff


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
            res=2.*fr-6./7.*wr
        elif index==1:
            res=-15.*Omr*fr
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
            res=2*fr
        elif index==1:
            res=-6.*Omr*fr
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
        res=-9./2.*Omr*fr
    return  res

############################################################################# definition of fnm as function of chi
#@njit
def f0_nm(r, which, Dr, fr, vr, wr, Omr, Hr):
    res=np.zeros((4, len(r)), dtype=np.float64)
    res[0]= (beta_chi(r, 0, which, Dr, fr, vr, wr, Omr) - alpha_chi(r, 0, which, Dr, fr, vr, wr, Omr))/2.
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
        y*=Hr*mathcalR
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
        y*=Hr*mathcalR
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

@njit
def integrand_Am_F2(r, chi, ell, which, Cl2_m2, Cl2_0, Cl2_p2, Cl3_m2, Cl3_0, Cl3_p2, r_list, Dr, fr, vr, wr, Omr, Hr, r0, ddr, normW):
    t_list=r[:,0]/chi
    Am4=np.zeros(len(t_list), dtype=np.float64)
    Am2=np.zeros(len(t_list), dtype=np.float64)
    
    Drr=np.interp(r[:,0], r_list, Dr)
    frr=np.interp(r[:,0], r_list, fr)
    vrr=np.interp(r[:,0], r_list, vr)
    wrr=np.interp(r[:,0], r_list, wr)
    Omrr=np.interp(r[:,0], r_list, Omr)
    Hrr=np.interp(r[:,0], r_list, Hr)

    for ind,t in enumerate(t_list):
        if t>1:
            Am2[ind]=t**(-ell+1.)
            fact=t
            t=1./t
        else:
            fact=1.
            Am2[ind]=t**(ell+2.)
        Am4[ind]=fact*(Il(-1+0.j, t+0.j, ell)).real/4/np.pi
    
    fm2=fm2_nm(r[:,0], which, Drr, frr, vrr, wrr, Omrr, Hrr)
    fm4=fm4_nm(r[:,0], which, Drr, frr, vrr, wrr, Omrr, Hrr)
    out=chi*((fm2[0]*Cl2_0*Cl3_0 + fm2[1]*(Cl3_p2*Cl2_m2+Cl3_m2*Cl2_p2) + fm2[2]*(Cl3_0*Cl2_m2+Cl3_m2*Cl2_0))*Am2/(1.+2.*ell)/r[:,0]**2\
        +(fm4[0]*Cl2_0*Cl3_0 + fm4[1]*(Cl3_p2*Cl2_m2+Cl3_m2*Cl2_p2))*Am4)\
        *Drr**2*W(r[:,0], r0, ddr, normW) 
    return out


@njit
def integrand_Am_G2(r, chi, ell, which, Cl2_m2, Cl2_0, Cl2_p2, Cl3_m2, Cl3_0, Cl3_p2, r_list, f0_tab, fm2_tab):
    t_list=r[:,0]/chi
    Am4=np.zeros(len(t_list), dtype=np.float64)
    Am2=np.zeros(len(t_list), dtype=np.float64)
    
    for ind,t in enumerate(t_list):
        if t>1:
            Am2[ind]=t**(-ell+1.)
            fact=t
            t=1./t
        else:
            fact=1.
            Am2[ind]=t**(ell+2.)
        Am4[ind]=fact*(Il(-1+0.j, t+0.j, ell)).real/4/np.pi

    f00=np.interp(r[:,0], r_list, f0_tab[0])
    f01=np.interp(r[:,0], r_list, f0_tab[1])
    f02=np.interp(r[:,0], r_list, f0_tab[2])

    fm20=np.interp(r[:,0], r_list, fm2_tab[0])
    fm21=np.interp(r[:,0], r_list, fm2_tab[1])

    out=chi*((f00*Cl2_0*Cl3_0 + f01*(Cl3_p2*Cl2_m2+Cl3_m2*Cl2_p2) + f02*(Cl3_0*Cl2_m2+Cl3_m2*Cl2_0))*Am2/(1.+2.*ell)/ r[:,0]**2\
            +(fm20*Cl2_0*Cl3_0 + fm21*(Cl3_p2*Cl2_m2+Cl3_m2*Cl2_p2))*Am4)
        
    return out


def get_Am(chi_list, ell1, ell2, ell3, which, lterm, time_dict, r0, ddr, normW, rmin, rmax):
    Cl2n_chi = sum_qterm_and_linear_term(which, lterm, ell2)
    Cl3n_chi = sum_qterm_and_linear_term(which, lterm, ell3)

    Am_fn = 'Am/Am_{}_{}_ell{},{},{}.txt'

    res=np.zeros((len(chi_list)))
    
    if which=='F2':
        for ind, chi in enumerate(chi_list):
            Cl2_m2 = np.interp(chi, Cl2n_chi[:,0], Cl2n_chi[:,2])
            Cl2_0 = np.interp (chi, Cl2n_chi[:,0], Cl2n_chi[:,1])
            Cl2_p2 = np.interp(chi, Cl2n_chi[:,0], Cl2n_chi[:,3])

            Cl3_m2 = np.interp(chi, Cl3n_chi[:,0], Cl3n_chi[:,2])
            Cl3_0 = np.interp (chi, Cl3n_chi[:,0], Cl3n_chi[:,1])
            Cl3_p2 = np.interp(chi, Cl3n_chi[:,0], Cl3n_chi[:,3])
 
            val, err = cubature.cubature(integrand_Am_F2, ndim=1, fdim=1, xmin=[rmin], xmax=[rmax],\
                                         args=(chi, ell1, which, Cl2_m2, Cl2_0, Cl2_p2, Cl3_m2, Cl3_0, Cl3_p2,\
                                         time_dict['r_list'], time_dict['Dr'], time_dict['fr'], time_dict['vr'],\
                                         time_dict['wr'], time_dict['Omr'], time_dict['Hr'], r0, ddr, normW)\
                                         , relerr=relerr, maxEval=0, abserr=0, vectorized=True)
            res[ind]=val*chi**2
        np.savetxt(output_dir+Am_fn.format(lterm, which, int(ell1), int(ell2), int(ell3)), np.vstack([chi_list, res]).T) 

    else:

        f0_tab=f0_nm(time_dict['r_list'], which, time_dict['Dr'],\
                time_dict['fr'], time_dict['vr'], time_dict['wr'],\
                time_dict['Omr'], time_dict['Hr'])*time_dict['Dr']**2*time_dict['Wr']
        fm2_tab=fm2_nm(time_dict['r_list'], which, time_dict['Dr'],\
                time_dict['fr'], time_dict['vr'], time_dict['wr'],\
                time_dict['Omr'], time_dict['Hr'])*time_dict['Dr']**2*time_dict['Wr']

        if which=='G2':
            f0_tab =np.gradient(np.gradient(f0_tab, time_dict['r_list'], axis=1), time_dict['r_list'], axis=1)
            fm2_tab=np.gradient(np.gradient(fm2_tab, time_dict['r_list'], axis=1), time_dict['r_list'], axis=1)
        else:
            f0_tab *=time_dict['mathcalR']*time_dict['Hr']
            fm2_tab*=time_dict['mathcalR']*time_dict['Hr']
            f0_tab =np.gradient(f0_tab, time_dict['r_list'], axis=1)
            fm2_tab=np.gradient(fm2_tab, time_dict['r_list'], axis=1)

        for ind, chi in enumerate(chi_list):
            Cl2_m2 = np.interp(chi, Cl2n_chi[:,0], Cl2n_chi[:,2])
            Cl2_0 = np.interp (chi, Cl2n_chi[:,0], Cl2n_chi[:,1])
            Cl2_p2 = np.interp(chi, Cl2n_chi[:,0], Cl2n_chi[:,3])

            Cl3_m2 = np.interp(chi, Cl3n_chi[:,0], Cl3n_chi[:,2])
            Cl3_0 = np.interp (chi, Cl3n_chi[:,0], Cl3n_chi[:,1])
            Cl3_p2 = np.interp(chi, Cl3n_chi[:,0], Cl3n_chi[:,3])
 
            val, err = cubature.cubature(integrand_Am_G2, ndim=1, fdim=1, xmin=[rmin], xmax=[rmax],\
                                         args=(chi, ell1, which, Cl2_m2, Cl2_0, Cl2_p2, Cl3_m2, Cl3_0, Cl3_p2,\
                                         time_dict['r_list'], f0_tab, fm2_tab)\
                                         , relerr=relerr, maxEval=0, abserr=0, vectorized=True)
            res[ind]=val*chi**2
        np.savetxt(output_dir+Am_fn.format(lterm, which, int(ell1), int(ell2), int(ell3)), np.vstack([chi_list, res]).T) 

    return res


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
    res[0]= 3.*fr*(fr+3.*Omr/2.)
    res[1]=-res[0]/2.
    return res*Hr**4 * Dr/ar


@njit
def integrand_Il(r, chi, ell, which, Cl2_m2, Cl2_0, Cl2_p2, Cl3_m2, Cl3_0, Cl3_p2, ar, r_list, Dr, fr, vr, wr, Omr, Hr, r0, ddr, normW, cp_tr, bphi, Nphi, eta):
    t_list=r[:,0]/chi
    
    ar=np.interp(r[:,0],  r_list, ar)
    Dr=np.interp(r[:,0],  r_list, Dr)
    fr=np.interp( r[:,0], r_list, fr)
    vr=np.interp( r[:,0], r_list, vr)
    wr=np.interp( r[:,0], r_list, wr)
    Omr=np.interp(r[:,0], r_list, Omr)
    Hr=np.interp( r[:,0], r_list, Hr)

    fm2=fm2R_nm(which, Dr, fr, vr, wr, Omr, Hr, ar)
    fm4=fm4R_nm(which, Dr, fr, vr, wr, Omr, Hr, ar)

    Ilm4=np.zeros(len(t_list), dtype=np.complex128)
    Ilm2=np.zeros(len(t_list), dtype=np.complex128)

    for p in range(-Nphi//2, Nphi//2+1):
        nu=bphi+1j*p*eta
        if ell>=5: 
            t1min = tmin_fct(ell, nu)
        else: 
            t1min=0
        
        if which=='F2':
            cp=np.ones((len(t_list)))*cp_tr[p+Nphi//2]
        else:
            cp=np.interp(r[:,0], r_list, cp_tr[p+Nphi//2])

        for ind, t in enumerate(t_list):
            Ilm2[ind]+=cp[ind]*myhyp21(nu, t, chi, ell, t1min)
            Ilm4[ind]+=cp[ind]*myhyp21(nu-2., t, chi, ell, t1min)

    out=((fm2[0]*Cl2_0*Cl3_0 + fm2[1]*(Cl3_p2*Cl2_m2+Cl3_m2*Cl2_p2))*Ilm2\
        +(fm4[0]*Cl2_0*Cl3_0 + fm4[1]*(Cl3_p2*Cl2_m2+Cl3_m2*Cl2_p2))*Ilm4)\
            *Dr**2*W(r[:,0], r0, ddr, normW)
    
    return np.real(out)


def get_Il(chi_list, ell1, ell2, ell3, which, time_dict, r0, ddr, normW, rmin, rmax, cp_tr, bphi, Nphi, kmax, kmin):

    Cl2n_chi = sum_qterm_and_linear_term(which, lterm, ell2)
    Cl3n_chi = sum_qterm_and_linear_term(which, lterm, ell3)

    Il_fn = 'Il/Il_{}_{}_ell{},{},{}.txt'

    res=np.zeros((len(chi_list)))
    eta=2.*np.pi/np.log(kmax/kmin)


    for ind, chi in enumerate(chi_list):
        if ind%10==0: print(' {}/{}'.format(ind, len(chi_list)))
        Cl2_m2 = np.interp(chi, Cl2n_chi[:,0], Cl2n_chi[:,2])
        Cl2_0 = np.interp (chi, Cl2n_chi[:,0], Cl2n_chi[:,1])
        Cl2_p2 = np.interp(chi, Cl2n_chi[:,0], Cl2n_chi[:,3])

        Cl3_m2 = np.interp(chi, Cl3n_chi[:,0], Cl3n_chi[:,2])
        Cl3_0 = np.interp (chi, Cl3n_chi[:,0], Cl3n_chi[:,1])
        Cl3_p2 = np.interp(chi, Cl3n_chi[:,0], Cl3n_chi[:,3])

        val, err = cubature.cubature(integrand_Il, ndim=1, fdim=1, xmin=[rmin], xmax=[rmax],\
                                     args=(chi, ell1, which, Cl2_m2, Cl2_0, Cl2_p2, Cl3_m2, Cl3_0, Cl3_p2,\
                                     time_dict['ar'], time_dict['r_list'], time_dict['Dr'], time_dict['fr'], time_dict['vr'], time_dict['wr'], 
                                     time_dict['Omr'], time_dict['Hr'],
                                     r0, ddr, normW, cp_tr, bphi, Nphi, eta), relerr=relerr, maxEval=1e5, abserr=0, vectorized=True)
        res[ind]=val*chi**2*2./np.pi/4./np.pi
        np.savetxt(output_dir+Il_fn.format(lterm, which, int(ell1), int(ell2), int(ell3)), np.vstack([chi_list, res]).T) 
    return res

################################################################################ spherical bispectrum
def final_integrand(r, which, Cl2n1_chi, Cl3n1_chi, Cl2n2_chi=0, Cl3n2_chi=0,\
                        r_list=0, A0_tab=0, A2_tab=0, A4_tab=0, Am_tab=0, Il_tab=0):
    r=r[:,0]
    Cl2_0 = np.interp(r, Cl2n1_chi[:,0], Cl2n1_chi[:,1])
    Cl3_0 = np.interp(r, Cl3n1_chi[:,0], Cl3n1_chi[:,1])

    if which in ['F2', 'G2', 'dv2']:
        if which == 'F2':
            A00 = np.interp(r, r_list, A0_tab[0])
            A01 = np.interp(r, r_list, A0_tab[1])
            A02 = np.interp(r, r_list, A0_tab[2])
            A03 = np.interp(r, r_list, A0_tab[3])
        else :
            A00, A01, A02, A03 = 0, 0, 0, 0

        Cl2_p2 = np.interp(r, Cl2n1_chi[:,0], Cl2n1_chi[:,3])
        Cl2_m2 = np.interp(r, Cl2n1_chi[:,0], Cl2n1_chi[:,2])
        Cl3_m2 = np.interp(r, Cl3n1_chi[:,0], Cl3n1_chi[:,2])
        Cl3_p2 = np.interp(r, Cl3n1_chi[:,0], Cl3n1_chi[:,3])

        A20 = np.interp(r, r_list, A2_tab[0])
        A21 = np.interp(r, r_list, A2_tab[1])
        A40 = np.interp(r, r_list, A4_tab[0])

        Am = np.interp(r, Am_tab[:,0], Am_tab[:,1])
        if rad:
            Il_res = np.interp(r, Il_tab[:,0], Il_tab[:,1])
        else:
            Il_res = 0

        out = A00*Cl2_0*Cl3_0 + (A03+A21+A40)*Cl2_m2*Cl3_m2 + A01*(Cl3_p2*Cl2_m2+Cl3_m2*Cl2_p2)\
                  + (A02+A20)*(Cl2_0*Cl3_m2+Cl3_0*Cl2_m2) + Am + Il_res
    else:
        A00 = np.interp(r, r_list, A0_tab)
        if which=='d2vd2v':
            out = A00*Cl2_0*Cl3_0 #/r**4
        else:
            Cl2_0_2 = np.interp(r, Cl2n2_chi[:,0], Cl2n2_chi[:,1])
            Cl3_0_2 = np.interp(r, Cl3n2_chi[:,0], Cl3n2_chi[:,1])
            out = A00*(Cl2_0*Cl3_0_2+Cl3_0*Cl2_0_2)

            #if which in ['d1vd3v', 'd0pd3v']:
            #    out = A00*(Cl2_0*Cl3_0_2+Cl3_0*Cl2_0_2)/r**4
            #elif which=='d1vd2v':
            #    out = A00*(Cl2_0*Cl3_0_2+Cl3_0*Cl2_0_2)/r**3
            #elif which=='d1vdod':
            #    out = A00*(Cl2_0*Cl3_0_2+Cl3_0*Cl2_0_2)/r
            #else:
            #    out = A00*(Cl2_0*Cl3_0_2+Cl3_0*Cl2_0_2)/r**2

    return out


################################################################################ spherical bispectrum

def spherical_bispectrum_perm1(which, lterm, ell1, ell2, ell3, time_dict, rmax, rmin):

    if which in ['F2', 'G2', 'dv2']:
        Cl2n_chi = sum_qterm_and_linear_term(which, lterm, ell2)
        Cl3n_chi = sum_qterm_and_linear_term(which, lterm, ell3)

        Am_fn='Am/Am_{}_{}_ell{},{},{}.txt'
        Il_fn='Il/Il_{}_{}_ell{},{},{}.txt'

        if rad: 
            Il_tab = np.loadtxt(output_dir+Il_fn.format(lterm, which, int(ell1), int(ell2), int(ell3)))
        else:
            Il_tab = 0 

        Am_tab = np.loadtxt(output_dir+Am_fn.format(lterm, which, int(ell1), int(ell2), int(ell3)))
        A0_tab = A0_chi(time_dict['r_list'], which,       time_dict['Dr'], time_dict['fr'], time_dict['vr'], time_dict['wr'], time_dict['Omr'], time_dict['Hr'], time_dict['Wr'])
        A2_tab = A2_chi(time_dict['r_list'], which, ell1, time_dict['Dr'], time_dict['fr'], time_dict['vr'], time_dict['wr'], time_dict['Omr'], time_dict['Hr'], time_dict['Wr'], time_dict['mathcalR'])
        A4_tab = A4_chi(time_dict['r_list'], which, ell1, time_dict['Dr'], time_dict['fr'], time_dict['vr'], time_dict['wr'], time_dict['Omr'], time_dict['Hr'], time_dict['Wr'], time_dict['mathcalR'])

        val, err = cubature.cubature(final_integrand, ndim=1, fdim=1, xmin=[rmin], xmax=[rmax],\
                                     args=(which, Cl2n_chi, Cl3n_chi, 0, 0, time_dict['r_list'], A0_tab, A2_tab, A4_tab, Am_tab, Il_tab), \
                                     relerr=relerr, maxEval=0, abserr=0, vectorized=True)
    else:

        if which=='d2vd2v':
            A0_tab = time_dict['Dr']**2*time_dict['fr']**2*time_dict['Wr']/time_dict['r_list']**4

            Cl2_chi = sum_qterm_and_linear_term('d2v', lterm, ell2)
            Cl3_chi = sum_qterm_and_linear_term('d2v', lterm, ell3)

            txt = final_integrand((Cl2_chi[:,0])[:,None] , which, Cl2_chi, Cl3_chi, 0, 0, time_dict['r_list'], A0_tab)
            np.savetxt('d2vd2v.txt', np.vstack([Cl2_chi[:,0], txt]).T )

            val, err = cubature.cubature(final_integrand, ndim=1, fdim=1, xmin=[rmin], xmax=[rmax],\
                                     args=(which, Cl2_chi, Cl3_chi, 0, 0, time_dict['r_list'], A0_tab),\
                                     relerr=relerr, maxEval=0, abserr=0, vectorized=True)
        else:
            if which in ['d2vd0d', 'd1vd1d', 'd1vd0d', 'd1vdod']:
                A0_tab = time_dict['Dr']**2*time_dict['fr']*time_dict['Wr'] 
                if which in ['d1vd0d']:
                    A0_tab*=time_dict['mathcalR'] 

            elif which in ['d0pd3v', 'd0pd1d', 'd1vd2p']:
                A0_tab = time_dict['Dr']**2*time_dict['Wr']/time_dict['Hr']/time_dict['ar']
                if 'v' in which:
                    A0_tab*=time_dict['fr']
            else:
                A0_tab = time_dict['Dr']**2*time_dict['fr']**2*time_dict['Wr'] 
                if which in ['d1vd2v']:
                    A0_tab *= time_dict['Hr'] * \
                            (1.+time_dict['dHr']/time_dict['Hr']**2 + 4./time_dict['Hr']/time_dict['r_list'])
                elif which in ['davd1v']:
                    A0_tab *= 2.*time_dict['Hr']*Al123(ell1, ell2, ell3)*np.sqrt(ell2*(ell2+1.)*ell3*(ell3+1.))

            try:
                A0_tab/=time_dict['r_list']**(int(which[1]) + int(which[4]))
            except ValueError:
                try: 
                    A0_tab/=time_dict['r_list']**(int(which[1]))
                except ValueError:
                    A0_tab/=time_dict['r_list']**(int(which[4]))

            Cl2_1_chi = sum_qterm_and_linear_term(which[:3], lterm, ell2, time_dict['r_list'], time_dict['Hr'], time_dict['fr'], time_dict['Dr'], time_dict['ar'])
            Cl3_1_chi = sum_qterm_and_linear_term(which[:3], lterm, ell3, time_dict['r_list'], time_dict['Hr'], time_dict['fr'], time_dict['Dr'], time_dict['ar'])
                                                                                                                                                             
            Cl2_2_chi = sum_qterm_and_linear_term(which[3:], lterm, ell2, time_dict['r_list'], time_dict['Hr'], time_dict['fr'], time_dict['Dr'], time_dict['ar'])
            Cl3_2_chi = sum_qterm_and_linear_term(which[3:], lterm, ell3, time_dict['r_list'], time_dict['Hr'], time_dict['fr'], time_dict['Dr'], time_dict['ar'])

            val, err = cubature.cubature(final_integrand, ndim=1, fdim=1, xmin=[rmin], xmax=[rmax],\
                                     args=(which, Cl2_1_chi, Cl3_1_chi, Cl2_2_chi, Cl3_2_chi, time_dict['r_list'], A0_tab),\
                                     relerr=relerr, maxEval=0, abserr=0, vectorized=True)
            val/=2.
    return val[0]*8./np.pi**2


def spherical_bispectrum(which, lterm, ell1, ell2, ell3, time_dict, rmax, rmin):
    return   spherical_bispectrum_perm1(which, lterm, ell1, ell2, ell3, time_dict, rmax, rmin)\
            +spherical_bispectrum_perm1(which, lterm, ell2, ell1, ell3, time_dict, rmax, rmin)\
            +spherical_bispectrum_perm1(which, lterm, ell3, ell2, ell1, time_dict, rmax, rmin)



































