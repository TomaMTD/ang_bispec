import numpy as np
from numba import njit
import os
from mathematica import *
from lincosmo import *

#import sys, importlib
#importlib.import_module(sys.argv[-1])



############################################################################# fftlog
@njit
def get_cp(px, bx, x_list, N, xmin, xmax, fct_x):
    eta_p = 2.*np.pi*px/np.log(xmax/xmin)
    l=np.arange(N)
    return np.sum(fct_x * x_list**(-bx) * xmin**(-1j*eta_p) * np.exp(-2.*1j*np.pi*px*l/N)) / N


############################################################################# get fftlog coef
def mathcalD(x, y, ell):
    dy=np.gradient(y, x, axis=1)
    return -np.gradient(dy, x, axis=1)+2./x*dy+(ell*(ell+1)-2.)/x**2*y

def set_bias(gauge, lterm, which, qterm, rad):

    if not rad:
        if which=='FG2':
            if gauge=='new' and lterm =='density':
                b=-0.9
            else:
                b=-0.3
        else:
            if which in ['d2v', 'd1v', 'd3v']:
                if lterm=='density':
                    b=-0.9
                else:
                    if qterm==1:
                        b=-2.1
                    elif qterm==2:
                        b=-2.3
                    else:
                        b=-2.5

            elif which=='d1d':
                if lterm=='density':
                    if qterm==1:
                        b=0.8
                    else:
                        b=0.9
                else:
                    b=-0.5

            elif which=='d0d':
                if lterm=='density':
                    b=1.1 #0.9
                else:
                    b=-0.5 #-0.3

            else: print('!!!!!!!!!!!!!!! bias not set')
 
    else:
        if which == 'F2':
            b = 0.3
        else:
            b = 0.6

    print(' bias: {}'.format(b))
    return b
 


def mathcalB(k, Pk, r_list, gauge, lterm, which, rad, ra, a, Ha, D, f, r0, ddr, normW):
    if not rad:
        Wa = W(ra, r0, ddr, normW)
        fr = np.interp(r_list, ra, f)
        Hr = np.interp(r_list, ra, Ha)
        WDr = np.interp(r_list, ra, D*Wa)

        if lterm in ['density', 'den']:
            if gauge in ['sync']:
                out=Pk*WDr[:,None]
            else:
                out=Pk*k**2*WDr[:,None]*(k**2+3.*fr[:,None]*Hr[:,None]**2) 

        elif lterm=='rsd':
            ggDWf_a = np.gradient(np.gradient(D*f*Wa, ra), ra)
            ggDWf = np.interp(r_list, ra, ggDWf_a)
            if gauge=='sync':
                out=-Pk*ggDWf[:,None]
            else:
                out=-Pk*k**4*ggDWf[:,None]

        elif lterm in ['pot', 'potential']:
            mathcalR=np.interp(r_list, ra, dotH_(1./a-1.))/Hr**2 + 2./Hr/r_list
            dHr = (-1. + mathcalR) / np.interp(r_list, ra, a)
            out=-Pk*k**4*WDr[:,None]*dHr[:,None]
        
        elif lterm in ['doppler', 'dop']:
            mathcalR=np.interp(r_list, ra, dotH_(1./a-1.))/Hr**2 + 2./Hr/r_list
            gDWfR = np.gradient(WDr*Hr*fr*mathcalR, r_list)

            out=Pk*k**4*gDWfR[:,None]
    else:
        Hr = np.interp(r_list, ra, Ha)
        if which=='F2':
            out=Pk
        else:
            out=Pk/(1.+3.*Hr[:,None]**2/k**2)
    return out


def quadratic_terms(qterm, k, Pk, r_list, gauge, lterm, which, ra, a, Ha, D, f, r0, ddr, normW):
    B=mathcalB(k, Pk, r_list, gauge, lterm, which, False, ra, a, Ha, D, f, r0, ddr, normW)
    
    #print('     qterm={}'.format(qterm))
    if which=='d2v':
        if lterm=='density':
            B*=k**2

        if qterm==1:
            deriv1_k  = np.gradient(B, k, axis=1)
            deriv2_k  = np.gradient(deriv1_k, k, axis=1)
            out=deriv2_k
        elif qterm==2:
            deriv1_k  = np.gradient(B, k, axis=1)
            deriv2_kr = np.gradient(r_list[:,None]*deriv1_k, r_list, axis=0)
            out=-2./k*deriv2_kr
        else:
            deriv2_r  = np.gradient(np.gradient(r_list[:,None]**2*B, r_list, axis=0), r_list, axis=0)
            out=deriv2_r/k**2

    elif which in ['d1v', 'd1d']:
        if lterm=='density':
            B*=k
        else:
            B/=k

        if which=='d1d':
            B*=k**2 #+3.*np.interp(r_list[:,None], ra, Ha)**2*np.interp(r_list[:,None], ra, f)

        if qterm==1:
            deriv1_k =np.gradient(B, k, axis=1)
            out=deriv1_k
        else:
            deriv1_r  = np.gradient(r_list[:,None]*B, r_list, axis=0)
            out=-deriv1_r/k

    elif which=='d3v':
        if lterm=='density':
            B*=k**3
        else:
            B*=k

        if qterm==1:
            deriv3_k = np.gradient(np.gradient(np.gradient(B, k, axis=1), k, axis=1) , k, axis=1)
            out=deriv3_k
        elif qterm==2:
            deriv2_k = np.gradient(np.gradient(B, k, axis=1), k, axis=1)
            deriv2_kr = np.gradient(r_list[:,None]*deriv2_k, r_list, axis=0)
            out=-3./k*deriv2_kr
        elif qterm==3:
            deriv1_k = np.gradient(B, k, axis=1)
            deriv2_kr = np.gradient(np.gradient(r_list[:,None]**2*deriv1_k, r_list, axis=0), r_list, axis=0)
            out=3./k**2*deriv2_kr
        else:
            deriv3_r  = np.gradient(np.gradient(np.gradient(r_list[:,None]**3*B, r_list, axis=0), r_list, axis=0), r_list, axis=0)
            out=-deriv3_r/k**3

    elif which=='d0d':
        #if lterm=='density':
        #    B*=k
        if lterm!='density':
            B/=k**2

        #out=B*(k**2+3.*np.interp(r_list[:,None], ra, Ha)**2*np.interp(r_list[:,None], ra, f))
        out=B*k**2

    return out


@njit
def compute(r_list, k, Pk, fct_rk, b):
    Nk = len(k)
    kmin, kmax = np.min(k), np.max(k)
    res=np.zeros((Nk+1, len(r_list)), dtype=np.complex128)
    for ind, r in enumerate(r_list):
        for p in range(-Nk//2, Nk//2+1):
            res[p+Nk//2, ind] = get_cp(p, b, k, Nk, kmin, kmax, fct_rk[ind])

    return res

def get_cp_of_r(r_list, k, Pk, gauge, lterm, which, qterm, rad, ra, a, Ha, D, f, r0, ddr, normW, b):

    if which=='FG2': # not in ['d2v', 'd1v', 'd3v', 'd1d', 'd0d']:
        fact_rk = mathcalB(k, Pk, r_list, gauge, lterm, which, rad, ra, a, Ha, D, f, r0, ddr, normW) 
        np.save(output_dir+'fct_kr'.format(which, lterm, qterm), fact_rk)
    else: 
        fact_rk = quadratic_terms(qterm, k, Pk, r_list, gauge, lterm, which, ra, a, Ha, D, f, r0, ddr, normW) 
        np.save(output_dir+'fct_kr_{}_lterm{}_qterm{}'.format(which, lterm, qterm), fact_rk)
        
    return compute(r_list, k, Pk, fact_rk, b)























