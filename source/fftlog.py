import numpy as np
from numba import njit
import os
from mathematica import *
from lincosmo import *
from param import *

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

def set_bias(k, fctk):
    b, ind = 0, 0
    for i in range(5, 16):
        b+=(np.log(np.abs(fctk[-i])) - np.log(np.abs(fctk[i]))) / (np.log(k[-i]) - np.log(k[i]))
        ind+=1
    b/=ind
    print(' bias: {:.2f}'.format(b))
    return b

#def set_bias(gauge, lterm, which, qterm, rad):
#    if not rad:
#        if which in ['FG2', 'F2', 'G2']:
#            b=-0.5
#        else:
#            if which in ['d2v', 'd1v', 'd3v']:
#                if lterm=='density':
#                    b=-0.9
#                else:
#                    if qterm==1:
#                        b=-2.1
#                    elif qterm==2:
#                        b=-2.3
#                    else:
#                        b=-2.5
#
#            elif which=='d1d':
#                if lterm=='density':
#                    if qterm==1:
#                        b=0.8
#                    else:
#                        b=0.9
#                else:
#                    b=-0.5
#
#            elif which=='d0d':
#                if lterm=='density':
#                    b=1.1 #0.9
#                else:
#                    b=-0.5 #-0.3
#
#            else: print('!!!!!!!!!!!!!!! bias not set')
# 
#    else:
#        if which == 'F2':
#            b = 0.3
#        else:
#            b = 0.6
#
#    print(' bias: {}'.format(b))
#    return b
 
def P_of_k(k, Pk, gauge, rad): #, which, rad, time_dict, r0, ddr, normW):
    if not rad:
        if gauge in ['sync']:
            out=Pk 
        else:
            out=Pk*k**4 

    else:
        out=Pk
    return out

def quadratic_terms(qterm, k, Pk, gauge, lterm, which, time_dict, r0, ddr, normW):
    B=P_of_k(k, Pk, gauge, rad) #, which, 0, time_dict, r0, ddr, normW)
    
    #print('     qterm={}'.format(qterm))
    if which=='d2v':
        if lterm=='density':
            B*=k**2

        if qterm==1:
            deriv1_k  = np.gradient(np.gradient(B, k, axis=0, edge_order =2)
                                                 , k, axis=0, edge_order =2)
            out=deriv1_k
        elif qterm==2:
            deriv1_k  = np.gradient(B, k, axis=0, edge_order =2)
            out=-2./k*deriv1_k
        else:
            out=B/k**2

    elif which in ['d1v', 'd1d']:
        if lterm=='density':
            B*=k
        else:
            B/=k

        if which=='d1d':
            B*=k**2 

        if qterm==1:
            out=np.gradient(B, k, axis=0, edge_order =2)
        else:
            out=-B/k

    elif which=='d3v':
        if lterm=='density':
            B*=k**3
        else:
            B*=k

        if qterm==1:
            out = np.gradient(np.gradient(np.gradient(B, k, axis=0, edge_order =2),
                                                         k, axis=0, edge_order =2), 
                                                         k, axis=0, edge_order =2)
        elif qterm==2:
            deriv2_k = np.gradient(np.gradient(B, k, axis=0, edge_order =2),
                                                  k, axis=0, edge_order =2)
            out=-3./k*deriv2_k
        elif qterm==3:
            deriv1_k = np.gradient(B, k, axis=0, edge_order =2)
            out=B*3./k**2
        else:
            out=-B/k**3

    elif which=='d0d':
        if lterm!='density':
            B/=k**2

        out=B*k**2

    return out

@njit
def compute(k, Pk, fct_k, b):
    Nk = len(k)
    kmin, kmax = np.min(k), np.max(k)
    res=np.zeros((Nk+1), dtype=np.complex128)
    for p in range(-Nk//2, Nk//2+1):
        res[p+Nk//2] = get_cp(p, b, k, Nk, kmin, kmax, fct_k)
    return res

def get_cp_of_r(k, Pk, gauge, lterm, which, qterm, rad, Newton, time_dict, r0, ddr, normW):
    if which in ['FG2', 'F2', 'G2', 'dv2']: 
        fct_k = P_of_k(k, Pk, gauge, rad) #, which, rad, time_dict, r0, ddr, normW) 
        np.save(output_dir+'fct_k'.format(which, lterm, qterm), np.vstack([k, fct_k]).T)
    else: 
        fct_k = quadratic_terms(qterm, k, Pk, gauge, lterm, which, time_dict, r0, ddr, normW) 
        np.save(output_dir+'fct_k_{}_lterm{}_qterm{}'.format(which, lterm, qterm), np.vstack([k, fct_k]).T)

    b=set_bias(k, fct_k)
    if not rad:
        fct_r = mathcalB(which, lterm, qterm, Newton, time_dict, r0, ddr, normW)
        np.save(output_dir+'fct_r_{}_lterm{}_qterm{}'.format(which, lterm, qterm), fct_r)
        return compute(k, Pk, fct_k, b)[:,None]*fct_r, b
    else:
        return compute(k, Pk, fct_k, b)[:,None], b

