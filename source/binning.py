import numpy as np
import os, sys
from sympy.physics.wigner import wigner_3j
from numba import njit, objmode, prange

from param_used import *

#@njit
def get_wigner(ellmax):
    N=0
    for l1 in range(2, ellmax):
        for l2 in range(l1, ellmax):

            l3c = max(l2, np.abs(l2-l1))
            odd = (l1 + l2 + l3c) % 2
            for l3 in range(l3c+odd, min(ellmax,l1+l2+1), 2):
                N+=1

    wigner=np.zeros((N))
    ind=0
    for l1 in range(2, ellmax):
        print('     ell1='+str(l1))
        for l2 in range(l1, ellmax):

            l3c = max(l2, np.abs(l2-l1))
            odd = (l1 + l2 + l3c) % 2
            for l3 in range(l3c+odd, min(ellmax,l1+l2+1), 2):
                with objmode(result='float64'):
                    result=float(wigner_3j(l1, l2, l3, 0,0,0))
                wigner[ind] = result
                ind+=1

    return wigner

@njit
def find_B(bl1, l1, bin1, bin2, bin3, wigner=1):
    ind1=0
    ind2=0
    B=0
    Xi=0
    lenght=len(np.unique(np.array([bin1[0], bin2[0], bin3[0]])))
    if lenght==1:
        perm=[1, 3, 6] # permutation of triangles inside a bin
    elif lenght==2:
        perm=[1, 1, 2]
    elif lenght==3:
        perm=[1, 1, 1]

    for l2 in range(l1, ellmax):
        l3c = max(l2, abs(l2-l1)) #max(bin3[0], abs(l2-l1)) 
        odd = (l1 + l2 + l3c) % 2
        for l3 in range(l2, ellmax):
            truth = l3>=l3c+odd and l3<min(ellmax,l1+l2+1) and (l3-l3c-odd)%2==0
            #truth = l3>=l3c+odd and l3<min(bin3[1],l1+l2)+1 and (l3-l3c-odd)%2==0
            if l1>=bin1[0] and l1<=bin1[1] \
                    and l2>=bin2[0] and l2<=bin2[1] \
                    and l3>=bin3[0] and l3<=bin3[1] \
                    and truth:
                if len(np.unique(np.array([l1, l2, l3]))) == 1: p=0
                elif len(np.unique(np.array([l1, l2, l3]))) == 2: p=1
                else: p=2
                B+=perm[p]*bl1[ind2]*(2.*l1+1)*(2.*l2+1)*(2.*l3+1)/4./np.pi#*wigner[ind1]**2
                Xi+=perm[p]
            if truth: ind1+=1
            ind2+=1
    return B, Xi, ind1


def load_bl(l1, lterm, which_list, name):
    for ind,w in enumerate(which_list):
        if ind==0:
            bl=np.load(name.format(lterm, w, l1))
        else:
            bl+=np.load(name.format(lterm, w, l1))
    return bl


def get_binned_B(ell, which, lterm, Newton=0, rad=0, isolate=False):
    Nbin = len(ell[:-1])
    
    #try:
    #    print(' try loading {}'.format(output_dir+'wigner_ellmax{}.npy'.format(ellmax)))
    #    wigner=np.load(output_dir+'wigner_ellmax{}.npy'.format(ellmax), allow_pickle=True)
    #except FileNotFoundError:
    #    print(' not found, computing...')
    #    print(' Computing Wigner up to ell={}'.format(ellmax))
    #    wigner=get_wigner(ellmax)
    #    np.save(output_dir+'wigner_ellmax{}'.format(ellmax), wigner)
    #    #wigner=np.load(output_dir+'wigner_ellmax{}.npy'.format(ellmax), allow_pickle=True)

    if Newton: name=output_dir+"bl/bl_{}_{}_newton_ell{}.npy"
    else: name=output_dir+"bl/bl_{}_{}_ell{}.npy"

    if rad: rad_key = '_rad'
    else: rad_key=''
    
    if which=='all':
        if lterm!='nopot':
            if isolate and rad: which_list=['F2{}'.format(rad_key), 'G2{}'.format(rad_key), 'dv2{}'.format(rad_key)]
            elif isolate and not Newton: which_list=['F2', 'G2', 'dv2', 'd1vd1d', 'd2vd0d', 'd1vd0d', 'd1vdod',]
            else: which_list=['F2{}'.format(rad_key), 'G2{}'.format(rad_key), 'dv2{}'.format(rad_key), \
                    'd2vd2v', 'd1vd3v', 'd1vd1d', 'd2vd0d', \
                'd1vd2v', 'd1vd0d', 'd1vdod', 'davd1v', 'd0pd3v', 'd0pd1d', 'd1vd2p']
        else:
            which_list=['F2{}'.format(rad_key), 'G2{}'.format(rad_key), 'dv2{}'.format(rad_key), \
                    'd2vd2v', 'd1vd3v', 'd1vd1d', 'd2vd0d', \
                'd1vd2v', 'd1vd0d', 'd1vdod', 'davd1v']

    elif which=='rsd':
        which_list=['G2', 'd2vd2v', 'd1vd3v', 'd1vd1d', 'd2vd0d']
    elif which=='pot':
        which_list=['d0pd3v', 'd0pd1d', 'd1vd2p']
    elif which in ['F2', 'G2', 'dv2']:
        which_list=[which+rad_key]
    else:
        which_list=[which]

    B_list=np.array([])
    Xi_list=np.array([])
    for i in range(Nbin):
        print("{}/{}".format(i, Nbin))
        for j in range(i, Nbin):
            for k in range(j, Nbin):
                bin1, bin2, bin3 = np.array([ell[i], ell[i+1]-1]), \
                                      np.array([ell[j], ell[j+1]-1]), \
                                      np.array([ell[k], ell[k+1]-1])
                print(bin1, bin2, bin3)
                B=0
                Xi=0
                ind1=0
                for l1 in range(2, ellmax):
                    bl=load_bl(l1, lterm, which_list, name) \
                            * np.load(output_dir+'wigner/wigner_ellmax{}_ell{}.npy'\
                            .format(ellmax, l1))**2
                    if isolate and rad: 
                        bl-=load_bl(l1, lterm, ['F2', 'G2', 'dv2'], name) \
                            * np.load(output_dir+'wigner/wigner_ellmax{}_ell{}.npy'\
                            .format(ellmax, l1))**2
                    elif isolate:
                        bl-=load_bl(l1, lterm, which_list, output_dir+"bl/bl_{}_{}_newton_ell{}.npy") \
                            * np.load(output_dir+'wigner/wigner_ellmax{}_ell{}.npy'\
                            .format(ellmax, l1))**2

                    res=find_B(bl, l1, bin1, bin2, bin3) #, wigner[ind1:])
                    B+=res[0]
                    Xi+=res[1]
                    ind1+=res[2]
                            
                B_list = np.append(B_list, B)
                Xi_list = np.append(Xi_list, Xi)

    if isolate and rad: isolate_key='_isolate'
    elif isolate and not Newton: isolate_key='_isolateGR'
    else: isolate_key=''
    if Newton:
        np.save(output_dir+"bl/bl_{}_{}_newton_binned.npy".format(lterm, which), np.array(B_list))
    else:
        if rad:
            np.save(output_dir+"bl/bl_{}_{}{}_rad_binned.npy".format(lterm, which, isolate_key), np.array(B_list))
        else:
            np.save(output_dir+"bl/bl_{}_{}{}_binned.npy".format(lterm, which, isolate_key), np.array(B_list))

    np.save(output_dir+'Xi', np.array(Xi_list))


#if __name__ == "__main__":
#    ell = [2, 6, 14, 25, 44, 57, 73, 94, 120, 152, 194, 244] #294, 344, 394, 444, \
#            #494, 544, 594, 644, 694, 744, 801, 901, 1001, 1101, 1201, 1301, 1401, \
#            #1501, 1601, 1701, 1801, 1901, 2000]
#    ellmax=256
#    
#    print('compute B')
#    get_binned_B(ell, 'F2', 'density')
#    #get_wigner(ellmax)
