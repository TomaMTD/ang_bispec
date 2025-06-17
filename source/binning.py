import numpy as np
import os, sys, h5py
from sympy.physics.wigner import wigner_3j
from numba import njit, objmode, prange

from param_used import *

@njit
def find_B(bl1, l1, bin1, bin2, bin3, wigner=1):
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
        for l3 in range(l2, min(l2+l1+1, ellmax)):
            truth = l3>=l3c+odd and l3<min(ellmax,l1+l2+1) and (l3-l3c-odd)%2==0
            #truth = l3>=l3c+odd and l3<min(bin3[1],l1+l2)+1 and (l3-l3c-odd)%2==0
            if l1>=bin1[0] and l1<=bin1[1] \
                    and l2>=bin2[0] and l2<=bin2[1] \
                    and l3>=bin3[0] and l3<=bin3[1] \
                    and truth:
                if len(np.unique(np.array([l1, l2, l3]))) == 1: p=0
                elif len(np.unique(np.array([l1, l2, l3]))) == 2: p=1
                else: p=2
                B+=perm[p]*bl1[ind2]*(2.*l1+1)*(2.*l2+1)*(2.*l3+1)/4./np.pi
                Xi+=perm[p]
            ind2+=1
    return B, Xi


@njit
def load_bl(bl, ell, l1, Nbin, B_list, Xi_list):
    ind=0
    for i in range(Nbin):
        bin1 = np.array([ell[i], ell[i+1]-1])
        for j in range(i, Nbin):
            bin2 = np.array([ell[j], ell[j+1]-1])
            for k in range(j, Nbin):
                bin3 = np.array([ell[k], ell[k+1]-1])
                res=find_B(bl, l1, bin1, bin2, bin3) 
                        
                B_list[ind] += res[0]
                Xi_list[ind] += res[1]
                ind+=1
    return B_list, Xi_list


def isfile(l1, lterm, which_list, name):
    for ind,w in enumerate(which_list):
        if not os.path.isfile(name.format(lterm, w, l1)):
            print(name.format(lterm, w, l1) + ' not found')
    return 0 



def get_binned_B(ell, which_list, lterm, Newton=0, rad=0):
    Nbin = len(ell[:-1])
    num_bin_triplet = int(Nbin*(Nbin**2+3*Nbin+2)/6)

    
    if Newton: name=output_dir+"bl/bl_{}_{}_newton.h5"
    else: name=output_dir+"bl/bl_{}_{}.h5"

    try:
        for rien, w in enumerate(which_list):
            B_list=np.zeros((num_bin_triplet))
            Xi_list=np.zeros((num_bin_triplet))

            print(' open file {}'.format(name.format(lterm, w)))
            with h5py.File(name.format(lterm, w), "a") as f:
                for l1 in range(2, ellmax):
                    print(' ell1={}'.format(l1))
                    bl=f[f"bl_ell{l1}"][:] * f[f"wigner_ell{l1}"][:]**2
                    B_list, Xi_list = load_bl(bl, ell, l1, Nbin, B_list, Xi_list)

    except BlockingIOError:
        print('{} already open?'.format(name.format(lterm, w)))
        exit()


    if len(which_list)==1:
        which_key=which_list[0]
    else:
        which_key=which

        
    if Newton:
        np.save(output_dir+"bl/bl_{}_{}_newton_binned.npy".format(lterm, which_key), np.array(B_list))
    else:
        if rad:
            np.save(output_dir+"bl/bl_{}_{}_rad_binned.npy".format(lterm, which_key), np.array(B_list))
        else:
            np.save(output_dir+"bl/bl_{}_{}_binned.npy".format(lterm, which_key), np.array(B_list))

    np.save(output_dir+'Xi', np.array(Xi_list))
