import numpy as np
import os, sys
from sympy.physics.wigner import wigner_3j
from numba import njit

from param import *

def get_wigner(ellmax):
    wigner=[]
    for l1 in range(2, ellmax):
        for l2 in range(l1, ellmax):

            l3c = max(l2, abs(l2-l1))
            odd = (l1 + l2 + l3c) % 2
            for l3 in range(l3c+odd, min(ellmax,l1+l2+1), 2):
                wigner.append(float(wigner_3j(l1, l2, l3, 0,0,0)))
    np.save(output_dir+'wigner', np.array(wigner))

@njit
def find_B(bl1, l1, bin1, bin2, bin3, wigner):
    ind1=0
    ind2=0
    B=0
    Xi=0
    for l2 in range(l1, ellmax):

        l3c = max(l2, abs(l2-l1))
        odd = (l1 + l2 + l3c) % 2
        for l3 in range(l2, ellmax):
            truth=l3>=l3c+odd and l3<min(ellmax,l1+l2+1) and (l3-l3c-odd)%2==0
            if l1>=bin1[0] and l1<=bin1[1] \
                    and l2>=bin2[0] and l2<=bin2[1] \
                    and l3>=bin3[0] and l3<=bin3[1] \
                    and truth:
                B+=bl1[ind2]*(2.*l1+1)*(2.*l2+1)*(2.*l3+1)/4./np.pi*wigner[ind1]**2
                Xi+=1

            if truth: ind1+=1
            ind2+=1
    return B, Xi, ind1


def load_bl(l1, lterm, which_list, name):
    for ind,w in enumerate(which_list):
        print(name.format(lterm, w, l1))
        if ind==0:
            bl=np.load(name.format(lterm, w, l1))
        else:
            bl+=np.load(name.format(lterm, w, l1))

    return bl


def get_binned_B(ell, which, lterm, Newton=0, rad=0):
    Nbin = len(ell[:-1])
    wigner=np.load(output_dir+'wigner.npy', allow_pickle=True)

    if Newton: name=output_dir+"bl/bl_{}_{}_newton_ell{}.npy"
    else: name=output_dir+"bl/bl_{}_{}_ell{}.npy"

    if rad: rad_key = '_rad'
    else: rad_key=''
    
    if which=='all':
        if lterm!='nopot':
            which_list=['F2{}'.format(rad_key), 'G2{}'.format(rad_key), 'dv2{}'.format(rad_key), \
                    'd2vd2v', 'd1vd3v', 'd1vd1d', 'd2vd0d', \
                'd1vd2v', 'd1vd0d', 'd1vdod', 'd0pd3v', 'd0pd1d', 'd1vd2p']
        else:
            which_list=['F2'.format(rad_key), 'G2'.format(rad_key), 'dv2{}'.format(rad_key), \
                    'd2vd2v', 'd1vd3v', 'd1vd1d', 'd2vd0d', \
                'd1vd2v', 'd1vd0d', 'd1vdod']

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

                B=0
                Xi = 0
                ind1=0
                for l1 in range(2, ellmax):
                    bl=load_bl(l1, lterm, which_list, name)
                    res=find_B(bl, l1, bin1, bin2, bin3, wigner[ind1:])
                    B+=res[0]
                    Xi+=res[1]
                    ind1+=res[2]
                            
                B_list = np.append(B_list, B)
                Xi_list = np.append(Xi_list, Xi)

    if Newton:
        np.save(output_dir+"bl/bl_{}_{}_newton_binned.npy".format(lterm, which), np.array(B_list))
    else:
        if rad:
            np.save(output_dir+"bl/bl_{}_{}_rad_binned.npy".format(lterm, which), np.array(B_list))
        else:
            np.save(output_dir+"bl/bl_{}_{}_binned.npy".format(lterm, which), np.array(B_list))

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
