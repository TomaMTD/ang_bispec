import numpy as np
import os, sys
from classy import Class
from scipy import integrate
import time
from numba import njit
import cubature

sys.path.insert(1, './source')
from param import *
from fftlog import *
from lincosmo import *
from mathematica import *
from general_ps import *
from bispectrum import *

def main(argv):
    global which, lterm, qterm    

    ell_start=2
    for ind,arg in enumerate(argv):
        if '-' in arg:
            if arg[1:]=='which':
                print('change parameter: which={}'.format(argv[ind+1]))
                which=argv[ind+1]
                ell_start+=2
            elif arg[1:]=='lterm':
                print('change parameter: lterm={}'.format(argv[ind+1]))
                lterm=argv[ind+1]
                ell_start+=2
            elif arg[1:]=='qterm':
                print('change parameter: qterm={}'.format(argv[ind+1]))
                qterm=int(argv[ind+1])
                ell_start+=2

    Wrmin, Wrmax = get_distance(z0-dz)[0], get_distance(z0+dz)[0]
    r0=(Wrmin+Wrmax)/2.
    ddr=(-Wrmin+Wrmax)

    rmin, rmax = get_distance(zmin)[0], get_distance(zmax)[0]
    print('rmin={:.2f} rmax={:.2f}'.format(rmin, rmax))
    #normW = 1./4.*bb*(1. + 1./np.tanh((Wrmax - Wrmin)/bb))*(np.log(1. + np.exp((2.*Wrmax)/bb)) - np.log(1. + np.exp((2.*Wrmin)/bb)))
    normW = 1./4.*bb*(1. + 1./np.tanh((Wrmax - Wrmin)/bb))*2./bb*(Wrmax-Wrmin)

    a, ra, Ha, Oma, D, f, v, w = growth_fct()
    tr, Pk = get_power(0, gauge)
    kmin, kmax = np.min(tr['k']), np.max(tr['k'])

    r_list=ra[np.logical_and(ra<=rmax, ra>=rmin)]

    #which=argv[2]
    if argv[ell_start-1] == 'debug':
        ell = np.float64(argv[ell_start])

        b=set_bias(gauge, lterm, which, qterm, False)
        y=get_cp_of_r(r_list, tr['k'], Pk, gauge, lterm, which, qterm, False, ra, a, Ha, D, f, r0, ddr, normW, b)
        y1=mathcalD(r_list, y, ell)

        chi_list=np.linspace(rmin, rmax, Nchi)
        plot_integrand(ell, 0, r_list, y, y1, rmin, rmax, len(tr['k']), kmax, kmin, 0, b)


    elif argv[ell_start-1] == 'search':
        for ell in range(2, 128):
            for w in ['d3v', 'd2v', 'd1v', 'd1d', 'd0d']:

                if w=='d3v':
                    ql=[1, 2, 3, 4]
                elif w=='d2v':
                    ql=[1, 2, 3]
                elif w in ['d1v', 'd1d']:
                    ql=[1, 2]
                else:
                    ql=[1]

                for l in ['density', 'rsd', 'pot', 'doppler']:
                    for q in ql:
                        clname=output_dir+'Cln_{}_qterm{}_{}_ell{}.txt'.format(w, q, l, int(ell))
                        try:
                            cl=np.loadtxt(clname)
                        except FileNotFoundError:
                            print('{} not found'.format(clname))
                            continue

                        try: 
                            for ind, chi in enumerate(cl[:,0]):
                                if cl[ind,1]==0:
                                    print(clname, '{}/100'.format(ind))
                                    break
                        except IndexError:
                            print('{} empty'.format(clname))

    elif argv[ell_start-1] == 'cl':

        #if which=='d1d':
        #    which_list = ['d1d1', 'd1d2']
        #else:
        #    which_list = [which]

        #for wh in which_list:
        if which=='d2v':
            qterm_list=[1,2,3]
        elif which in ['d1v', 'd1d']:
            qterm_list=[1,2]
        elif which in ['d0d']:
            qterm_list=[1]
        elif which=='d3v':
            qterm_list=[1,2,3,4]
        else:
            qterm_list=[0]

        if qterm in qterm_list or which==['FG2']:
            qterm_list = [qterm]
        
        if qterm==-1:
            compute_all_separate=True
        else:
            compute_all_separate=False
        compute_all=False

        if lterm=='all':
            lterm_list=['density', 'pot', 'rsd', 'doppler']
        else:
            lterm_list=[lterm]
        
        chi_list=np.linspace(rmin, rmax, Nchi)
        for lt in lterm_list:
            b_list=np.zeros((len(qterm_list)))
            y=np.zeros((len(tr['k'])+1, len(r_list), len(qterm_list)), dtype=complex)
            y1=np.zeros((len(tr['k'])+1, len(r_list), len(qterm_list)), dtype=complex)

            if lt=='density' and which=='FG2': kpow=2.
            else: kpow=0

            for ind, qt in enumerate(qterm_list):
                print('computing integrand tab of chi qterm={}'.format(qt))
                b_list[ind]=set_bias(gauge, lt, which, qt, False)
                y[:,:,ind]=get_cp_of_r(r_list, tr['k'], Pk, gauge, lt, which, qt, False, ra, a, Ha, D, f, r0, ddr, normW, b_list[ind])
                np.save('cp_of_r', y)
                
                for ell in np.float64(argv[ell_start:]):
                    if which=='FG2':
                        y1[:,:,ind]=mathcalD(r_list, y[:,:,ind], ell)

                    if len(qterm_list)==1:
                        get_all_Cln(which, qt, lt, chi_list, ell, r_list, y[:,:,0], y1[:,:,0], rmin, rmax, len(tr['k']), kmax, kmin, kpow, b_list[0])
                    elif compute_all_separate:
                        get_all_Cln(which, qt, lt, chi_list, ell, r_list, y[:,:,ind], y1[:,:,ind], rmin, rmax, len(tr['k']), kmax, kmin, kpow, b_list[ind])
                    else:
                        compute_all=True

            if compute_all:
                for ell in np.float64(argv[ell_start:]):
                    get_all_Cln(which, qterm, lt, chi_list, ell, r_list, y, y1, rmin, rmax, len(tr['k']), kmax, kmin, kpow, b_list)

    else:
        if which=='all':
            if argv[ell_start-1] == 'bl':
                which_list=['F2', 'G2', 'd2vd2v', 'd1vd1d', 'd2vd0d', 'd1vd3v'] 
            else:
                which_list=['F2', 'G2']
        else:
            which_list=[which]

        if lterm=='each':
            lterm_list=['density', 'rsd', 'pot', 'doppler']
        else:
            lterm_list=[lterm]

        for wh in which_list:
            for lt in lterm_list:
                print('computing {} for which={} lterm={} ell=equi'.format(argv[ell_start-1], wh, lt))

                if argv[ell_start-1] == 'Am':
                    if argv[ell_start]=='equi':
                        ell_list=np.arange(2, 128)
                    else:
                        ell_list=[int(c) for c in argv[ell_start:]]
                    
                    for ell in ell_list:
                        print(' Am_ell={}'.format(int(ell)))

                        chi_list=np.linspace(rmin, rmax, Nchi)
                        get_Am(chi_list, ell, ell, ell, wh, lt, ra, D, f, v, w, Oma, Ha, r0, ddr, normW, rmin, rmax, r_list)

                elif argv[ell_start-1] == 'Il':
                    b=set_bias(gauge, wh, True)

                    if wh == 'F2':
                        r_arg = np.array([0])
                    else:
                        r_arg = r_list
                    
                    cp_tr = get_cp_of_r(r_arg, tr['k'], tr['dTdk'], gauge, lt, wh, 0, True, ra, a, Ha, D, f, r0, ddr, normW, b)
                    np.savetxt(output_dir+'cpTr_{}.txt'.format(wh), cp_tr.T)

                    for ell in np.float64(argv[ell_start:]):
                        print(' Il_ell={}'.format(int(ell)))
                        chi_list=np.linspace(rmin, rmax, Nchi)
                        get_Il(chi_list, ell, ell, ell, wh, a, ra, D, f, v, w, Oma, Ha, r0, ddr, normW, rmin, rmax, r_list, cp_tr, bphi, len(tr['k']), kmax, kmin)
 
                elif argv[ell_start-1] == 'bl':
                    fich = open(output_dir+"bl_{}_{}.txt".format(lt, wh), "w")
                    
                    if argv[ell_start]=='equi':
                        for ell in range(2, 128):

                            try: 
                                bl=spherical_bispectrum(wh, lt, ell, ell, ell, ra, D, Ha, Oma, f, v, w, r_list, rmax, rmin, r0, ddr, normW)
                                fich.write('{} {} {} {:.16e} \n'.format(ell, ell, ell, bl))
                            except IndexError:
                                print(' fail')
                                fich.write('{} {} {} {:.16e} \n'.format(ell, ell, ell, -1))
                                continue


                    else:     
                        for ell_ind in range(ell_start, len(argv)):
                            ell = [int(c) for c in argv[ell_ind].split(',')]
                            bl=spherical_bispectrum(wh, lt, ell[0], ell[1], ell[2], ra, D, Ha, Oma, f, v, w, r_list, rmax, rmin, r0, ddr, normW)
                            fich.write('{} {} {} {:.16e} \n'.format(ell[0], ell[1], ell[2], bl))
                    fich.close
    return 0

if __name__ == "__main__":
    r=main(sys.argv)
