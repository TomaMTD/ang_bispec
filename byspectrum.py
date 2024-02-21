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
    global Newton, which, lterm, qterm    
    ell_start=2
    squ=0
    for ind,arg in enumerate(argv):
        if '-' in arg:
            if arg[1:] in ['which', 'w']:
                print('change parameter: which={}'.format(argv[ind+1]))
                which=argv[ind+1]
                ell_start+=2
            elif arg[1:] in ['lterm', 'l']:
                print('change parameter: lterm={}'.format(argv[ind+1]))
                lterm=argv[ind+1]
                ell_start+=2
            elif arg[1:] in ['qterm', 'q']:
                print('change parameter: qterm={}'.format(argv[ind+1]))
                qterm=int(argv[ind+1])
                ell_start+=2
            elif arg[1:] in ['N', 'Newton']:
                print('change parameter: Newton={}'.format(argv[ind+1]))
                Newton=int(argv[ind+1])
                ell_start+=2
            elif arg[1:] in ['squ', 'squeeze']:
                print('change parameter: squeeze={}'.format(argv[ind+1]))
                squ=int(argv[ind+1])
                ell_start+=2

    Wrmin, Wrmax = get_distance(z0-dz)[0], get_distance(z0+dz)[0]
    r0=(Wrmin+Wrmax)/2.
    ddr=(-Wrmin+Wrmax)

    rmin, rmax = get_distance(zmin)[0], get_distance(zmax)[0]
    print('rmin={:.2f} rmax={:.2f}'.format(rmin, rmax))
    normW = 1./4.*bb*(1. + 1./np.tanh((Wrmax - Wrmin)/bb))*2./bb*(Wrmax-Wrmin)

    tr, Pk = get_power(0, gauge)
    kmin, kmax = np.min(tr['k']), np.max(tr['k'])

    time_dict = interp_growth(r0, ddr, normW, rmin, rmax)
    r_list = time_dict["r_list"]
    np.save(output_dir+'time_dict', time_dict)

    if Nchi>0:
        chi_list=np.linspace(rmin, rmax, Nchi)
    else:
        chi_list=np.copy(r_list)


    #which=argv[2]
    if argv[ell_start-1] == 'debug':
        ell = np.float64(argv[ell_start])

        b=set_bias(gauge, lterm, which, qterm)
#y=get_cp_of_r(r_list, tr['k'], Pk, gauge, lterm, which, qterm, False, ra, a, Ha, D, f, r0, ddr, normW, b)
        y=get_cp_of_r(tr['k'], Pk, gauge, lterm, which, qterm, time_dict, r0, ddr, normW, b)
        y1=mathcalD(r_list, y, ell)

        plot_integrand(ell, 0, r_list, y, y1, rmin, rmax, len(tr['k']), kmax, kmin, 0, b)


    elif argv[ell_start-1] == 'search':
        for ell in range(2, 128*2, 2):
            for w in ['FG2', 'd3v', 'd2v', 'd1v', 'd1d']:

                if w=='d3v':
                    ql=[1, 2, 3, 4]
                elif w=='d2v':
                    ql=[1, 2, 3]
                elif w in ['d1v', 'd1d']:
                    ql=[1, 2]
                else:
                    ql=[1]

                for l in ['density', 'rsd', 'pot', 'doppler']:
                    #for q in ql:
                    #clname=output_dir+'Cln_{}_qterm{}_{}_ell{}.txt'.format(w, q, l, int(ell))
                    if w == 'FG2':
                        clname=output_dir+'cln/Cln_{}_ell{}.txt'.format(l, int(ell))
                    else:
                        clname=output_dir+'cln/Cln_{}_{}_ell{}.txt'.format(w, l, int(ell))
                    try:
                        cl=np.loadtxt(clname)
                    except FileNotFoundError:
                        #print(clname, '0/100')
                        for q in ql:
                            clname=output_dir+'cln/Cln_{}_qterm{}_{}_ell{}.txt'.format(w, q, l, int(ell))
                            try:
                                cl=np.loadtxt(clname)
                                try: 
                                    for ind, chi in enumerate(cl[:,0]):
                                        if cl[ind,1]==0:
                                            print(clname, '{}/100'.format(ind))
                                            break
                                except IndexError:
                                    print('{} empty'.format(clname))

                            except FileNotFoundError:
                                print(clname, 'qterm: 0/100')
                                continue
                        continue

                    try: 
                        for ind, chi in enumerate(cl[:,0]):
                            if cl[ind,1]==0:
                                print(clname, '{}/100'.format(ind))
                                break
                    except IndexError:
                        print('{} empty'.format(clname))

    elif argv[ell_start-1] == 'cl':
        if which in ['FG2', 'F2', 'G2']:
            qterm_list=[0]
        else:
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

        if qterm in qterm_list :
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
        
        for lt in lterm_list:
            b_list=np.zeros((len(qterm_list)))
            y=np.zeros((len(tr['k'])+1, len(r_list), len(qterm_list)), dtype=complex)
            y1=np.zeros((len(tr['k'])+1, len(r_list), len(qterm_list)), dtype=complex)

            if lt=='density' and which in ['FG2', 'F2', 'G2']: kpow=2.
            else: kpow=0

            for ind, qt in enumerate(qterm_list):
                print('computing integrand tab of chi qterm={}'.format(qt))
                #b_list[ind]=set_bias(gauge, lt, which, qt, False)
#y[:,:,ind]=get_cp_of_r(r_list, tr['k'], Pk, gauge, lt, which, qt, False, ra, a, Ha, D, f, r0, ddr, normW, b_list[ind])
                y[:,:,ind], b_list[ind]=get_cp_of_r(tr['k'], Pk, gauge, lt, which, qt, time_dict, r0, ddr, normW)
                np.save(output_dir+'cp_of_r', y)
                
                for ell in np.float64(argv[ell_start:]):
                    if which in ['FG2', 'F2', 'G2']:
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
                which_list=['F2', 'G2', 'd2vd2v', 'd1vd1d', 'd2vd0d', 'd1vd3v',\
                        'dv2', 'd1vd2v', 'd1vd0d', 'd1vdod', 'd0pd3v', 'd0pd1d', 'd1vd2p', 'davd1v'] #RG2
            else:
                which_list=['F2', 'G2', 'dv2']
        else:
            which_list=[which]

        if lterm=='each':
            lterm_list=['density', 'rsd', 'pot', 'doppler']
        else:
            lterm_list=[lterm]

        if argv[ell_start] in ['equi', 'squ']:
            ell_list=equi
        else:
            ell_list=[int(c) for c in argv[ell_start:]]
 
        for wh in which_list:
            for lt in lterm_list:
                print('computing {} for which={} lterm={} ell=equi'.format(argv[ell_start-1], wh, lt))

                if argv[ell_start-1] == 'Am':
                   
                    for ell in ell_list:
                        print(' Am_ell={}'.format(int(ell)))

                        if squ>0:
                            get_Am(chi_list, squ, ell, ell, wh, Newton, lt, time_dict,\
                                    r0, ddr, normW, rmin, rmax)
                            get_Am(chi_list, ell, squ, ell, wh, Newton, lt, time_dict,\
                                    r0, ddr, normW, rmin, rmax)
                            get_Am(chi_list, ell, ell, squ, wh, Newton, lt, time_dict,\
                                    r0, ddr, normW, rmin, rmax)
                        else:
                            get_Am(chi_list, ell, ell, ell, wh, Newton, lt, time_dict,\
                                    r0, ddr, normW, rmin, rmax)

                elif argv[ell_start-1] == 'Il':
                    #b=set_bias(gauge, wh, True)
                    cp_tr, b = get_cp_of_r(tr['k'], tr['dTdk'], gauge, lt, wh, 0, time_dict\
                            , r0, ddr, normW)
                    np.savetxt(output_dir+'cpTr_{}.txt'.format(wh), cp_tr.T)

                    for ell in ell_list: #np.float64(argv[ell_start:]):
                        print(' Il_ell={}'.format(int(ell)))
                        if squ>0:
                            get_Il(chi_list, squ, ell, ell, wh, time_dict, r0, ddr, normW, rmin, rmax,\
                                    cp_tr[:,0], b, len(tr['k']), kmax, kmin)
                            get_Il(chi_list, ell, squ, ell, wh, time_dict, r0, ddr, normW, rmin, rmax,\
                                    cp_tr[:,0], b, len(tr['k']), kmax, kmin)
                            get_Il(chi_list, ell, ell, squ, wh, time_dict, r0, ddr, normW, rmin, rmax,\
                                    cp_tr[:,0], b, len(tr['k']), kmax, kmin)
                        else:
                            get_Il(chi_list, ell, ell, ell, wh, time_dict, r0, ddr, normW, rmin, rmax,\
                                    cp_tr[:,0], b, len(tr['k']), kmax, kmin)
 
                elif argv[ell_start-1] == 'bl':

                    if argv[ell_start] == 'equi':
                        shape = '_equi'
                        if squ>0 : shape = '_squ'
                    else:
                        shape=''

                    if rad and wh in ['F2', 'G2', 'dv2']:
                        fich = open(output_dir+"bl_{}_{}_rad{}.txt".format(lt, wh, shape), "w")
                    elif Newton:
                        fich = open(output_dir+"bl_{}_{}_newton{}.txt".format(lt, wh, shape), "w")
                    else:
                        fich = open(output_dir+"bl_{}_{}{}.txt".format(lt, wh, shape), "w")
                    
                    if argv[ell_start] in ['equi', 'squ']:
                        for ell in ell_list:
                            #try: \
                            if squ>0:
                                bl=spherical_bispectrum(wh, Newton, lt, squ, ell, ell, time_dict, rmax, rmin)
                                fich.write('{} {} {} {:.16e} \n'.format(squ, ell, ell, bl))
                            else:
                                bl=spherical_bispectrum(wh, Newton, lt, ell, ell, ell, time_dict, rmax, rmin)
                                fich.write('{} {} {} {:.16e} \n'.format(ell, ell, ell, bl))
                            #except IndexError:
                            #    print(' fail')
                            #    fich.write('{} {} {} {:.16e} \n'.format(ell, ell, ell, -1))
                            #    continue

                    else:     
                        for ell_ind in range(ell_start, len(argv)):
                            ell = [int(c) for c in argv[ell_ind].split(',')]
                            bl=spherical_bispectrum(wh, Newton, lt, ell[0], ell[1], ell[2], time_dict, rmax, rmin)
                            fich.write('{} {} {} {:.16e} \n'.format(ell[0], ell[1], ell[2], bl))
                    fich.close
    return 0

if __name__ == "__main__":
    r=main(sys.argv)
