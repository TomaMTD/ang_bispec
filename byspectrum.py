import numpy as np
import os, sys, argparse
from art import text2art
from classy import Class
from scipy import integrate
import time
from numba import njit
import cubature

import path
sys.path.insert(1, path.path+'/byspectrum/source')
from param import *
sys.path.insert(1, output_dir)

text = text2art("Angular BISPECTRUM", font='small')  # Vous pouvez changer 'block' pour d'autres styles disponibles
print(text)

def arguments():
    global output_dir
    parser = argparse.ArgumentParser(description="Script pour traiter des paramètres")
    parser.add_argument('-w', '--which', default=which, type=str, help='second order terms')
    parser.add_argument('-l', '--lterm', default=lterm, type=str, help='linear terms')
    parser.add_argument('-q', '--qterm', default=qterm, type=int, help='-1, 0, 1, 2, 3, 4 only for which neq F2 G2')
    parser.add_argument('-N', '--Newton', default=Newton, type=int, help='Newtonian gravity: 0=No, 1=Yes')
    parser.add_argument('-r', '--rad',default=rad, type=int, help='Radiation: 0=No, 1=Yes')
    parser.add_argument('-L', '--Limber',default=Limber, type=int, help='Radiation: 0=No, 1=Yes')
    parser.add_argument('-z0'    , default=z0, type=float, help='center of redshift bin')
    parser.add_argument('-dz'    , default=dz, type=float, help='half width of redshift bin')
    parser.add_argument('-bb'    , default=bb, type=float, help='How fast is the window function decaying')
    parser.add_argument('-f'     , '--force', default=force, type=int, help='Wether you and to overwrite all output (force computation)')
    parser.add_argument('-Nchi'  , default=Nchi, type=int, help='Number of r/chi value to evaluate cl, Am and Il')
    parser.add_argument('-ell'   , default=ell, type=int, help='')
    parser.add_argument('-ellmax',default=ellmax, type=int, help='')
    parser.add_argument('-o', '--output_dir', default=output_dir+'/', type=str, help='path of output')
    parser.add_argument('-m', '--mode', default='bl', type=str, help='Computation mode: [cl, Il, bl, bin]')
    parser.add_argument('-config', '--configuration', default='all', type=str, help='what triangle configuration to compute')

    parser.add_argument('-h100'     , type=float,default=h100) 
    parser.add_argument('-omega_b'  , type=float,default=omega_b) 
    parser.add_argument('-omega_cdm', type=float,default=omega_cdm) 
    parser.add_argument('-omega_m'  , type=float,default=omega_m)
    parser.add_argument('-omega_r'  , type=float,default=omega_r)
    parser.add_argument('-omega_k'  , type=float,default=omega_k)
    parser.add_argument('-omega_l'  , type=float,default=omega_l)
    parser.add_argument('-A_s'      , type=float,default=A_s) 
    parser.add_argument('-n_s'      , type=float,default=n_s)
    parser.add_argument('-k_pivot'  , type=float,default=k_pivot) 
    parser.add_argument('-c'        , type=float,default=c) 
    parser.add_argument('-H0'       , type=float,default=H0) 
    parser.add_argument('-chi_ind'  , type=int,default=-1) 
    parser.add_argument('-relerr'   , type=float,default=relerr) 
    parser.add_argument('-bins'   , type=list,default=bins) 

    argv=parser.parse_args()
    if argv.output_dir[-1]!='/': argv.output_dir+='/'
    return argv

def write_args(argv):
    if not os.path.exists(output_dir+'param_used.py'):
        with open(output_dir+'param_used.py', 'w') as file:
            for key, value in vars(argv).items():
                if isinstance(value, str):
                    file.write(f"{key} = '{value}'\n")
                else:
                    file.write(f"{key} = {value}\n")
            file.write("h = {}\n".format(h100/100))

def ensure_directory_exists(path):
    """checks wether the output path exists"""
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path+'/Am')
        os.makedirs(path+'/Il')
        os.makedirs(path+'/bl')
        os.makedirs(path+'/cln')
        print(f"output dir created : {path}")

def main(argv):

    import lincosmo 
    import fftlog
    import general_ps
    import bispectrum
    import binning
    

    if argv.force!=0: print('-force is activated, overwritting files')

    Wrmin, Wrmax = lincosmo.get_distance(argv.z0-argv.dz)[0], lincosmo.get_distance(argv.z0+argv.dz)[0]
    r0=(Wrmin+Wrmax)/2.
    ddr=(-Wrmin+Wrmax)

    rmin, rmax = lincosmo.get_distance(argv.z0-2.*argv.dz)[0], lincosmo.get_distance(argv.z0+2.*argv.dz)[0]
    print('Window function limits: rmin={:.0f} rmax={:.0f}'.format(rmin, rmax))

    tr, Pk = lincosmo.get_power(0)
    kmin, kmax = np.min(tr['k']), np.max(tr['k'])

    time_dict = lincosmo.interp_growth(r0, ddr, rmin, rmax)
    r_list = time_dict["r_list"]
    normW = time_dict["normW"]
    np.save(output_dir+'time_dict', time_dict)

    # norm of W not W_tilde
    # print(1./4.*bb*(1. + 1./np.tanh((Wrmax - Wrmin)/bb))*2./bb*(Wrmax-Wrmin))

    if argv.Nchi>0:
        chi_list=np.linspace(rmin, rmax, argv.Nchi)
    else:
        chi_list=np.copy(r_list)

    if argv.chi_ind>-1:
        chi_list=np.array([chi_list[argv.chi_ind]])

    if argv.ellmax<=argv.ell:
        ell_list=[argv.ell]
    else:
        if argv.configuration in ['equi', 'squ', 'folded', 'esf']:
            if argv.ell%2!=0: argv.ell+=1
            ell_list=np.arange(argv.ell, argv.ellmax, 2)
        else:
            ell_list=np.arange(argv.ell, argv.ellmax, 1)

    if argv.lterm=='each':
        lterm_list=['density', 'pot', 'dpot', 'rsd', 'doppler']
    else:
        lterm_list=[argv.lterm]

    if argv.mode == 'search':
        wlist=[]
        llist=[]
        elllist=[]
        indlist=[]

        for ell in range(2, 128*2, 1):
            for w in ['FG2', 'd1v', 'd1d', 'd3v', 'd2v']:

                if w=='d3v':
                    ql=[1, 2, 3, 4]
                elif w=='d2v':
                    ql=[1, 2, 3]
                elif w in ['d1v', 'd1d']:
                    ql=[1, 2]
                else:
                    ql=[1]

                for l in ['density', 'rsd', 'pot', 'dpot', 'doppler']:
                    #for q in ql:
                    #clname=output_dir+'Cln_{}_qterm{}_{}_ell{}.txt'.format(w, q, l, int(ell))
                    if w == 'FG2':
                        clname=output_dir+'cln/Cln_{}_ell{}.txt'.format(l, int(ell))
                    else:
                        clname=output_dir+'cln/Cln_{}_{}_ell{}.txt'.format(w, l, int(ell))
                    
                    test=False
                    if os.path.isfile(clname):
                        cl=np.loadtxt(clname)
                        try:
                            for ind, chi in enumerate(cl[:,0]):
                                if cl[ind,1]==0:
                                    #print(clname, '{}/100'.format(ind))
                                    test=True
                                    break
                        except IndexError:
                            print('{} empty'.format(clname))

                    if test:
                        printt=False
                        #print(clname, '0/100')
                        for q in ql:
                            clname_q=output_dir+'cln/Cln_{}_qterm{}_{}_ell{}.txt'.format(w, q, l, int(ell))
                            try:
                                cl=np.loadtxt(clname_q)
                                try:
                                    for indq, chi in enumerate(cl[:,0]):
                                        if cl[indq,1]==0:
                                            print(clname_q, '{}/100'.format(indq))
                                            break
                                except IndexError:
                                    print('{} empty'.format(clname_q))

                            except FileNotFoundError:
                                #print(clname, '{}/100'.format(ind))
                                #print(clname, 'qterm: 0/100')
                                printt = True
                                continue
                        if printt: 
                            wlist.append(w)
                            llist.append(l)
                            elllist.append(ell)
                            indlist.append(ind)

                            print(clname, '{}/100'.format(ind))
                            #os.system("python -u byspectrum.py -w {} -l {} -q 0 -i {} cl {}".format(w,l,ind,ell))
                        continue
        print(wlist)
        print(llist)
        print(elllist)
        print(indlist)

    elif argv.mode in ['cl', 'cln', 'Cl', 'Cln']:
        if argv.which in ['FG2', 'F2', 'G2']:
            qterm_list=[0]
        else:
            if argv.which=='d2v':
                qterm_list=[1,2,3]
            elif argv.which in ['d1v', 'd1d']:
                qterm_list=[1,2]
            elif argv.which in ['d0d']:
                qterm_list=[1]
            elif argv.which=='d3v':
                qterm_list=[1,2,3,4]
            else:
                qterm_list=[0]

        if argv.which=='all':
            which_list=['FG2', 'd2v', 'd1v', 'd3v', 'd1d']
        else:
            which_list=[argv.which]

        if argv.qterm in qterm_list and not Limber:
            qterm_list = [argv.qterm]
        
        if argv.qterm==-1:
            compute_all_separate=True
        else:
            compute_all_separate=False
        compute_all=False
        
        print('Computing generalised power spectra for:')
        print('     which={}'.format(which_list))
        print('     lterm={}'.format(lterm_list))
        print('     Limber={}'.format(argv.Limber))
        if argv.ell>1:
            print('     ell={}'.format(argv.ell))
        else:
            print('     ell=[{}, {}]'.format(argv.ell, argv.ellmax))
            
        for wh in which_list:
            for lt in lterm_list:
                b_list=np.zeros((len(qterm_list)))
                cp=np.zeros((len(tr['k'])+1, len(qterm_list)), dtype=complex)
                fctr=np.zeros((3, len(r_list), len(qterm_list)), dtype=complex)

                if lt=='density' and wh in ['FG2', 'F2', 'G2']: kpow=2.
                else: kpow=0

                for ind, qt in enumerate(qterm_list):
                    print('computing integrand tab of chi qterm={}'.format(qt))
                    cp[:,ind], fctr[0, :,ind], b_list[ind]=\
                            fftlog.get_cp_of_r(tr['k'], Pk, lt, wh, qt, 0, argv.Newton, time_dict, r0, ddr, normW)
                    #np.save(argv.output_dir+'cp_of_r', cp)

                    if argv.Limber: cp_arg = np.vstack([tr['k'], tr['k']**4*Pk]).T
                    elif compute_all: cp_arg = cp
                    else: cp_arg = cp[:,ind]
                    
                    for ell in ell_list:
                        #if wh in ['FG2', 'F2', 'G2', 'd1d']:
                        fctr[1, :,ind]=fftlog.mathcalD(r_list, fctr[0, :,ind], ell, axis=0)
                        fctr[2, :,ind]=fftlog.mathcalD(r_list, fctr[1, :,ind], ell, axis=0)

                        if len(qterm_list)==1 or compute_all_separate:
                            general_ps.get_all_Cln(wh, qt, lt, argv.Newton, chi_list, ell, r_list, \
                                    cp_arg, fctr[:,:,ind], rmin, rmax, len(tr['k']), kmax, kmin, kpow, b_list[ind], argv.Limber)
                        else:
                            compute_all=True

                if compute_all:
                    for ell in ell_list:
                        general_ps.get_all_Cln(wh, qterm, lt, argv.Newton, chi_list, ell, r_list, cp_arg, fctr, \
                                rmin, rmax, len(tr['k']), kmax, kmin, kpow, b_list, argv.Limber)

    elif argv.mode in ['prim', 'primordial', 'Primordial']:
        cp_tr, b = fftlog.get_cp_of_r(tr['k'], tr['phi'], '', 'local', 0, 1)
        return 0

    else:
        cp_tr, b = fftlog.get_cp_of_r(tr['k'], tr['dTdk'], '', 'FG2', 0, 1)
        #cp_tr=cp_tr[:,0]
        np.savetxt(argv.output_dir+'cpTr.txt', cp_tr.T)

        if argv.mode in ['Il', 'Am']:
            if argv.which=='all' and not argv.mode=='bin':
                if argv.lterm == 'noproj': which_list=['F2', 'G2']
                else: which_list=['F2', 'G2', 'dv2']
            else:
                which_list=[argv.which]

            for wh in which_list:
                print('computing {} for which={} Newton={} rad={}'.format(argv.mode, wh, argv.Newton, argv.rad))
                for ell in ell_list: 
                    bispectrum.get_Am_and_Il(chi_list, ell, wh, argv.Newton, argv.rad, time_dict,\
                            r0, ddr, normW, rmin,\
                            rmax, cp_tr, b, len(tr['k']), kmax, kmin, True)

        else:
            if argv.Newton and argv.rad:
                Newton_rad_list = [[0, 0], [1, 0], [0, 1]]
            else:
                Newton_rad_list = [[argv.Newton, argv.rad]]

            for Newton_rad in Newton_rad_list:
                Newton, rad = Newton_rad[0], Newton_rad[1]

                if argv.mode=='bin' and rad: rad_key='_rad'
                else: rad_key=''
                if argv.which=='all':
                    if argv.mode=='bin' or (argv.mode=='bl' and not rad):

                        #if argv.lterm == 'noproj': which_list=['F2{}'.format(rad_key), 'G2{}'.format(rad_key), \
                        #        'd2vd2v', 'd1vd1d', 'd2vd0d', 'd1vd3v']
                        #else: 
                        which_list=['F2{}'.format(rad_key), 'G2{}'.format(rad_key), \
                                'd2vd2v', 'd1vd1d', 'd2vd0d', 'd1vd3v',\
                                    'dv2{}'.format(rad_key), 'd1vd2v', 'd1vd0d', \
                                    'd1vdod', 'd0pd3v', 'd0pd1d', 'd1vd2p', 'davd1v'] #RG2
                    else:
                        if argv.lterm == 'noproj': which_list=['F2', 'G2']
                        else: 
                            which_list=['F2', 'G2', 'dv2']

                elif argv.which=='newton': 
                    which_list=['F2', 'G2', \
                                'd2vd2v', 'd1vd1d', 'd2vd0d', 'd1vd3v']
                else:
                    which_list=[argv.which+rad_key]

                for lt in lterm_list:
                    if argv.mode == 'bin':
                       print('Binning bispectrum...')
                       binning.get_binned_B(argv.bins, which_list, lt, Newton, rad)

                    elif argv.mode == 'bl':
                        for wh in which_list:
                            print('computing {} for which={} lterm={} ell={} Newton={} rad={}'\
                                    .format(argv.mode, wh, lt, argv.configuration, Newton, rad))


                            if argv.configuration=='esf':
                                config_list=['equi', 'squ', 'folded']
                            else:
                                config_list=[argv.configuration]

                            for config in config_list:
                                if config == 'equi':
                                    shape_name = '_equi'
                                elif config == 'squ':
                                    shape_name = '_squ'
                                elif config == 'folded':
                                    shape_name = '_folded'
                                else:
                                    shape_name=''

                                if rad and wh in ['F2', 'G2', 'dv2']:
                                    name=argv.output_dir+"bl/bl_{}_{}_rad{}".format(lt, wh, shape_name)
                                elif Newton:
                                    name=argv.output_dir+"bl/bl_{}_{}_newton{}".format(lt, wh, shape_name)
                                else:
                                    name=argv.output_dir+"bl/bl_{}_{}{}".format(lt, wh, shape_name)
                                if argv.Limber: name+='_Limber'
                                
                                print(' bispectrum file={}'.format(name))
                                if config in ['equi', 'squ', 'folded']:
                                    fich = open(name+'.txt', "w")
                                    for ell in ell_list:
                                        if config == 'squ':
                                            bl, wigner=bispectrum.spherical_bispectrum(wh, Newton, rad, argv.Limber, \
                                                    lt, argv.ell, ell, ell,\
                                                    time_dict, r0, ddr, normW, rmax, rmin, chi_list, cp_tr, b, \
                                                    len(tr['k']), kmax, kmin)
                                            if bl!=0: fich.write('{} {} {} {:.16e} {:.16e} \n'.format(argv.ell, ell, ell, bl, wigner))
                                        elif config == 'folded':
                                            bl, wigner=bispectrum.spherical_bispectrum(wh, Newton, rad, argv.Limber, \
                                                    lt, ell, ell, argv.ellmax,\
                                                    time_dict, r0, ddr, normW, rmax, rmin, chi_list, cp_tr, b, \
                                                    len(tr['k']), kmax, kmin)
                                            if bl!=0: fich.write('{} {} {} {:.16e} {:.16e} \n'.format(argv.ellmax, ell, ell, bl, wigner))
                                        else:
                                            bl, wigner=bispectrum.spherical_bispectrum(wh, Newton, rad, argv.Limber, \
                                                    lt, ell, ell, ell,\
                                                    time_dict, r0, ddr, normW, rmax, rmin, chi_list, cp_tr, b, \
                                                    len(tr['k']), kmax, kmin)
                                            if bl!=0: fich.write('{} {} {} {:.16e} {:.16e} \n'.format(ell, ell, ell, bl, wigner))

                                else:     
                                    ell1=argv.ell
                                    bispectrum.write_all_configuration(ell1, argv.ellmax, wh, lt, name, argv.Limber, rad, Newton, time_dict, chi_list,\
                                            r0, ddr, normW, rmin, rmax, cp_tr, b, len(tr['k']), kmax, kmin)


    return 0

if __name__ == "__main__":
    argv=arguments()
    ensure_directory_exists(argv.output_dir)
    write_args(argv)
    r=main(argv)
