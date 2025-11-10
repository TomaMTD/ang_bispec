import numpy as np
import os, sys, argparse
from art import text2art
from scipy import integrate
import time

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
    parser.add_argument('-z0'    , default=z0, type=float, help='center of redshift bin')
    parser.add_argument('-dz'    , default=dz, type=float, help='half width of redshift bin')
    parser.add_argument('-bb'    , default=bb, type=float, help='How fast is the window function decaying')
    parser.add_argument('-f'     , '--force', default=force, type=int, help='Wether you and to overwrite all output (force computation)')
    parser.add_argument('-Nchi'  , default=Nchi, type=int, help='Number of r/chi value to evaluate cl, Am and Il')
    parser.add_argument('-ell'   , default=ell, type=int, help='')
    parser.add_argument('-ellmax',default=ellmax, type=int, help='')
    parser.add_argument('-Nell',default=Nell, type=int, help='')
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


class parameters:
    def __init__(self, argv):
        self.which = argv.which
        self.lterm = argv.lterm
        self.qterm = argv.qterm
        self.Newton = argv.Newton
        self.rad = argv.rad
        self.z0 = argv.z0
        self.dz = argv.dz
        self.bb = argv.bb
        self.force = argv.force
        self.Nchi = argv.Nchi
        self.ell = argv.ell
        self.Nell = argv.Nell
        self.ellmax = argv.ellmax
        self.output_dir = argv.output_dir
        self.mode = argv.mode
        self.configuration = argv.configuration




def main(argv):

    import lincosmo 
    import fftlog
    import general_ps
    import bispectrum
    import binning
    import fctr
    
    if argv.force!=0: print('-force is activated, overwritting files')

    p=parameters(argv)

    Wrmin, Wrmax = lincosmo.get_distance(argv.z0-argv.dz)[0], lincosmo.get_distance(argv.z0+argv.dz)[0]
    rmin, rmax = Wrmin-15*bb, Wrmax+15*bb

    time_dict = lincosmo.growth_fct()
    np.save(output_dir+'time_dict', time_dict)

    # Prepare H/a data for window normalization: (ra_grid, H_over_a_values)
    H_over_a_data = (time_dict['ra'], time_dict['Ha'] / time_dict['a'])
    window_args = (Wrmin, Wrmax, H_over_a_data, bb)

    print('Window function limits: rmin={:.0f} rmax={:.0f}'.format(rmin, rmax))

    tr, Pk = lincosmo.get_power(0)
    kmin, kmax = np.min(tr['k']), np.max(tr['k'])

    chi_list=np.linspace(rmin, rmax, argv.Nchi)
    r_list  =np.linspace(rmin, rmax, argv.Nchi)
    t_grid = np.linspace(rmin/rmax, rmax/rmin, 1000)

    if argv.ellmax<=argv.ell:
        ell_list=np.array([argv.ell])
    else:
        log_vals = np.logspace(np.log10(argv.ell), np.log10(argv.ellmax), num=argv.Nell)  # Adjust `num` as needed
        # Round to nearest integer and remove duplicates
        ell_list = np.unique(np.round(log_vals).astype(int))

        # For specific configurations, ensure all ells are even
        if argv.configuration in ['equi', 'squ', 'folded']:
            # Add 1 to odd ells to make them even
            ell_list = np.array([ell if ell % 2 == 0 else ell + 1 for ell in ell_list])
            # Remove duplicates again in case some became the same
            ell_list = np.unique(ell_list)
        #if argv.mode in ['cl', 'cln', 'Cl', 'Cln']:
        #else:
        #    if argv.configuration in ['equi', 'squ', 'folded', 'esf']:
        #        if argv.ell%2!=0: argv.ell+=1
        #        ell_list=np.arange(argv.ell, argv.ellmax, 2)
        #    else:
        #        ell_list=np.arange(argv.ell, argv.ellmax, 1)



    # Define lterm_list based on p.lterm
    if p.Newton:
        if p.lterm == 'all':
            lterm_list = ['density', 'rsd', 'doppler', 'pot', 'dpot']
        else:
            lterm_list = [p.lterm]
    elif p.lterm == 'all':
        lterm_list = ['density', 'rsd', 'doppler', 'pot', 'dpot', 'pot_gr']
    else:
        lterm_list = [p.lterm]

    # =========================================================================
    # Precompute window derivatives ONCE (expensive operation ~18 seconds!)
    # This is cached to disk and reused by all subsequent computations
    # =========================================================================
    print('Loading/computing window function derivatives...')
    W_derivs_list = fctr.load_or_compute_window_derivatives(window_args, r_list, output_dir, max_deriv=11)

    if argv.mode in ['cl', 'cln', 'Cl', 'Cln']:
        if argv.which=='all':
            which_list=['FG2', 'd2v', 'd1v', 'd3v', 'd1d', 'F2', 'G2', 'dv2']
        else:
            which_list=[argv.which]

        print('Computing generalised power spectra for:')
        print('     which={}'.format(which_list))
        print('     ell_list={}'.format(ell_list))
        print('     Newton={}'.format(argv.Newton))
        print('     radiation={}'.format(argv.rad))

        for p.which in which_list:
            # Compute fctr and cp dicts organized by lterm
            fctr_dict = fctr.fct_of_r_analytical(p, ell_list, r_list, time_dict, window_args, lterm_list)
            np.save(argv.output_dir+'fctr_of_r_{}'.format(p.which), fctr_dict)

            cp_dict = fftlog.apply_fftlog_dict(tr['k'], tr['dTdk'] if argv.rad else Pk, p, lterm_list)
            np.save(f'cp_{p.which}', cp_dict)

            general_ps.compute_integral_generalized(p, ell_list, chi_list, r_list, t_grid, cp_dict, fctr_dict)


    else:
        if argv.which=='all':
            which_list=['F2', 'G2', 'dv2', 'd2vd2v', 'd1vd3v', 'd1vd1d', 'd2vd0d', 'd1vd2v', 'd1vd0d', 'd1vdod', 'davd1v', 'd0pd3v', 'd0pd1d', 'd1vd2p']
        elif argv.which=='rad' and p.rad:
            which_list=['F2', 'G2', 'dv2']
        else:
            which_list=[argv.which]

        for p.which in which_list:
            bispectrum.compute_all_bispectra_efficient(p, ell_list, chi_list, time_dict, window_args, lterm_list,
                                                   W_derivs_list=W_derivs_list, tr=tr, Pk=Pk, t_grid=t_grid)

        #cp_tr, b = fftlog.get_cp_of_r(tr['k'], tr['dTdk'], '', 'FG2', 0, 1)
        ##cp_tr=cp_tr[:,0]
        #np.savetxt(argv.output_dir+'cpTr.txt', cp_tr.T)
        #if argv.Newton and argv.rad:
        #    Newton_rad_list = [[0, 0], [1, 0], [0, 1]]
        #else:
        #    Newton_rad_list = [[argv.Newton, argv.rad]]

        #for Newton_rad in Newton_rad_list:
        #    Newton, rad = Newton_rad[0], Newton_rad[1]

        #    if argv.mode=='bin' and rad: rad_key='_rad'
        #    else: rad_key=''
        #    if argv.which=='all':
        #        if argv.mode=='bin' or (argv.mode=='bl' and not rad):

        #            #if argv.lterm == 'noproj': which_list=['F2{}'.format(rad_key), 'G2{}'.format(rad_key), \
        #            #        'd2vd2v', 'd1vd1d', 'd2vd0d', 'd1vd3v']
        #            #else: 
        #            which_list=['F2{}'.format(rad_key), 'G2{}'.format(rad_key), \
        #                    'd2vd2v', 'd1vd1d', 'd2vd0d', 'd1vd3v',\
        #                        'dv2{}'.format(rad_key), 'd1vd2v', 'd1vd0d', \
        #                        'd1vdod', 'd0pd3v', 'd0pd1d', 'd1vd2p', 'davd1v'] #RG2
        #        else:
        #            if argv.lterm == 'noproj': which_list=['F2', 'G2']
        #            else: 
        #                which_list=['F2', 'G2', 'dv2']

        #    elif argv.which=='newton': 
        #        which_list=['F2', 'G2', \
        #                    'd2vd2v', 'd1vd1d', 'd2vd0d', 'd1vd3v']
        #    else:
        #        which_list=[argv.which+rad_key]

        #    for lt in lterm_list:
        #        if argv.mode == 'bin':
        #           print('Binning bispectrum...')
        #           binning.get_binned_B(argv.bins, which_list, lt, Newton, rad)

        #        elif argv.mode == 'bl':
        #            for wh in which_list:
        #                print('computing {} for which={} lterm={} ell={} Newton={} rad={}'\
        #                        .format(argv.mode, wh, lt, argv.configuration, Newton, rad))


        #                if argv.configuration=='esf':
        #                    config_list=['equi', 'squ', 'folded']
        #                else:
        #                    config_list=[argv.configuration]

        #                for config in config_list:
        #                    if config == 'equi':
        #                        shape_name = '_equi'
        #                    elif config == 'squ':
        #                        shape_name = '_squ'
        #                    elif config == 'folded':
        #                        shape_name = '_folded'
        #                    else:
        #                        shape_name=''

        #                    if rad and wh in ['F2', 'G2', 'dv2']:
        #                        name=argv.output_dir+"bl/bl_{}_{}_rad{}".format(lt, wh, shape_name)
        #                    elif Newton:
        #                        name=argv.output_dir+"bl/bl_{}_{}_newton{}".format(lt, wh, shape_name)
        #                    else:
        #                        name=argv.output_dir+"bl/bl_{}_{}{}".format(lt, wh, shape_name)
        #                    
        #                    print(' bispectrum file={}'.format(name))
        #                    if config in ['equi', 'squ', 'folded']:
        #                        fich = open(name+'.txt', "w")
        #                        for ell in ell_list:
        #                            if config == 'squ':
        #                                bl, wigner=bispectrum.spherical_bispectrum(wh, Newton, rad, \
        #                                        lt, argv.ell, ell, ell,\
        #                                        time_dict, r0, ddr, normW, rmax, rmin, chi_list, cp_tr, b, \
        #                                        len(tr['k']), kmax, kmin)
        #                                if bl!=0: fich.write('{} {} {} {:.16e} {:.16e} \n'.format(argv.ell, ell, ell, bl, wigner))
        #                            elif config == 'folded':
        #                                bl, wigner=bispectrum.spherical_bispectrum(wh, Newton, rad, \
        #                                        lt, ell, ell, argv.ellmax,\
        #                                        time_dict, r0, ddr, normW, rmax, rmin, chi_list, cp_tr, b, \
        #                                        len(tr['k']), kmax, kmin)
        #                                if bl!=0: fich.write('{} {} {} {:.16e} {:.16e} \n'.format(argv.ellmax, ell, ell, bl, wigner))
        #                            else:
        #                                bl, wigner=bispectrum.spherical_bispectrum(wh, Newton, rad, \
        #                                        lt, ell, ell, ell,\
        #                                        time_dict, r0, ddr, normW, rmax, rmin, chi_list, cp_tr, b, \
        #                                        len(tr['k']), kmax, kmin)
        #                                if bl!=0: fich.write('{} {} {} {:.16e} {:.16e} \n'.format(ell, ell, ell, bl, wigner))

        #                    else:     
        #                        ell1=argv.ell
        #                        bispectrum.write_all_configuration(ell1, argv.ellmax, wh, lt, name, rad, Newton, time_dict, chi_list,\
        #                                r0, ddr, normW, rmin, rmax, cp_tr, b, len(tr['k']), kmax, kmin)


    return 0

if __name__ == "__main__":
    argv=arguments()
    ensure_directory_exists(argv.output_dir)
    write_args(argv)
    r=main(argv)
