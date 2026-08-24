import numpy as np
import os, sys, argparse, importlib
from art import text2art
from scipy import integrate
import time

import path
sys.path.insert(1, path.path+'/ang_bispec/source')
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
    parser.add_argument('-z0'     , default=globals().get('z0',     None), type=float, help='center of redshift bin')
    parser.add_argument('-dz'     , default=globals().get('dz',     None), type=float, help='half width of redshift bin')
    parser.add_argument('-sigma_z', default=globals().get('sigma_z', None), type=float, help='How fast is the window function decaying')
    parser.add_argument('-window_type'    , default=window_type, type=str, help='Type of window function, eg nbody, ska, euclid')
    parser.add_argument('-euclid_bin_idx' , default=euclid_bin_idx, type=int, help='Euclid photometric bin index (0-9)')
    parser.add_argument('-f'     , '--force', default=force, type=int, help='Wether you and to overwrite all output (force computation)')
    parser.add_argument('-Nchi'  , default=Nchi, type=int, help='Number of r/chi value to evaluate cl, Am and Il')
    parser.add_argument('-ell'   , default=ell, type=int, help='')
    parser.add_argument('-ellmax',default=ellmax, type=int, help='')
    parser.add_argument('-Nell',default=Nell, type=int, help='')
    parser.add_argument('-o', '--output_dir', default=output_dir+'/', type=str, help='path of output')
    parser.add_argument('-m', '--mode', default='bl', type=str, help='Computation mode: [cl, Il, bl, bin, merge]')
    parser.add_argument('-config', '--configuration', default=configuration, type=str, help='what triangle configuration to compute')

    parser.add_argument('-h100'     , type=float,default=h100) 
    parser.add_argument('-omega_b'  , type=float,default=omega_b) 
    parser.add_argument('-omega_cdm', type=float,default=omega_cdm) 
    parser.add_argument('-omega_m'  , type=float,default=omega_m)
    parser.add_argument('-omega_r'  , type=float,default=omega_r)
    parser.add_argument('-omega_k'  , type=float,default=omega_k)
    parser.add_argument('-omega_l'  , type=float,default=omega_l)
    parser.add_argument('-A_s'      , type=float,default=A_s) 
    parser.add_argument('-n_s'      , type=float,default=n_s)
    parser.add_argument('-fnl_local'      , type=float,default=fnl_local) 
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

def build_lterm_list(lterm, Newton):
    """Expand the -l argument into the list of linear terms.

    Depends on Newton (GR adds pot_gr), so it must be rebuilt for every
    (rad, Newton) pass rather than once from the command line.
    """
    if '+' in lterm:            # checked first: the Newton branch below has no '+' handling
        return lterm.split('+')
    elif Newton:
        if lterm == 'all':
            return ['density', 'rsd', 'doppler', 'pot', 'dpot', 'pot_fnl', 'lensing']
        elif lterm == 'noproj':
            return ['density', 'rsd', 'pot_fnl', 'lensing']
        else:
            return [lterm]
    elif lterm == 'all':
        return ['density', 'rsd', 'doppler', 'pot', 'dpot', 'pot_gr', 'pot_fnl', 'lensing']
    elif lterm == 'noproj':
        return ['density', 'rsd', 'pot_gr', 'pot_fnl', 'lensing']
    else:
        return [lterm]


def ensure_directory_exists(path):
    """checks wether the output path exists"""
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path+'/bl')
        print(f"output dir created : {path}")


class parameters:
    def __init__(self, argv):
        self.which = argv.which
        self.lterm = argv.lterm
        self.qterm = argv.qterm
        self.Newton = argv.Newton
        self.rad = argv.rad
        self.fnl_local = argv.fnl_local
        self.z0 = argv.z0
        self.dz = argv.dz
        self.sigma_z = argv.sigma_z
        self.window_type = argv.window_type
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
    import mathematica

    if argv.force!=0: print('-force is activated, overwritting files')

    p=parameters(argv)

    # Handle merge mode early (doesn't need cosmology setup)
    if argv.mode in ['merge']:
        print('='*70)
        print('MERGE MODE: Merging fallback files into HDF5')
        print('='*70)
        general_ps.merge_fallback_files(argv.output_dir)
        return 0

    _ska_input = globals().get('input_ska', 0) if argv.window_type == 'ska' else 0
    time_dict = lincosmo.growth_fct(input_data=_ska_input, window_type=argv.window_type)
    np.save(output_dir+'time_dict', time_dict)

    if argv.window_type == 'nbody':
        Wrmin, Wrmax = lincosmo.get_distance(argv.z0-argv.dz)[0], \
                       lincosmo.get_distance(argv.z0+argv.dz)[0]
        if sigma_input == 'redshift':
            argv.sigma_z = (lincosmo.get_distance(argv.z0+argv.sigma_z/2)[0]
                            - lincosmo.get_distance(argv.z0-argv.sigma_z/2)[0])
        rmin, rmax = Wrmin-30*argv.sigma_z, Wrmax+30*argv.sigma_z
        H_over_a_data = (time_dict['ra'], time_dict['Ha'] / time_dict['a'])
        window_args = (Wrmin, Wrmax, H_over_a_data, argv.sigma_z)
        print('Integration range: rmin={:.0f} Mpc/h, rmax={:.0f} Mpc/h'.format(rmin, rmax))
        print('Window: Wrmin={:.0f}, Wrmax={:.0f}, sigma_z={:.2f} Mpc/h'.format(Wrmin, Wrmax, argv.sigma_z))

    elif argv.window_type == 'euclid':
        _BIN_EDGES = [0.001, 0.56, 0.79, 1.02, 1.32, 2.50]
        _zlo = _BIN_EDGES[argv.euclid_bin_idx]
        _zhi = _BIN_EDGES[argv.euclid_bin_idx + 1]
        rmin = lincosmo.get_distance(max(0.015, _zlo - 0.5))[0]
        rmax = lincosmo.get_distance(min(3.5, _zhi + 1))[0]
        H_over_a_data = (time_dict['ra'], time_dict['Ha'] / time_dict['a'])
        window_args = (argv.euclid_bin_idx, H_over_a_data)
        print('Euclid bin {} [z={:.3f}, {:.3f}]: rmin={:.0f}, rmax={:.0f} Mpc/h'.format(
              argv.euclid_bin_idx, _zlo, _zhi, rmin, rmax))

    else:  # ska
        rmin, rmax = lincosmo.get_distance(argv.z0-argv.dz)[0], \
                     lincosmo.get_distance(argv.z0+argv.dz)[0]
        if sigma_input == 'redshift':
            argv.sigma_z = (lincosmo.get_distance(argv.z0+argv.sigma_z/2)[0]
                            - lincosmo.get_distance(argv.z0-argv.sigma_z/2)[0])
        Wrmin, Wrmax = rmin+20*argv.sigma_z, rmax-20*argv.sigma_z
        H_over_a_data = (time_dict['ra'], time_dict['Ha'] / time_dict['a'])
        if 'data' in time_dict:
            n_angular_data = (time_dict['data']['r'], time_dict['data']['n_angular'])
            print('  Found n(z) angular data for SKA-type window normalization')
        else:
            n_angular_data = None
        window_args = (Wrmin, Wrmax, H_over_a_data, argv.sigma_z, n_angular_data)
        print('Integration range: rmin={:.0f} Mpc/h, rmax={:.0f} Mpc/h, Dz={:.2f} Mpc/h'.format(rmin, rmax, (rmax-rmin)/2))
        print('Window function limits: Wrmin={:.0f} Mpc/h, Wrmax={:.0f} Mpc/h, Dz={:.2f} Mpc/h'.format(Wrmin, Wrmax, (Wrmax-Wrmin)/2))
        print('Window decay scale sigma_z={:.2f} Mpc/h, {:.2f}\\% of the window function size'.format(argv.sigma_z, 100*argv.sigma_z/(Wrmax-Wrmin)*2))

    # after the branch: ska derives its window edges from rmin, so overriding earlier moves the window
    rmin = rmin_global
    print('Radial grid: rmin={:.0f} Mpc/h (rmin_global), rmax={:.0f} Mpc/h'.format(rmin, rmax))

    tr, Pk = lincosmo.get_power(0)

    # Keep r_list EVENLY spaced: general_ps.cubic_interp_uniform looks y1 up on it by index
    # arithmetic rather than by searching, so a log-spaced grid here would silently give
    # wrong numbers instead of raising.
    chi_list=np.linspace(rmin, rmax, argv.Nchi)
    r_list  =np.linspace(rmin, rmax, argv.Nchi)

    if argv.ellmax<=argv.ell:
        ell_list=np.array([argv.ell])
    else:
        if ell_spacing=='log':
            log_vals = np.logspace(np.log10(argv.ell), np.log10(argv.ellmax), num=argv.Nell)  # Adjust `num` as needed
            # Round to nearest integer and remove duplicates
            ell_list = np.unique(np.round(log_vals).astype(int))
        else:
            ell_list = np.array(range(argv.ell, argv.ellmax, 1), dtype=np.int64)

        # For specific configurations, ensure all ells are even
        if argv.configuration in ['equi', 'squ', 'squ2', 'folded', 'esf']:
            # Add 1 to odd ells to make them even
            ell_list = np.array([ell if ell % 2 == 0 else ell + 1 for ell in ell_list])
            # Remove duplicates again in case some became the same
            ell_list = np.unique(ell_list)

    # t grid, per ell, concentrated on the support of I_ell (shape (n_t, n_ell))
    t_grid = mathematica.build_t_grid(ell_list, rmin, rmax, tr['k'])

    # -r 1 -N 1 has no meaning as a single run: it is the shorthand for "do the three
    # physical cases", GR (0,0), radiation (1,0) and Newtonian (0,1), in one go.
    # 'cl' and the primordial shapes ignore rad/Newton, so they stay a single GR pass.
    if p.rad and p.Newton:
        if argv.which in ('cl', 'local', 'equi', 'ortho', 'primordial'):
            rad_Newton_list = [[0, 0]]
        else:
            rad_Newton_list = [[0, 0], [1, 0], [0, 1]]
        print('-rad and -Newton both set: running passes (rad, Newton) = {}'.format(
              [tuple(rN) for rN in rad_Newton_list]))
    else:
        rad_Newton_list = [[p.rad, p.Newton]]

    # =========================================================================
    # Precompute window derivatives ONCE (expensive operation ~18 seconds!)
    # This is cached to disk and reused by all subsequent computations
    # =========================================================================
    print('Loading/computing window function derivatives...')
    W_derivs_list, W_lens_derivs_list = fctr.load_or_compute_window_derivatives(p, window_args, r_list, output_dir, max_deriv=11)

    if argv.mode in ['cl', 'cln', 'Cl', 'Cln']:
        if argv.which=='cl':
            p.rad, p.Newton = rad_Newton_list[0]
            lterm_list = build_lterm_list(p.lterm, p.Newton)
            Cl = general_ps.compute_power_spectrum(p, ell_list, r_list, time_dict, window_args, lterm_list,
                                                   W_derivs_list, W_lens_derivs_list)

        else:
            for p.rad, p.Newton in rad_Newton_list:
                lterm_list = build_lterm_list(p.lterm, p.Newton)

                if argv.which=='all':
                    if p.rad:
                        which_list=['F2', 'G2', 'dv2']
                    else:
                        which_list=['FG2', 'd2v', 'd1v', 'd3v', 'd1d', 'F2', 'G2', 'dv2']

                elif argv.which in ['F2', 'G2', 'dv2']:
                    which_list=[argv.which]

                elif argv.which=='primordial':
                    p.rad = 0
                    p.Newton = 0
                    which_list=['primordial']

                else:
                    which_list=[argv.which]

                print('Computing generalised power spectra for:')
                print('     which={}'.format(which_list))
                print('     lterm={}'.format(lterm_list))
                print('     ell_list={}'.format(ell_list))
                print('     Newton={}'.format(p.Newton))
                print('     radiation={}'.format(p.rad))

                for p.which in which_list:
                    print('='*70)
                    print('='*70)
                    print(f'Processing which={p.which} (rad={p.rad}, Newton={p.Newton})')
                    # Compute fctr and cp dicts organized by lterm
                    fctr_dict = fctr.fct_of_r_analytical(p, ell_list, r_list, time_dict, window_args, lterm_list,
                                                         W_derivs_list=W_derivs_list, W_lens_derivs_list=W_lens_derivs_list)
                    np.save(argv.output_dir+'fctr_of_r_{}'.format(p.which), fctr_dict)

                    if p.which == 'primordial':
                        fctk = tr['phi']
                    elif p.rad and p.which in ['F2', 'G2', 'dv2']:
                        fctk = tr['dTdk']
                    else:
                        fctk = Pk

                    cp_dict = fftlog.apply_fftlog_dict(tr['k'], fctk, p)
                    np.save(argv.output_dir+f'cp_{p.which}', cp_dict)

                    general_ps.compute_integral_generalized(p, ell_list, chi_list, r_list, t_grid, cp_dict, fctr_dict, lterm_list)

    else:

        if argv.configuration=='esf':
            config_list = ['equi', 'squ', 'folded', 'squ2']
        elif argv.configuration=='es':
            config_list = ['equi', 'squ']
        else:
            config_list = [argv.configuration]

        # Primordial shapes, now under -m bl like everything else. local uses potential
        # transfer (tr['phi']) with no rad/Newton. -w local computes local only; equi/ortho/
        # primordial all compute the three blocks and save the combined shapes.
        if argv.which in ('local', 'equi', 'ortho', 'primordial'):
            p.rad = 0
            p.Newton = 0
            lterm_list = build_lterm_list(p.lterm, p.Newton)
            for p.configuration in config_list:
                print(f'Computing bispectrum for config={p.configuration}')
                p.which = argv.which
                if argv.which == 'local':
                    bispectrum.compute_all_bispectra_efficient(p, ell_list, chi_list, time_dict, window_args, lterm_list,
                                                       W_derivs_list=W_derivs_list, W_lens_derivs_list=W_lens_derivs_list, tr=tr, Pk=tr['phi'], t_grid=t_grid)
                else:
                    bispectrum.get_all_primordial_shapes(p, ell_list, chi_list, time_dict, window_args, lterm_list,
                                                       W_derivs_list=W_derivs_list, W_lens_derivs_list=W_lens_derivs_list, tr=tr, Pk=tr['phi'], t_grid=t_grid)
                print(f' ')
            return 0

        for p.rad, p.Newton in rad_Newton_list:
            # lterm_list depends on Newton (pot_gr is GR only), so rebuild it per pass
            lterm_list = build_lterm_list(p.lterm, p.Newton)

            if argv.which=='all':
                if p.rad:
                    which_list=['F2', 'G2', 'dv2']
                elif p.lterm == 'noproj':
                    which_list=['F2', 'G2', 'd2vd2v', 'd1vd3v', 'd1vd1d', 'd0dd0d', 'd2vd0d']
                else:    
                    which_list=['F2', 'G2', 'd2vd2v', 'd1vd3v', 'd1vd1d', 'd0dd0d', \
                                'dv2', 'd2vd0d', 'd1vd2v', 'd1vd0d', 'd1vdod', 'davd1v',\
                                'd0pd3v', 'd0pd1d', 'd1vd2p']
            else:
                which_list=[argv.which]

            for p.configuration in config_list:
                print(f'Computing bispectrum for config={p.configuration}')
                for p.which in which_list:
                    bispectrum.compute_all_bispectra_efficient(p, ell_list, chi_list, time_dict, window_args, lterm_list,
                                                       W_derivs_list=W_derivs_list, W_lens_derivs_list=W_lens_derivs_list, tr=tr, Pk=Pk, t_grid=t_grid)
                print(f' ')


    return 0

if __name__ == "__main__":
    argv=arguments()
    ensure_directory_exists(argv.output_dir)
    write_args(argv)
    importlib.invalidate_caches()
    r=main(argv)
