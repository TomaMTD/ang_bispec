############################################################################# Parameters
force=True # force to recompute from scratch

which='all' # for cl: FG2, d2v, d1v, d3v, d1d, d0d,
            # for bl: F2, G2, d2vd2v, d1vd3v, d1vd1d, d2vd0d
            #       'RG2', 'd1vd2v', 'd1vd0d', 'd1vdod', 'davd1v', 'd0pd3v', 'd0pd1d', 'd1vd2p'
           
lterm='all' # each, all, density, rsd, pot or doppler

qterm=0 # 1, 2, 3, 4 only for which neq F2 G2

ell=4
ellmax=1024
Nell=16 #128
ell_spacing= 'log' # log

Nchi=301

bins = [2, 6, 14, 25, 44, 57, 73, 94, 120, 152, 194, 244, 294, 344, 394, 444, \
        494, 514] 

configuration='esf' # esf

Newton=False
rad=True

####################################################
### Window function
window_type=  'euclid' #'nbody' # ska
euclid_bin_idx = 9   # 0-9; bin edges [0.001,0.42,0.56,0.68,0.79,0.90,1.02,1.15,1.32,1.58,2.50]

#z0, dz= 0.50, 0.20
#ska bin2 = 0.24, 0.059
#ska bin1 = 0.50, 0.20

#sigma_input='redshift' #'distance'  #
#sigma_z=5e-3
#1e-3 for bin 2
#5e-3 for bin 1

####################################################
# input ska file#
# input_ska ='ska_data/SKAO_params_fcut100.txt'
fnl_local = 0

### output directory
output_dir = 'output_euclid_bin{}_fnl{}/'.format(euclid_bin_idx, fnl_local)
#'output_ska_z{}_dz{}_png/'.format(z0, dz)
#'output_ska_z{}_dz{}_review_c1c2/'.format(z0, dz)

####################################################
### Cosmology Planck 2018
h100=67.66
h=h100/100
omega_b=0.02242/h**2
omega_cdm=0.11933/h**2
omega_m =omega_cdm + omega_b
omega_r=9.16714e-05
omega_k=0
omega_l=1-omega_r-omega_cdm-omega_b
A_s=2.105e-9
n_s=0.9665
k_pivot=0.05
c=2.99792458
H0=100/c/10**5

