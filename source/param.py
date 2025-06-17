############################################################################# Parameters
force=False

which='all' # for cl: FG2, d2v, d1v, d3v, d1d, d0d
            # for bl, F2, G2, d2vd2v, d1vd3v, d1vd1d, d2vd0d
            #     'RG2', 'd1vd2v', 'd1vd0d', 'd1vdod', 'davd1v', 'd0pd3v', 'd0pd1d', 'd1vd2p'
           
lterm='all' # each, all, density, rsd, pot or doppler

qterm=0 # 1, 2, 3, 4 only for which neq F2 G2

ell=2
ellmax=514

Nchi=500
chi_ind = -1

#bins = [2, 6, 14, 25, 44, 57, 73, 94, 120, 152, 194, 244] #294, 344, 394, 444, \
#        #494, 544, 594, 644, 694, 744, 801, 901, 1001, 1101, 1201, 1301, 1401, \
#        #1501, 1601, 1701, 1801, 1901, 2000] 
bins = [2, 6, 14, 25, 44, 57, 73, 94, 120, 152, 194, 244, 294, 344, 394, 444, \
        494, 514] #544, 594, 644, 694, 744, 801, 901, 1001, 1101, 1201, 1301, 1401, \
        #1501, 1601, 1701, 1801, 1901, 2000] 

configuration='equi' #

Newton=False
rad=True
Limber=False

relerr=1e-2

####################################################
### Window function
z0, dz=2, 0.25
bb=25

#z0, dz=0.6, 0.05
#bb=10

####################################################
### output directory
output_dir = 'output_z{}_dz{}_final/'.format(z0, dz)
#'test/' #'output_z{}_dz{}_simpson/'.format(z0, dz)
#

####################################################
### Cosmology
h100=67.556 #=0.67556
h=h100/100
omega_b=0.0482754
omega_cdm=0.263771
omega_m =omega_cdm + omega_b
omega_r=9.16714e-05
omega_k=0
omega_l=1-omega_r-omega_cdm-omega_b
A_s =2.215e-9
n_s = 0.9619
k_pivot=0.05
c=2.99792458
H0=100/c/10**5
