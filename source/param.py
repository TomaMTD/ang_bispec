############################################################################# Parameters
force=True

which='d0d' # for cl: FG2, d2v, d1v, d3v, d1d, d0d
            # for bl, F2, G2, d2vd2v, d1vd3v, d1vd1d, d2vd0d
            #     'RG2', 'd1vd2v', 'd1vd0d', 'd1vdod', 'd0pd3v', 'd0pd1d', 'd1vd2p'
           
lterm='all' # each, all, density, rsd, pot or doppler

qterm=0 # 1, 2, 3, 4 only for which neq F2 G2
gauge='new'

rad=False #True

z0, dz=2, 0.25
bb=25
zmin, zmax = z0-2.*dz, z0+2.*dz
Nchi=100
output_dir = 'output_z{}_dz{}_bb25/'.format(z0, dz)

equi = range(2, 128*2, 2)

relerr=1e-3
h=0.67556
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
