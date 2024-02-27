import numpy as np
import h5py
import os
import sys



#the_ell_list=[13]
#chi0_list=[97]
#
#
#
## for cl: FG2, d2v, d1v, d3v, d1d, d0d
#which=['d3v'] #['FG2', 'd3v', 'd2v', 'd1v', 'd1d']
#lterm=['rsd'] #['density', 'rsd', 'doppler', 'pot']
#qterm= [[0]]


#which       =['d3v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v']
#lterm       =['doppler', 'rsd', 'doppler', 'rsd', 'pot', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd']
#the_ell_list=[2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 9, 11, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]
#chi0_list   =[121, 68, 196, 125, 186, 95, 72, 159, 167, 75, 188, 196, 81, 89, 104, 110, 115, 122, 131, 141, 149, 163, 168, 184, 195]

#which       =['FG2', 'd1v', 'd1v', 'd1v', 'd1v', 'd1v', 'd1v', 'd1v', 'd1v', 'd1v']
#lterm       =['rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd']
#the_ell_list=[2, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#chi0_list   =[192, 92, 106, 115, 126, 138, 156, 175, 189, 197]

which       =['d3v', 'd3v', 'd3v', 'd3v', 'd2v', 'd2v', 'd3v', 'd3v', 'd3v', 'd3v', 'd2v', 'd2v', 'd3v', 'd3v', 'd3v', 'd3v', 'd2v', 'd2v', 'd3v', 'd3v', 'd3v', 'd3v', 'd2v', 'd2v', 'd3v', 'd3v', 'd3v', 'd3v', 'd2v', 'd2v', 'd3v', 'd3v', 'd3v', 'd3v', 'd2v', 'd2v', 'd3v', 'd3v', 'd3v', 'd2v', 'd2v', 'd3v', 'd3v', 'd3v', 'd2v', 'd2v', 'd3v', 'd3v', 'd3v', 'd2v', 'd2v', 'd3v', 'd3v', 'd3v', 'd2v', 'd2v', 'd3v', 'd3v', 'd3v', 'd2v', 'd2v', 'd3v', 'd3v', 'd3v', 'd2v', 'd2v', 'd3v', 'd3v', 'd3v', 'd2v', 'd2v', 'd3v', 'd3v', 'd3v', 'd2v', 'd2v', 'd3v', 'd3v', 'd3v', 'd2v', 'd2v', 'd3v', 'd3v', 'd3v', 'd2v', 'd2v', 'd3v', 'd3v', 'd3v', 'd2v', 'd2v', 'd3v', 'd3v', 'd3v', 'd2v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd2v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v', 'd3v']
lterm       =['density', 'rsd', 'pot', 'doppler', 'rsd', 'doppler', 'density', 'rsd', 'pot', 'doppler', 'rsd', 'doppler', 'density', 'rsd', 'pot', 'doppler', 'rsd', 'doppler', 'density', 'rsd', 'pot', 'doppler', 'rsd', 'doppler', 'density', 'rsd', 'pot', 'doppler', 'rsd', 'doppler', 'density', 'rsd', 'pot', 'doppler', 'rsd', 'doppler', 'rsd', 'pot', 'doppler', 'rsd', 'doppler', 'rsd', 'pot', 'doppler', 'rsd', 'doppler', 'rsd', 'pot', 'doppler', 'rsd', 'doppler', 'rsd', 'pot', 'doppler', 'rsd', 'doppler', 'rsd', 'pot', 'doppler', 'rsd', 'doppler', 'rsd', 'pot', 'doppler', 'rsd', 'doppler', 'rsd', 'pot', 'doppler', 'rsd', 'doppler', 'rsd', 'pot', 'doppler', 'rsd', 'doppler', 'rsd', 'pot', 'doppler', 'rsd', 'doppler', 'rsd', 'pot', 'doppler', 'rsd', 'doppler', 'rsd', 'pot', 'doppler', 'rsd', 'doppler', 'rsd', 'pot', 'doppler', 'rsd', 'doppler', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'rsd', 'doppler', 'rsd', 'doppler', 'rsd', 'doppler', 'rsd', 'doppler', 'rsd', 'doppler', 'rsd', 'doppler', 'rsd', 'doppler', 'rsd', 'doppler', 'rsd', 'doppler', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd', 'rsd']
the_ell_list=[2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 49, 50, 50, 51, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108]
chi0_list   =[115, 18, 48, 18, 26, 57, 132, 18, 61, 22, 29, 73, 144, 18, 75, 24, 33, 84, 170, 18, 85, 28, 37, 92, 179, 18, 92, 30, 41, 97, 195, 18, 95, 32, 45, 102, 19, 98, 36, 49, 108, 20, 100, 38, 54, 115, 21, 103, 41, 57, 122, 22, 106, 44, 60, 126, 23, 109, 48, 64, 119, 24, 121, 51, 68, 133, 26, 135, 54, 71, 140, 27, 155, 56, 76, 164, 29, 174, 58, 80, 182, 30, 187, 59, 84, 190, 31, 192, 60, 87, 193, 32, 196, 61, 89, 196, 33, 60, 90, 34, 60, 92, 37, 64, 94, 39, 72, 98, 40, 74, 100, 42, 80, 102, 43, 83, 103, 44, 87, 103, 45, 91, 106, 45, 94, 109, 46, 95, 112, 45, 97, 115, 48, 98, 118, 48, 99, 120, 50, 102, 122, 50, 104, 125, 53, 107, 134, 56, 111, 145, 61, 114, 149, 65, 121, 169, 66, 122, 177, 67, 126, 194, 70, 127, 197, 70, 136, 74, 150, 78, 164, 82, 168, 85, 186, 87, 188, 88, 190, 92, 193, 93, 196, 96, 97, 99, 100, 101, 101, 102, 102, 103, 103, 103, 103, 104, 103, 106, 107, 108, 109, 109, 110, 110, 111, 111, 113, 115, 116, 117, 119, 120, 121, 122, 122, 124, 125, 127, 129, 134, 138, 136, 140, 149, 151, 151, 155, 163, 166, 170, 173, 177, 176, 183, 186, 184, 188, 193, 193, 188]




Nchi=200
core=128
script=''


start = 30
end   = 50
for indell,the_ell in enumerate(the_ell_list[start:end]):
    indell+=start
    chi_list = np.arange(chi0_list[indell], 200)
    Nchi=len(chi_list)
    Nnode=Nchi//core + 1

    w=which[indell]
    l=lterm[indell]

    q_list=[0]# qterm[indell]
    for q in q_list:
        ind=0
    
        script=''
        for chi_i in range(0, Nchi):
    
            script += str(chi_i)+r"""    python -u byspectrum.py -w {} -l {} -q {} -i {} cl {}""".format(w, l, q, chi_list[chi_i], the_ell)
            script += '\n'
        chi_i+=1
    
        text_file = open('script/silly_w{}_l{}_q{}_ell{}_i.conf'.format(w, l, q, the_ell), "w")
        text_file.write(script)
        text_file.close()
    
        batch = r"""#!/bin/bash
                
#SBATCH -N """+str(Nnode)+r"""
#SBATCH --ntasks="""+str(chi_i)+r"""
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --job-name=w{}_l{}_q{}_ell{}

source $HOME/myenv/bin/activate

srun -n """.format(w, l, q, the_ell)+str(chi_i)+""" -l --multi-prog script/silly_w{}_l{}_q{}_ell{}_i.conf
""".format(w, l, q, the_ell)

        text_file = open('script/script_integral_w{}_l{}_q{}_ell{}_i.sh'.format(w, l, q, the_ell), "w")
        text_file.write(batch)
        text_file.close()
    
        os.system("sbatch {}".format('script/script_integral_w{}_l{}_q{}_ell{}_i.sh'.format(w, l, q, the_ell)))
        print("sbatch {}".format('script/script_integral_w{}_l{}_q{}_ell{}_i.sh'.format(w, l, q, the_ell)))

