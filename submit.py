import numpy as np
import h5py
import os
import sys

ell0=2
core=128
ellmax= core+ell0
ell_list = np.arange(ell0, ellmax*2, 2)
Nell=len(ell_list)
Nnode = 1
Nproc = Nnode*core

Nell_per_proc = Nell//Nproc
lenght = Nell_per_proc
script=''


# for cl: FG2, d2v, d1v, d3v, d1d, d0d
which=['d3v', 'd2v'] #['FG2', 'd3v', 'd2v', 'd1v', 'd1d']
lterm=['rsd', 'doppler'] #['density', 'rsd', 'doppler', 'pot']
qterm= [[0], [0], [0], [0], [0]]

for w in which:
    for l in lterm:
        for q_list in qterm:
            for q in q_list:
                ind=0

                script=''
                for ell_i in range(0, Nproc):
                    if ell_i < Nell%Nproc:
                        lenght2 = lenght+1
                    else:
                        lenght2 = lenght+0

                    script += str(ell_i)+r"""    python -u byspectrum.py -w {} -l {} -q {} cl """.format(w, l, q)
                    for i in range(lenght2):
                        ell = ell_list[ind]
                        #if not os.path.isfile('output/Cln_ell{}.txt'.format(ell)): print('pas trouve {}'.format(ell))
                        script += """ {} """.format(ell)
                        ind+=1
                    script += '\n'
                ell_i+=1

                text_file = open('script/silly_w{}_l{}_q{}.conf'.format(w, l, q), "w")
                text_file.write(script)
                text_file.close()

                batch = r"""#!/bin/bash
                
#SBATCH -N """+str(Nnode)+r"""
#SBATCH --ntasks="""+str(ell_i)+r"""
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --job-name=w{}_l{}_q{}

source $HOME/myenv/bin/activate

srun -n """.format(w, l, q)+str(ell_i)+""" -l --multi-prog script/silly_w{}_l{}_q{}.conf
""".format(w, l, q)

                text_file = open('script/script_integral_w{}_l{}_q{}.sh'.format(w, l, q), "w")
                text_file.write(batch)
                text_file.close()

                #os.system("sbatch {}".format('script_integral_w{}_l{}_q{}.sh'.format(w, l, q)))

