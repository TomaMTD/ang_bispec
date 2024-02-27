import numpy as np
import h5py
import os
import sys

kern='dv2'
squ = 10


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

ind=0
script=''
for ell_i in range(0, Nproc):
    if ell_i < Nell%Nproc:
        lenght2 = lenght+1
    else:
        lenght2 = lenght+0

    script += str(ell_i)+r"""    python -u byspectrum.py -w {} -l all -squ {} Il """.format(kern, squ)
    for i in range(lenght2):
        ell = ell_list[ind]
        script += """ {} """.format(ell)
        ind+=1
    script += '\n'
ell_i+=1

text_file = open('script/silly_il.conf', "w")
text_file.write(script)
text_file.close()

batch = r"""#!/bin/bash

#SBATCH -N """+str(Nnode)+r"""
#SBATCH --ntasks="""+str(ell_i)+r"""
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --job-name="radiation"

source $HOME/myenv/bin/activate

srun -n """+str(ell_i)+""" -l --multi-prog script/silly_il.conf
"""

text_file = open('script/script_integral_il.sh', "w")
text_file.write(batch)
text_file.close()
