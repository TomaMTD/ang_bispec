#!/bin/bash
                
#SBATCH -N 1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --job-name=wd3v_lall_q4

source $HOME/myenv/bin/activate

srun -n 128 -l --multi-prog silly_wd3v_lall_q4.conf
