#!/bin/bash -l
#SBATCH -q shared
#SBATCH -t 00:05:00
##SBATCH --mem=4GB
#SBATCH -n 4
##SBATCH -n 1
#SBATCH -C haswell
#SBATCH -J test-nbodykit-parallel
#SBATCH --mail-user=boryana.hadzhiyska@cfa.harvard.edu
#SBATCH --mail-type=ALL

conda activate desc

which mpirun

# #srun -n 4 python example.py
#python example.py
mpirun -np 4 python example.py
