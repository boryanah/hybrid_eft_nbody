#!/bin/bash -l

#SBATCH --tasks-per-node=8   # base 8
#SBATCH --nodes=36  #45 # base 32, hugebase 8, hugebase Modi 12
#SBATCH --mem=64GB #32  # og 64GB

##SBATCH -q regular
#SBATCH -q debug

#SBATCH -t 00:30:00
##SBATCH -t 00:45:00
#SBATCH -C haswell
#SBATCH -J test-templates
#SBATCH --mail-user=boryana.hadzhiyska@cfa.harvard.edu
#SBATCH --mail-type=ALL

conda activate desc

which mpirun

echo $SLURM_NTASKS

mpirun -n $SLURM_NTASKS ../../obtain_templates.py --machine NERSC 