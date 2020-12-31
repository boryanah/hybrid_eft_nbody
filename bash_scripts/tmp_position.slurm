#!/bin/bash -l

#SBATCH --tasks-per-node=1
#SBATCH --nodes=34
#SBATCH --mem=64GB # memory per node

##SBATCH -q regular
#SBATCH -q debug

#SBATCH -t 00:30:00 # do at least 80 mins
#SBATCH -C haswell
#SBATCH -J test-positions
#SBATCH --mail-user=boryana.hadzhiyska@cfa.harvard.edu
#SBATCH --mail-type=ALL

conda activate desc

#which mpirun

echo $SLURM_NTASKS

mpirun -n $SLURM_NTASKS ../../obtain_position.py --machine NERSC 