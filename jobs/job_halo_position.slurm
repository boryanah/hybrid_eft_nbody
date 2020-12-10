#!/bin/bash -l

##SBATCH --array=0
##SBATCH --partition=normal
##SBATCH --account=desi
##SBATCH --ntasks=32
##SBATCH -q debug
##SBATCH --mem=128GB
##SBATCH -C knl
##SBATCH --account=m1727
##SBATCH --cpus-per-task=1
##SBATCH --mem-per-cpu=32GB

#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=45GB # memory per node

#SBATCH -q regular
##SBATCH -q debug

#SBATCH -t 00:60:00
#SBATCH -C haswell
#SBATCH -J test-positions
#SBATCH --mail-user=boryana.hadzhiyska@cfa.harvard.edu
#SBATCH --mail-type=ALL

conda activate desc

#which mpirun

echo $SLURM_NTASKS

python obtain_halo_position.py
