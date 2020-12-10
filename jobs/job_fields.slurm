#!/bin/bash -l

## THIS WAS MY ATTEMPT TO DO CONVERSION BUT IT"S KINDA TOO MUCH (yu thinks we need 520 GB, which is a bit less than alotted)
## evil me ran this (convert to bigfile and compute fields for Sim1024) on the login node -- it took 2663s

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

#SBATCH --tasks-per-node=4
#SBATCH --nodes=16
#SBATCH --mem=64GB # 16GB

##SBATCH -q regular
#SBATCH -q debug

#SBATCH -t 00:30:00
#SBATCH -C haswell
#SBATCH -J test-fields
#SBATCH --mail-user=boryana.hadzhiyska@cfa.harvard.edu
#SBATCH --mail-type=ALL

conda activate desc

which mpirun

echo $SLURM_NTASKS

mpirun -n $SLURM_NTASKS python obtain_fields.py
