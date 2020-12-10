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

#SBATCH --tasks-per-node=4
#SBATCH --nodes=8
#SBATCH --mem=64GB

##SBATCH -q regular
#SBATCH -q debug

#SBATCH -t 00:30:00
#SBATCH -C haswell
#SBATCH -J test-templates
#SBATCH --mail-user=boryana.hadzhiyska@cfa.harvard.edu
#SBATCH --mail-type=ALL

conda activate desc

which mpirun

echo $SLURM_NTASKS

mpirun -n $SLURM_NTASKS python obtain_templates.py
