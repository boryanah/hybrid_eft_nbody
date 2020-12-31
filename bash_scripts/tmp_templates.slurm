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

mpirun -n $SLURM_NTASKS ../obtain_templates.py --z_nbody  --sim_name 
