#!/bin/bash -l

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

mpirun -n $SLURM_NTASKS ../../obtain_fields.py --sim_name 