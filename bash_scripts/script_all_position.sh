#!/bin/bash
input="${HOME}/repos/hybrid_eft_nbody/bash_scripts/sim_names.txt"
index_z=0
exec="sbatch "
output="sbatch_jobs/sbatch_all_position_${index_z}.sh"
counter=0
while IFS= read -r line; do
    echo -n "${exec} sbatch_position_${counter}_${index_z}.sbatch" >> "${output}"
    echo "" >> "${output}"
    let "counter++"
done < "$input"