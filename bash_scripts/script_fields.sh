#!/bin/bash
input="${HOME}/repos/hybrid_eft_nbody/bash_scripts/sim_names.txt"
template="${HOME}/repos/hybrid_eft_nbody/bash_scripts/tmp_fields.slurm"
tmp="tmp.txt"
counter=0
while IFS= read -r line; do
    output="sbatch_fields_${counter}.slurm"
    echo -n "${line}" > "$tmp"
    cat "${template}" "$tmp" >> "${output}"
    echo "" >> "${output}"
    let "counter++"
done < "$input"
