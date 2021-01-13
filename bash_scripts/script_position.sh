#!/bin/bash
#input="${HOME}/repos/hybrid_eft_nbody/bash_scripts/sim_names.txt"
input="${HOME}/repos/hybrid_eft_nbody/bash_scripts/sim_names_small.txt"
input_z="${HOME}/repos/hybrid_eft_nbody/bash_scripts/redshifts.txt"
template="${HOME}/repos/hybrid_eft_nbody/bash_scripts/tmp_position.slurm"
#template="${HOME}/repos/hybrid_eft_nbody/bash_scripts/tmp_position_debug.slurm"
tmp="tmp.txt"
counter=0
while IFS= read -r line; do
    counter_z=0
    while read -r zline; do
	output="sbatch_jobs/sbatch_position_${counter}_${counter_z}.slurm"
	rm "${output}"
	echo -n "--sim_name ${line} --z_nbody ${zline}" > "$tmp"
	cat "${template}" "$tmp" >> "${output}"
	let "counter_z++"
    done < "$input_z"
    echo "" >> "${output}"
    let "counter++"
done < "$input"
rm "$tmp"
