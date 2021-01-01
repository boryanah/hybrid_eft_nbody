#!/bin/bash
input="${HOME}/repos/hybrid_eft_nbody/bash_scripts/sim_names.txt"
input_z="${HOME}/repos/hybrid_eft_nbody/bash_scripts/redshifts.txt"
exec="../obtain_theory.py --machine NERSC "
output="theory.sh"
counter=0
while IFS= read -r line; do
    counter_z=0
    while read -r zline; do
	echo -n "${exec} --sim_name ${line} --z_nbody ${zline}" >> "$output"
	echo "" >> "${output}"
	let "counter_z++"
    done < "$input_z"
    let "counter++"
done < "$input"