#!/bin/bash
input="${HOME}/repos/hybrid_eft_nbody/bash_scripts/sim_names.txt"
input_z="${HOME}/repos/hybrid_eft_nbody/bash_scripts/redshifts.txt"
loc_tmp="/global/cscratch1/sd/boryanah/data_hybrid/abacus/"
loc_save="${HOME}/repos/hybrid_eft_nbody/data/"

output="copy_tmp.sh"
rm "$output"
while IFS= read -r line; do
    echo -n "mkdir" "${loc_save}/${line}/"  >> "${output}"
    echo "" >> "$output"
    while read -r line_z; do
	echo -n "mkdir" "${loc_save}/${line}/z${line_z}00/"  >> "${output}"
	echo "" >> "$output"
	echo -n "cp" "${loc_tmp}/${line}/z${line_z}00/*.npy" "${loc_save}/${line}/z${line_z}00/"  >> "${output}"
	echo "" >> "$output"
	echo -n "git add -f" "${loc_save}/${line}/z${line_z}00/*.npy"  >> "${output}"
	echo "" >> "$output"
    done < "$input_z"
done < "$input"
