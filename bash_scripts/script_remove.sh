#!/bin/bash
#input="${HOME}/repos/hybrid_eft_nbody/bash_scripts/sim_names.txt"
input="${HOME}/repos/hybrid_eft_nbody/bash_scripts/sim_names_small.txt"
input_z="${HOME}/repos/hybrid_eft_nbody/bash_scripts/redshifts.txt"
loc_nersc="/global/cscratch1/sd/boryanah/data_hybrid/abacus/"
exec="rm"
output="remove.sh"
#zline="0.5"
#zline="0.1"
zline="0.2"
rm "$output"
while IFS= read -r sline; do
    echo -n "${exec}" "${loc_nersc}${sline}/z${zline}00/*.fits"  >> "${output}"
    echo "" >> "$output"
done < "$input"
#while IFS= read -r sline; do
#    while read -r zline; do
#	echo -n "${exec}" "${loc_nersc}${sline}/z${zline}00/*.fits"  >> "${output}"
#	echo "" >> "$output"
#    done < "$input_z"
#done < "$input"
