#!/bin/bash
input_z="${HOME}/repos/hybrid_eft_nbody/bash_scripts/redshifts.txt"
input_p="${HOME}/repos/hybrid_eft_nbody/bash_scripts/parameters.txt"
exec="../obtain_derivatives.py --machine alan "
output="derivatives.sh"

rm "$output"
while read -r zline; do
    while read -r pline; do
	echo -n "${exec} --z_nbody ${zline} --pars_vary ${pline}" >> "$output"
	echo "" >> "${output}"
    done < "$input_p"
done < "$input_z"

