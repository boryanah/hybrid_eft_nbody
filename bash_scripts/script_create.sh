#!/bin/bash
input="${HOME}/repos/hybrid_eft_nbody/bash_scripts/sim_names.txt"
loc_nersc="/global/cscratch1/sd/boryanah/data_hybrid/abacus/"
exec="mkdir "
output="create.sh"
while IFS= read -r line
do
    echo -n "${exec}" "${loc_nersc}${line}"  >> "${output}"
    echo "" >> "$output"
    done < "$input"
