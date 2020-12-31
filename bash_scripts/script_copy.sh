#!/bin/bash
input="${HOME}/repos/hybrid_eft_nbody/bash_scripts/sim_names.txt"
loc_alan="/mnt/alan1/boryanah/"
loc_nersc="boryanah@cori.nersc.gov:/global/cscratch1/sd/boryanah/data_hybrid/abacus/"
name_file="/density_2304.bigfile"
exec="scp -r "
output="copy.sh"
while IFS= read -r line
do
    echo -n "${exec}" "${loc_alan}${line}_ICs${name_file} " "${loc_nersc}${line}/."  >> "${output}"
    echo "" >> "$output"
    done < "$input"
