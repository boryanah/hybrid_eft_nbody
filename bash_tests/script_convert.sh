#!/bin/bash
input="${HOME}/repos/hybrid_eft_nbody/bash_tests/sim_names.txt"
exec="/home/boryanah/anaconda3/envs/p3/bin/python obtain_fields.py --machine alan --convert_to_bigfile --sim_name "
output="convert.sh"
while IFS= read -r line
do
    echo -n "$exec" "$line" >> "$output"
    echo "" >> "$output"
    done < "$input"
