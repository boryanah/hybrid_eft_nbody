#!/bin/bash
file_loc="/global/cscratch1/sd/boryanah/data_hybrid/tape_data/"
link_loc="/global/project/projectdirs/desi/cosmosim/Abacus/"
f1="halo_rv_A"
f2="halo_pid_A"
f3="halo_info"
f4="field_rv_A"
f5="field_pid_A"
input="${HOME}/repos/hybrid_eft_nbody/bash_scripts/sim_names_symlink.txt"
input_z="${HOME}/repos/hybrid_eft_nbody/bash_scripts/redshifts_symlink.txt"
#sim_name="AbacusSummit_base_c113_ph000" #c101, c102, c104, c105, c112, c113
exec="ln -s"
output="symlink.sh"
rm "$output"

while IFS= read -r sim_name; do
    echo -n "cd ${file_loc}/${sim_name}/halos" >> "$output"
    echo "" >> "$output"
    while read -r zline; do
	echo -n "mkdir z${zline}00; cd z${zline}00" >> "${output}"
	echo "" >> "$output"
	echo -n "${exec}" "${link_loc}/${sim_name}/halos/z${zline}00/${f1}"  >> "${output}"
	echo "" >> "$output"
	echo -n "${exec}" "${link_loc}/${sim_name}/halos/z${zline}00/${f2}"  >> "${output}"
	echo "" >> "$output"
	echo -n "${exec}" "${link_loc}/${sim_name}/halos/z${zline}00/${f3}"  >> "${output}"
	echo "" >> "$output"
	echo -n "${exec}" "${link_loc}/${sim_name}/halos/z${zline}00/${f4}"  >> "${output}"
	echo "" >> "$output"
	echo -n "${exec}" "${link_loc}/${sim_name}/halos/z${zline}00/${f5}"  >> "${output}"
	echo "" >> "$output"
	echo -n "cd .." >> "$output"
	echo "" >> "$output"
    done < "$input_z"
done < "$input"