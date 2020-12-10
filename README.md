# Main functions:

fields -- load density and compute smoothed fields

positions -- save positions weighted by fields for matter; save halos

templates -- assemble matter files and compute 15 terms using templates at given redshift

power -- assemble halo files and compute true halo-halo halo-matter and matter-matter

covariance -- probably should be getting covariance from elsewhere but computes jackknife errors given halo pos (needs assembling)

solve -- analytically fit the power spectrum given templates, true power and covariance

fit -- fit using minimizing the power spectrum given templates, true power and covariance

# Also available functions:

obtain_halo_positions, plot_templates and plot_density

# Memory usage:

convert to bigfile and compute fields for Sim1024 took 2663s on the login node (fell asleep! wasn't on purpose)

Summit hugebase obtain fields only 11352 took 1847. seconds

requested 45 GB for summit hugebase obtain pisitions serially on debug about 15 mins per snapshot; took 70 mins total for 9 snapshots; s_sq takes:
25109 MB and 386 seconds
42670.98046875 MB in t =  1115.3992731571198 for 6th iteration for hugebase

obtain templates took 512 GB on 8 nodes 4 tasks per node (64 GB per node) and 26 mins (requested roughly twice the memory loaded)

obtain power took 4 tasks on a single node total 64 GB and < 18 mins (requested roughly twice the memory loaded)

obtain halo positions took 421.4453125 MB and 8.194952011108398 s for hugebase

obtain power for Pk_hh only took 2 mins with 64 GB for hugebase (and threshold)

Note: can't estimate memory usage using mpirun not too sure why

squeue --user=boryanah
top -u