main functions:
fields -- load density and compute smoothed fields
positions -- save positions weighted by fields for matter; save halos
templates -- assemble matter files and compute 15 terms using templates at given redshift
power -- assemble halo files and compute true halo-halo halo-matter and matter-matter
covariance -- probably should be getting covariance from elsewhere but computes jackknife errors given halo pos (needs assembling)
solve -- analytically fit the power spectrum given templates, true power and covariance
fit -- fit using minimizing the power spectrum given templates, true power and covariance

also available:
plotting templates and plotting density

memory usage:
convert to bigfile and compute fields for Sim1024 took 2663s on the login node (fell asleep! wasn't on purpose) 
requested 45 GB for summit hugebase obtain pisitions serially on debug about 15 mins per snapshot; took 70 mins total for 9 snapshots
obtain templates took 512 GB on 8 nodes 4 tasks per node (64 GB per node) and 26 mins (requested roughly twice the memory loaded)
obtain power took 4 tasks on a single node total 64 GB and < 18 mins (requested roughly twice the memory loaded)
can't estimate memory usage using mpirun not too sure why

squeue --user=boryanah
top -u