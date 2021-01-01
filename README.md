# Main functions:

obtain_fields.py -- load density and compute smoothed fields (delta, delta_sq, nabla_sq, s)

obtain_positions.py -- save positions weighted by fields for matter and also save halos

obtain_templates.py -- assemble matter files and compute 15 terms using templates at given redshift

obtain_theory.py -- interpolate between the LPT calculation on large scales and the N-body templates on small scales

obtain_power.py -- assemble halo files and compute true halo-halo halo-matter and matter-matter power spectra

solve_power.py -- analytically fit the power spectrum given templates, true power and covariance

project_Cl.py -- project the 3D power spectrum P(k,a) into the angular power spectrum C_ell

# Extra functions:

obtain_covariance_gadget.py -- probably should be getting covariance from elsewhere but computes jackknife errors given halo pos (needs assembling)

fit_power.py -- fit using minimizing the power spectrum given templates, true power and covariance

obtain_halo_positions.py -- save just the halo positions without any of the particle information

plot_templates.py -- plot the 15 templates

plot_density.py -- plot the density fields of the initial conditions

# Simulations used:

The simulations we are using are listed in bash_scripts/sim_names.txt. The cosmologies are c100-105, c112-113, c117-120, c125-126.

# Memory usage:

Note: We are now using the base sims, so all of the numbers below are off.

convert to bigfile and compute fields for Sim1024 took 2663s on the login node (fell asleep! wasn't on purpose)

Summit hugebase obtain fields only 11352 took 1847. seconds

requested 45 GB for summit hugebase obtain pisitions serially on debug about 15 mins per snapshot; took 70 mins total for 9 snapshots; s_sq takes:
25109 MB and 386 seconds
42670.98046875 MB in t =  1115.3992731571198 for 6th iteration for hugebase

obtain templates took 512 GB on 8 nodes 4 tasks per node (64 GB per node) and 26 mins (requested roughly twice the memory loaded)

obtain power took 4 tasks on a single node total 64 GB and < 18 mins (requested roughly twice the memory loaded)

obtain halo positions took 421.4453125 MB and 8.194952011108398 s for hugebase

obtain power for Pk_hh only took 2 mins with 64 GB for hugebase (and threshold)