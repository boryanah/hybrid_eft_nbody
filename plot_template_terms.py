import numpy as np
import matplotlib.pyplot as plt

# user choices
R_smooth = 2.
z_nbody = 1.
simulation_code = 'gadget'
machine = 'alan'

if simulation_code == 'abacus':
    sim_name = "AbacusSummit_hugebase_c000_ph000"#small/AbacusSummit_small_c000_ph3046
    if machine == 'alan':
        data_dir = "/mnt/store1/boryanah/data_hybrid/abacus/"+sim_name+"/z%.3f/"%z_nbody
    elif machine == 'NERSC':
        data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/abacus/"+sim_name+"/z%.3f/"%z_nbody
elif simulation_code == 'gadget':
    sim_name = "Sim256"    
    if machine == 'alan':
        data_dir = "/mnt/gosling1/boryanah/"+sim_name+"/z%.3f/"%z_nbody
    elif machine == 'NERSC':
        data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/gadget/"+sim_name+"/z%.3f/"%z_nbody


Pk_hh = np.load(data_dir+"Pk_hh.npy")
ks = np.load(data_dir+"ks.npy")
Pk_err = np.load(data_dir+"Pk_hh_err.npy")

ks_all = np.load(data_dir+"ks_all.npy")
Pk_all = np.load(data_dir+"Pk_all_%d.npy"%(int(R_smooth)))
k_lengths = np.load(data_dir+"k_lengths.npy").astype(int)
k_starts = np.zeros(len(k_lengths),dtype=int)
k_starts[1:] = np.cumsum(k_lengths)[:-1]


fields = ['1','\delta','\delta^2','\\nabla^2 \delta','s^2']
labels = []
for i in range(5):
    for j in range(5):
        if j < i: continue
        label = r'$\langle '+fields[i]+","+fields[j]+r" \rangle$"
        labels.append(label)
        
        
nrow = 1
ncol = 3
ncurve = len(k_lengths)
nperplot = ncurve/ncol

plt.subplots(nrow,ncol,figsize=(16,5))

for i in range(ncurve):
    i_plot = int(i//nperplot)
    plt.subplot(1,3,i_plot+1)
    label = labels[i]
    
    start = k_starts[i]
    size = k_lengths[i]

    Pk = Pk_all[start:start+size]
    ks = ks_all[start:start+size]

    if i % nperplot == 0:
        plt.errorbar(ks,Pk_hh,yerr=Pk_err,color='black',label='hh',zorder=1)
        #plt.plot(ks,Pk_hh,color='black',label='hh',zorder=1)
    plt.plot(ks,Pk,label=label)

    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$P(k)$")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1.e-2,2.])
    plt.legend()

plt.savefig("figs/Pk_template.png")
plt.show()
