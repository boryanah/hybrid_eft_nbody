import numpy as np
import matplotlib.pyplot as plt
from choose_parameters import load_dict

# redshift choice
#z_nbody = 1.1
z_nbody = 1.

machine = 'alan'
#machine = 'NERSC'

#sim_name = "AbacusSummit_hugebase_c000_ph000"
sim_name = "Sim256"

user_dict, cosmo_dict = load_dict(z_nbody,sim_name,machine)

R_smooth = user_dict['R_smooth']
data_dir = user_dict['data_dir']

emergency_dir = "/global/cscratch1/sd/boryanah/data_hybrid/gadget/%s/z%.3f/"%('Sim256',z_nbody)

Pk_hh = np.load(data_dir+"Pk_hh.npy")
ks = np.load(data_dir+"ks.npy")
#Pk_err = np.load(data_dir+"Pk_hh_err.npy")
#Pk_err = np.load(emergency_dir+"Pk_hh_err.npy")

ks_all = np.load(data_dir+"ks_all.npy")
Pk_all = np.load(data_dir+"Pk_all_%d.npy"%(int(R_smooth)))
k_lengths = np.load(data_dir+"k_lengths.npy").astype(int)
k_starts = np.zeros(len(k_lengths),dtype=int)
k_starts[1:] = np.cumsum(k_lengths)[:-1]

print(k_lengths)

fields = ['1','\delta','\delta^2','\\nabla^2 \delta','s^2']

labels = []
for i in range(len(fields)):
    for j in range(len(fields)):
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

    if 'nabla' in label:
        True
        #print(label,Pk)
        #Pk = np.abs(Pk)
        
    if i % nperplot == 0:
        #plt.errorbar(ks,Pk_hh,yerr=Pk_err,color='black',label='hh',zorder=1)
        #plt.plot(ks,Pk_hh,color='black',label='hh',zorder=1)
        print("skipping hh")
    plt.plot(ks,Pk,label=label)

    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$P(k)$")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1.e-2,2.])
    plt.legend()

plt.savefig("figs/Pk_template.png")
plt.show()
