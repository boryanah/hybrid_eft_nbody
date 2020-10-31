from tools.read_gadget import read_gadget
from nbodykit.lab import *
from tools.power_spectrum import get_Pk_true
import matplotlib.pyplot as plt
import glob
import numpy as np

# parameters
N_dim = 256
Lbox = 175.
interlaced = True
ind_snap = 0
N_jack = 3

machine = 'NERSC'
# directory of simulation
if machine == 'NERSC':
    directory = "/global/cscratch1/sd/damonge/NbodySims/Sim256/"
    data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/gadget/"
elif machine == 'alan':
    directory = "/mnt/gosling1/boryanah/small_box_damonge/"
    data_dir = "/mnt/gosling1/boryanah/small_box_damonge/output/"

# find all files
ic_fns = sorted(glob.glob(directory+"ic_*"))
snap_fns = sorted(glob.glob(directory+"snap_*"))
fof_fns = sorted(glob.glob(directory+"fof_*.fits"))

print(ic_fns,snap_fns,fof_fns)

# return position of the particles and halos
lagr_pos, pos_snap, pos_halo = read_gadget(ic_fns,snap_fns,fof_fns,ind_snap)
ks, Pk_true = get_Pk_true(pos_halo,N_dim,Lbox,interlaced)

Pk = np.zeros((len(ks),N_jack**3))

for i_x in range(N_jack):
    for i_y in range(N_jack):
        for i_z in range(N_jack):
            print(i_x,i_y,i_z)
            
            pos_halo_jack = pos_halo.copy()
            xyz = np.array([i_x,i_y,i_z],dtype=int)
            size = Lbox/N_jack

            bool_arr = np.prod((xyz == (pos_halo_jack/size).astype(int)),axis=1).astype(bool)
            pos_halo_jack[bool_arr] = np.array([0.,0.,0.])
            pos_halo_jack = pos_halo_jack[np.sum(pos_halo_jack,axis=1)!=0.]

            ks, Pk_jack = get_Pk_true(pos_halo_jack,N_dim,Lbox,interlaced)
            Pk_jack = Pk_jack.astype(float)
            
            Pk[:,i_x+N_jack*i_y+N_jack**2*i_z] = Pk_jack

Pk_mean = np.mean(Pk,axis=1)
Pk_err = np.sqrt(N_jack**3-1)*np.std(Pk,axis=1)

np.save(data_dir+"ks.npy",ks)
np.save(data_dir+"Pk_true.npy",Pk_true)
np.save(data_dir+"Pk_true_mean.npy",Pk_mean)
np.save(data_dir+"Pk_true_err.npy",Pk_err)

plt.plot(ks,Pk_true,'k')
plt.errorbar(ks,Pk_mean,yerr=Pk_err,color='dodgerblue')
plt.xscale('log')
plt.yscale('log')
plt.savefig("figs/Pk_errorbar.png")
plt.show()
