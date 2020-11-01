import matplotlib.pyplot as plt
import glob
import numpy as np

from tools.power_spectrum import get_Pk

# parameters
sim_name = 'Sim256'
N_dim = 256
Lbox = 175.
interlaced = True
ind_snap = 0; z_nbody = 1.
N_jack = 3

#machine = 'NERSC'
machine = 'alan'

# directory of data
if machine == 'alan':
    data_dir = "/mnt/gosling1/boryanah/"+sim_name+"/z%.3f/"%z_nbody    
elif machine == 'NERSC':
    data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/gadget/"+sim_name+"/z%.3f/"%z_nbody

# load position of the particles and halos
pos_halo = np.load(data_dir+"pos_halo.npy")
ks = np.load(data_dir+"ks.npy")
Pk_hh = np.load(data_dir+"Pk_hh.npy")

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

            ks, Pk_jack = get_Pk(pos_halo_jack,N_dim,Lbox,interlaced)
            Pk_jack = Pk_jack.astype(float)
            
            Pk[:,i_x+N_jack*i_y+N_jack**2*i_z] = Pk_jack

Pk_mean = np.mean(Pk,axis=1)
Pk_err = np.sqrt(N_jack**3-1)*np.std(Pk,axis=1)

np.save(data_dir+"Pk_hh_mean.npy",Pk_mean)
np.save(data_dir+"Pk_hh_err.npy",Pk_err)

plt.plot(ks,Pk_hh,'k')
plt.errorbar(ks,Pk_mean,yerr=Pk_err,color='dodgerblue')
plt.xscale('log')
plt.yscale('log')
plt.savefig("figs/Pk_errorbar.png")
plt.show()
