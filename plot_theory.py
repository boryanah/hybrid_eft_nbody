import numpy as np

import time
import sys

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from LPT.cleft_fftw import CLEFT

#colors = ['#0077BB','#33BBEE','#0099BB','#EE7733','#CC3311','#EE3377','#BBBBBB']
# colour table in HTML hex format
hexcols = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', 
           '#CC6677', '#882255', '#AA4499', '#661100', '#6699CC', '#AA4466',
           '#4477AA']

# user choices
#sim_name = "AbacusSummit_hugebase_c000_ph000"
#sim_name = "AbacusSummit_base_c000_ph000"
sim_name = "Sim256"
#z_nbody = 1.1
z_nbody = 1.
R_smooth = 1.
#R_smooth = 0.
#R_smooth = 2.
#data_dir =  "/home/boryanah/repos/hybrid_eft_nbody/data/%s/z%4.3f/r_smooth_%d/"%(sim_name,z_nbody,int(R_smooth))
#data_dir =  "/home/boryanah/repos/hybrid_eft_nbody/data/%s/z%4.3f/r_smooth_%d_tmp/"%(sim_name,z_nbody,int(R_smooth))
data_dir =  "/mnt/gosling1/boryanah/%s/z%4.3f/"%(sim_name,z_nbody)

# power spectrum file
pk_fn = "/home/boryanah/repos/AbacusSummit/Cosmologies/abacus_cosm000/abacus_cosm000.z2_pk.dat"
#pk_fn = "/home/boryanah/repos/velocileptors/pk.dat"

# templates from nbody
ks_all = np.load(data_dir+"ks_all.npy")
Pk_all = np.load(data_dir+"Pk_all_%d.npy"%(int(R_smooth)))
#Pk_all[np.isnan(Pk_all)] = 0.1

k_lengths = np.load(data_dir+"k_lengths.npy").astype(int)
k_starts = np.zeros(len(k_lengths),dtype=int)
k_starts[1:] = np.cumsum(k_lengths)[:-1]
#fields = ['1','\delta','\delta^2','\\nabla^2 \delta','s^2']
fields = ['1','b_1','b_2','b_{\\nabla^2}','b_s']

        
# To match the plots in Chen, Vlah & White (2020) let's
# work at z=0.8, and scale our initial power spectrum
# to that redshift:
#z,D,f      = 0.8,0.6819,0.8076
z,D,f      = z_nbody, 1., 1.
klin,plin  = np.loadtxt(pk_fn, unpack=True)
plin      *= D**2

# Initialize the class -- with no wisdom file passed it will
# experiment to find the fastest FFT algorithm for the system.
start= time.time()
cleft = CLEFT(klin,plin)
print("Elapsed time: ",time.time()-start," seconds.")
# You could save the wisdom file here if you wanted:
# mome.export_wisdom(wisdom_file_name)

# The first four are deterministic Lagrangian bias up to third order
# While alpha and sn are the counterterm and stochastic term (shot noise)

cleft.make_ptable()

kv = cleft.pktable[:,0]

print(cleft.pktable.shape)

spectra = {\
           r'$(1,1)$':cleft.pktable[:,1],\
           r'$(1,b_1)$':0.5*cleft.pktable[:,2], r'$(b_1,b_1)$': cleft.pktable[:,3],\
           r'$(1,b_2)$':0.5*cleft.pktable[:,4], r'$(b_1,b_2)$': 0.5*cleft.pktable[:,5],  r'$(b_2,b_2)$': cleft.pktable[:,6],\
           r'$(1,b_s)$':0.5*cleft.pktable[:,7], r'$(b_1,b_s)$': 0.5*cleft.pktable[:,8],  r'$(b_2,b_s)$':0.5*cleft.pktable[:,9], r'$(b_s,b_s)$':cleft.pktable[:,10], r'$(1,b_{\nabla^2})$':0.5*cleft.pktable[:,13]*kv**2, r'$(b_1,b_{\nabla^2})$':0.5*cleft.pktable[:,13]*kv**2}#, r'$(b_{\nabla^2},b_{\nabla^2})$':0.5*cleft.pktable[:,13]*kv**2}
#print(0.5*cleft.pktable[:,7]) # 1, bs
#print(0.5*cleft.pktable[:,8]) # b1,bs

nbody_spectra = {}
counter = 0
for i in range(len(fields)):
    for j in range(len(fields)):
        if j < i: continue
        #label = r'$\langle '+fields[i]+","+fields[j]+r" \rangle$"
        #labels.append(label)
        label = r'$('+fields[i]+","+fields[j]+r")$"
        start = k_starts[counter]
        size = k_lengths[counter]

        Pk = Pk_all[start:start+size]
        ks = ks_all[start:start+size]
        Pk = Pk[~np.isnan(ks)]
        ks = ks[~np.isnan(ks)]
        
        
        #Pk_fun = interp1d(ks,Pk,kind='cubic')
        #ks = np.logspace(-3,np.log10(3),100)
        #Pk = Pk_fun(ks)

        nbody_spectra[label] = Pk
        
        counter += 1

# Plot some of them!
#plt.figure(2,figsize=(15,10))

plot_dic ={\
           r'$(1,1)$':1,
           r'$(1,b_1)$':1, r'$(b_1,b_1)$': 2,\
           r'$(1,b_2)$':4, r'$(b_1,b_2)$': 2,  r'$(b_2,b_2)$': 3,\
           r'$(1,b_s)$':4, r'$(b_1,b_s)$': 5, r'$(b_1,b_{\nabla^2})$': 5,  r'$(b_2,b_s)$': 3, r'$(b_s,b_s)$':6, r'$(1,b_{\nabla^2})$':4}


spec_names = spectra.keys()
kpivot_dic = {}
plt.subplots(2,3,figsize=(18,10))
for i,spec_name in enumerate(spec_names):
    if i < 8:
        kpivot_dic[spec_name] = 4.e-1
    else:
        kpivot_dic[spec_name] = 4.e-1
    print(spec_name)
    
    kpivot = kpivot_dic[spec_name]
    iv_pivot = np.argmin(np.abs(kv-kpivot))
    is_pivot = np.argmin(np.abs(ks-kpivot))
    print(ks[is_pivot],kv[iv_pivot])
    Pk_tmp = nbody_spectra[spec_name]
    Pk_tmp *= spectra[spec_name][iv_pivot]/nbody_spectra[spec_name][is_pivot]

    
    plot_no = plot_dic[spec_name]
    plt.subplot(2,3,plot_no)
    plt.loglog(kv, spectra[spec_name], color=hexcols[i], label=spec_name)
    plt.loglog(ks, Pk_tmp, ls='--', color=hexcols[i])

    plt.legend(ncol=1)
    
#plt.ylim(1,3e4)

plt.xlabel('k [h/Mpc]')
plt.ylabel(r'$P_{ab}$ [(Mpc/h)$^3$]')
plt.savefig("figs/templates_LPT.png")
plt.show()
quit()

kv, pk = cleft.combine_bias_terms_pk(*pars)

pars = [0.70, -1.3, -0.06, 0, 7.4, 1.9e3]

plt.figure(1)
plt.plot(kv, kv * pk)

plt.xlim(0,0.25)
plt.ylim(850,1120)

plt.ylabel(r'k $P_{hh}(k)$ [h$^{-2}$ Mpc$^2$]')
plt.xlabel('k [h/Mpc]')
#plt.show()

