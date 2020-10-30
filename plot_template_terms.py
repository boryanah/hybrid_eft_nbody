import numpy as np
import matplotlib.pyplot as plt


R_smooth = 2.

data_dir = "data/"

Pk_true = np.load(data_dir+"Pk_true_mean.npy")
ks = np.load(data_dir+"ks.npy")
Pk_err = np.load(data_dir+"Pk_true_err.npy")

ks_all = np.load(data_dir+"ks_all.npy")
Pk_all = np.load(data_dir+"Pk_all_real_%d.npy"%(int(R_smooth)))
k_lengths = np.load(data_dir+"k_lengths.npy").astype(int)
k_starts = np.zeros(len(k_lengths),dtype=int)
k_starts[1:] = np.cumsum(k_lengths)[:-1]


fields = ['1','\delta','\delta^2','\\nabla^2 \delta','s^2']
labels = []
for i in range(5):
    for j in range(5):
        if j < i: continue
        
        labels.append(r'$\langle '+fields[i]+","+fields[j]+r" \rangle$")

nrow = 1
ncol = 3
ncurve = len(k_lengths)
nperplot = ncurve/ncol

plt.subplots(nrow,ncol,figsize=(16,5))

for i in range(ncurve):
    i_plot = int(i//nperplot)
    plt.subplot(1,3,i_plot+1)

    start = k_starts[i]
    size = k_lengths[i]

    Pk = Pk_all[start:start+size]
    ks = ks_all[start:start+size]
    
    if i % nperplot == 0:
        plt.errorbar(ks,Pk_true,yerr=Pk_err,color='black',label='halo-halo',zorder=1)
    plt.plot(ks,Pk,label=labels[i])

    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$P(k)$")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()

plt.savefig("figs/Pk_template.png")
plt.show()
