'''
Usage:
------

python3 check_fit.py bf_bs_cosmo_dz_nuis_0_5 bf_b0_cosmo_dz_nuis_lowk_0_5 0
python3 check_fit.py bf_bs_cosmo_dz_nuis_0_5 bf_b0_cosmo_dz_nuis_lowk_0_5 1


'''
import os
import sys

from getdist import plots, MCSamples
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import getdist
import numpy as np
import sacc

# matplotlib settings
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times'],'size':14})
rc('text', usetex=True)

# color settings
color_d = 'dodgerblue'
color_b = '#CC6677'
lss = ['-', '--']

# load data files
sG = sacc.Sacc.load_fits('data/cls_covG_covNG_DESgc_DESwl.fits')
#nG = sacc.Sacc.load_fits('data/cls_nls_covG_covNG_DESgc_DESwl.fits')
nG = sacc.Sacc.load_fits('/mnt/extraspace/gravityls_3/S8z/Cls_2/4096_asDavid_recompute_newfid/nls_covNG.fits')

# name of the chain(s) we want to show
chain_names = sys.argv[1:-1]
#chain_name = chain_name[3:]
want_ratio = bool(int(sys.argv[-1]))
print("chain names = ", chain_names, len(chain_names))


# locating chains and files
dis = []
bfs = []
for i in range(len(chain_names)):
    chain_dir = os.path.expanduser('~/repos/montepython_public/chains/'+chain_names[i]+'/')
    gcgc_gcwl_wlwl_di = np.load(chain_dir+'cl_cross_corr_data_info.npz')
    gcgc_gcwl_wlwl_bfi = np.load(chain_dir+'cl_cross_corr_bestfit_info.npz')
    dis.append(gcgc_gcwl_wlwl_di)
    bfs.append(gcgc_gcwl_wlwl_bfi)

if want_ratio: rat_lab = "chi2_"
else: rat_lab = ""

def split_vector_array(ells, chi2):
    ells_ar = []
    chi2_ar = []

    chi2_ar_tmp = []
    ells_ar_tmp = []
    for i, l in enumerate(ells):
        if (i > 0) and (l < ells_ar_tmp[-1]):
            ells_ar.append(np.array(ells_ar_tmp))
            chi2_ar.append(np.array(chi2_ar_tmp))
            
            ells_ar_tmp = []
            chi2_ar_tmp = []

        ells_ar_tmp.append(l)
        chi2_ar_tmp.append(chi2[i])
    ells_ar.append(np.array(ells_ar_tmp))
    chi2_ar.append(np.array(chi2_ar_tmp))
    
    return ells_ar, chi2_ar


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

# GCGC
f, axs = plt.subplots(1, 5, figsize=(18, 5.), gridspec_kw={'wspace': 0}, sharey=True)
ax = axs.reshape((-1))

ix = 0
for i, trs in enumerate(dis[0]['tracers']):
    tr1, tr2 = trs
    
    if ('gc' not in tr1) or ('gc' not in tr2):
        continue
    print(tr1, tr2)
    if ix == len(ax): break
    ell, cl, cov = sG.get_ell_cl('cl_00', tr1, tr2, return_cov=True)
    ell, nl = nG.get_ell_cl('cl_00', tr1, tr2, return_cov=False)
    err = np.sqrt(np.diag(cov))
        
    if want_ratio:
        for j in range(len(chain_names)):
            info = dis[j]
            bf = bfs[j]
            ells_bf_ar, cls_bf_ar = split_vector_array(info['ells'], bf['cls'])
            
            arg = np.argmin(np.abs(ells_bf_ar[i][0]-ell))
            #rat = cl[arg:arg+len(ells_bf_ar[i])]/cls_bf_ar[i]
            rat = (cl[arg:arg+len(ells_bf_ar[i])] - cls_bf_ar[i]) / err[arg:arg+len(ells_bf_ar[i])]


            if j == len(chain_names)-1:
                label = 'Bestfit, DY1'
            else:
                label = 'Bestfit, HEFT'

            ax[ix].plot(ell[arg:arg+len(ells_bf_ar[i])], rat, color=adjust_lightness(color_b, 0.3+j*2./5.), ls=lss[j], lw=2+j/2., label=label)

            if j == 0:
                ax[ix].plot(ell[arg:arg+len(ells_bf_ar[i])], np.zeros(len(ells_bf_ar[i])), color='silver')
                ax[ix].fill_between(ell[arg:arg+len(ells_bf_ar[i])], np.zeros(len(ells_bf_ar[i]))-1., np.zeros(len(ells_bf_ar[i]))+1., color='gray', alpha=0.1)
        
    else:
        ax[ix].errorbar(ell, cl, yerr=err, color=color_d, ls='', capsize=4, fmt='o', ms=4, label='Data')
        #ax[ix].errorbar(ell, nl, yerr=np.zeros_like(err), color='gray', ls='', capsize=2, fmt='o', ms=2, label='Data + SN')
        ax[ix].plot(ell, nl, color='gray', ls='--', label='Shotnoise')
        for j in range(len(chain_names)):
            info = dis[j]
            bf = bfs[j]
            ells_bf_ar, cls_bf_ar = split_vector_array(info['ells'], bf['cls'])
            if j == len(chain_names)-1:
                label = 'Bestfit, DY1'
            else:
                label = 'Bestfit, HEFT'
            ax[ix].loglog(ells_bf_ar[i], cls_bf_ar[i], color=adjust_lightness(color_b, 0.3+j*2./5.), ls=lss[j], lw=2+j/2., label=label)
            
    if 'DESgc' in tr1:
        label_tr1 = 'g'+tr1.split('DESgc')[-1]
    else:
        label_tr1 = 's'+tr1.split('DESwl')[-1]
    if 'DESgc' in tr2:
        label_tr2 = 'g'+tr2.split('DESgc')[-1]
    else:
        label_tr2 = 's'+tr2.split('DESwl')[-1]
    ax[ix].text(0.9, 0.9, label_tr1+'-'+label_tr2, horizontalalignment='right', 
               verticalalignment='top', transform=ax[ix].transAxes)

    ax[ix].set_xlabel(r'$\ell$', fontsize=18)
    if ix in [0]:
        if want_ratio:
            ax[ix].set_ylabel(r'$\frac{(C_{\ell, {\rm data}}-C_{\ell, {\rm bestfit}})}{\sigma_\ell}$', fontsize=18)
        else:
            ax[ix].set_ylabel(r'$C_\ell$', fontsize=18)

    # TESTING
    if want_ratio:
        ax[ix].set_xlim([0.95*ells_bf_ar[i][0], 1.05*ells_bf_ar[i][-1]])
    else:
        ax[ix].set_xlim([0., 2000.])
    ax[ix].axvline(x=0.3*ells_bf_ar[i][-1], ls=':', c='#CC6677')
    
    if ix == 0:
        if want_ratio:
            ax[i].legend(loc='upper left', frameon=False)
        else:
            ax[i].legend(loc='lower left', frameon=False)

    print(ix)
    ix += 1

if len(chain_names) > 1:
    plt.savefig("figs/"+rat_lab+"gcgc.pdf")
else:
    plt.savefig("figs/"+rat_lab+"gcgc_"+chain_name+".pdf")
plt.close()

# GCWL
f, axs = plt.subplots(5, 4, figsize=(15, 15))
ax = axs.reshape((-1))
ix = 0
for i, trs in enumerate(dis[0]['tracers']):
    tr1, tr2 = trs
    if ('gc' not in tr1) or ('wl' not in tr2):
        continue    
    print(tr1, tr2)
    ell, cl, cov = sG.get_ell_cl('cl_0e', tr1, tr2, return_cov=True)
    err = np.sqrt(np.diag(cov))
    
    if want_ratio:
        for j in range(len(chain_names)):
            info = dis[j]
            bf = bfs[j]
            ells_bf_ar, cls_bf_ar = split_vector_array(info['ells'], bf['cls'])
            
            arg = np.argmin(np.abs(ells_bf_ar[i][0]-ell))
            #rat = cl[arg:arg+len(ells_bf_ar[i])]/cls_bf_ar[i]
            rat = (cl[arg:arg+len(ells_bf_ar[i])] - cls_bf_ar[i]) / err[arg:arg+len(ells_bf_ar[i])]
            ax[ix].plot(ell[arg:arg+len(ells_bf_ar[i])], rat, color=adjust_lightness(color_b, 0.3+j*2./5.), ls=lss[j], lw=2+j/2.)

            if j == 0:
                ax[ix].plot(ell[arg:arg+len(ells_bf_ar[i])], np.zeros(len(ells_bf_ar[i])), color='silver')
                ax[ix].fill_between(ell[arg:arg+len(ells_bf_ar[i])], np.zeros(len(ells_bf_ar[i]))-1., np.zeros(len(ells_bf_ar[i]))+1., color='gray', alpha=0.1)

    else:
        ax[ix].errorbar(ell, cl, yerr=err, color=color_d, ls='', capsize=4, fmt='o', ms=4, label='Data')
        for j in range(len(chain_names)):
            info = dis[j]
            bf = bfs[j]
            ells_bf_ar, cls_bf_ar = split_vector_array(info['ells'], bf['cls'])
            if j == len(chain_names)-1:
                label = 'Bestfit'
            else:
                label = ''
            ax[ix].loglog(ells_bf_ar[i], cls_bf_ar[i], color=adjust_lightness(color_b, 0.3+j*2./5.), ls=lss[j], lw=2+j/2., label=label)

    if 'DESgc' in tr1:
        label_tr1 = 'g'+tr1.split('DESgc')[-1]
    else:
        label_tr1 = 's'+tr1.split('DESwl')[-1]
    if 'DESgc' in tr2:
        label_tr2 = 'g'+tr2.split('DESgc')[-1]
    else:
        label_tr2 = 's'+tr2.split('DESwl')[-1]
    ax[ix].text(0.9, 0.9, label_tr1+'-'+label_tr2, horizontalalignment='right', 
                verticalalignment='top', transform=ax[ix].transAxes)

    if ix > 15:
        ax[ix].set_xlabel(r'$\ell$', fontsize=18)
    if ix in [0, 4, 8, 12, 16]:
        if want_ratio:
            ax[ix].set_ylabel(r'$\frac{(C_{\ell, {\rm data}}-C_{\ell, {\rm bestfit}})}{\sigma_\ell}$', fontsize=18)
        else:
            ax[ix].set_ylabel(r'$C_\ell$', fontsize=18)

    # TESTING
    if want_ratio:
        ax[ix].set_xlim([0.95*ells_bf_ar[i][0], 1.05*ells_bf_ar[i][-1]])
    else:
        ax[ix].set_xlim([0., 2000.])
    
    #if ix == 0 and want_ratio == False:
    #    ax[ix].legend(loc='lower left', frameon=False)
    ix += 1

if len(chain_names) > 1:
    plt.savefig("figs/"+rat_lab+"gcwl.pdf")
else:
    plt.savefig("figs/"+rat_lab+"gcwl_"+chain_name+".pdf")
#plt.show()
plt.close()

print("Skipping wlwl")
quit()
# WLWL
info = gcgc_gcwl_wlwl_di
bf = gcgc_gcwl_wlwl_bfi

ells_bf_ar, cls_bf_ar = split_vector_array(info['ells'], bf['cls'])
f, axs = plt.subplots(4, 4, figsize=(12, 12))
ax = axs#.reshape((-1))
ix = 0
for i, trs in enumerate(info['tracers']):
    tr1, tr2 = trs
    if ('wl' not in tr1) or ('wl' not in tr2):
        continue
    ix = (int(tr1[-1]), int(tr2[-1]))
    ell, cl, cov = sG.get_ell_cl('cl_ee', tr1, tr2, return_cov=True)
    err = np.sqrt(np.diag(cov))
    print(tr1, tr2)
    arg = np.argmin(np.abs(ells_bf_ar[i][0]-ell))
    print("Fractional difference = ", (cl[arg:arg+len(ells_bf_ar[i])] - cls_bf_ar[i]) / err[arg:arg+len(ells_bf_ar[i])])
    print(arg, ix)
    if want_ratio:
        #rat = cl[arg:arg+len(ells_bf_ar[i])]/cls_bf_ar[i]
        rat = (cl[arg:arg+len(ells_bf_ar[i])] - cls_bf_ar[i]) / err[arg:arg+len(ells_bf_ar[i])]
        ax[ix].plot(ell[arg:arg+len(ells_bf_ar[i])], np.zeros(len(ells_bf_ar[i])), color='silver')
        ax[ix].fill_between(ell[arg:arg+len(ells_bf_ar[i])], np.zeros(len(ells_bf_ar[i]))-1., np.zeros(len(ells_bf_ar[i]))+1., color='gray', alpha=0.1)
        ax[ix].plot(ell[arg:arg+len(ells_bf_ar[i])], rat, color=color_b, ls='-')
    else:
        ax[ix].errorbar(ell, cl, yerr=err, color=color_d, ls='', capsize=2, fmt='o', ms=2)
        ax[ix].loglog(ells_bf_ar[i], cls_bf_ar[i], color=color_b, alpha=0.7+j/10., lw=3)

    if 'DESgc' in tr1:
        label_tr1 = 'g'+tr1.split('DESgc')[-1]
    else:
        label_tr1 = 's'+tr1.split('DESwl')[-1]
    if 'DESgc' in tr2:
        label_tr2 = 'g'+tr2.split('DESgc')[-1]
    else:
        label_tr2 = 's'+tr2.split('DESwl')[-1]
    ax[ix].text(0.9, 0.9, label_tr1+'-'+label_tr2, horizontalalignment='right', 
               verticalalignment='top', transform=ax[ix].transAxes)

    if ix > 16:
        ax[ix].set_xlabel(r'$\ell$')
    if ix in [0, 4, 8, 12, 16]:
        if want_ratio:
            ax[ix].set_ylabel(r'$\frac{(C_{\ell, {\rm data}}-C_{\ell, {\rm bestfit}})}{\sigma_\ell}$', fontsize=18)
        else:
            ax[ix].set_ylabel(r'$C_\ell$')

    # TESTING
    ax[ix].set_xlim([0.95*ells_bf_ar[i][0], 1.05*ells_bf_ar[i][-1]])
    
    if ix == 0 and want_ratio == False:
        plt.legend()
    ix += 1

plt.savefig("figs/"+rat_lab+"wlwl_"+chain_name+".png")
#plt.show()
plt.close()
