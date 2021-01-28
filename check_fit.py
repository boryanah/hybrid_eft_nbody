from getdist import plots, MCSamples
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import getdist
import numpy as np
import os
import sacc

sG = sacc.Sacc.load_fits('data/clfid_covG.fits')

'''
gcgc = getdist.loadMCSamples('./asDavid/cl_cross_corr_v2_gcgc-b_prior/2020-12-09_1000000_',
                           settings={'ignore_rows':0.5})
gcgc_np = getdist.loadMCSamples('./asDavid/cl_cross_corr_v2_gcgc_no-b-prior/2020-12-07_1000000_',
                           settings={'ignore_rows':0.5})


gcgc_di = np.load('./asDavid/cl_cross_corr_v2_gcgc-b_prior/cl_cross_corr_data_info.npz')
gcgc_bfi = np.load('./asDavid/cl_cross_corr_v2_gcgc-b_prior/cl_cross_corr_bestfit_info.npz')
'''
chain_dir = '/home/boryanah/repos/montepython_public/chains/test/'
gcgc_gcwl_wlwl_di = np.load(chain_dir+'cl_cross_corr_data_info.npz')
gcgc_gcwl_wlwl_bfi = np.load(chain_dir+'cl_cross_corr_bestfit_info.npz')
want_ratio = True

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


# GCGC
info = gcgc_gcwl_wlwl_di
bf = gcgc_gcwl_wlwl_bfi

ells_bf_ar, cls_bf_ar = split_vector_array(info['ells'], bf['cls'])
print(len(ells_bf_ar))

f, axs = plt.subplots(1, 5, figsize=(15, 2.5), gridspec_kw={'wspace': 0}, sharey=True)
ax = axs.reshape((-1))

ix = 0
for i, trs in enumerate(info['tracers']):
    tr1, tr2 = trs
    
    if ('gc' not in tr1) or ('gc' not in tr2):
        continue
    print(tr1, tr2)
    if ix == len(ax): break
    ell, cl, cov = sG.get_ell_cl('cl_00', tr1, tr2, return_cov=True)
    err = np.sqrt(np.diag(cov))
    
    
    arg = np.argmin(np.abs(ells_bf_ar[i][0]-ell))
    print("Fractional difference = ",(cl[arg:arg+len(ells_bf_ar[i])] - cls_bf_ar[i]) / err[arg:arg+len(ells_bf_ar[i])])
    print(arg)
    if want_ratio:
        ax[ix].plot(ell[arg:arg+len(ells_bf_ar[i])], cl[arg:arg+len(ells_bf_ar[i])]/cls_bf_ar[i], color='black', ls='--')
    else:
        ax[ix].errorbar(ell, cl, yerr=err, color='black', ls='--')
        ax[ix].loglog(ells_bf_ar[i], cls_bf_ar[i], color='orange', lw=2)
    ax[ix].text(0.9, 0.9, tr1+'-'+tr2, horizontalalignment='right', 
               verticalalignment='top', transform=ax[ix].transAxes)
    ix += 1
    print(ix)

plt.savefig("gcgc_fit.png")
plt.close()

# GCWL
info = gcgc_gcwl_wlwl_di
bf = gcgc_gcwl_wlwl_bfi

ells_bf_ar, cls_bf_ar = split_vector_array(info['ells'], bf['cls'])
f, axs = plt.subplots(5, 4, figsize=(15, 12))
ax = axs.reshape((-1))
ix = 0
for i, trs in enumerate(info['tracers']):
    tr1, tr2 = trs

    if ('gc' not in tr1) or ('wl' not in tr2):
        continue    
    
    ell, cl, cov = sG.get_ell_cl('cl_0e', tr1, tr2, return_cov=True)
    err = np.sqrt(np.diag(cov))
    
    arg = np.argmin(np.abs(ells_bf_ar[i][0]-ell))
    print("Fractional difference = ",(cl[arg:arg+len(ells_bf_ar[i])] - cls_bf_ar[i]) / err[arg:arg+len(ells_bf_ar[i])])
    print(arg)
    if want_ratio:
        ax[ix].plot(ell[arg:arg+len(ells_bf_ar[i])], cl[arg:arg+len(ells_bf_ar[i])]/cls_bf_ar[i], color='black', ls='--')
    else:
        ax[ix].errorbar(ell, cl, yerr=err, color='k', ls='--')
        ax[ix].loglog(ells_bf_ar[i], cls_bf_ar[i], color='orange', lw=2)
    ax[ix].text(0.9, 0.9, tr1+'-'+tr2, horizontalalignment='right', 
                verticalalignment='top', transform=ax[ix].transAxes)
    ix += 1
    print(ix)

plt.savefig("gcwl_fit.png")
#plt.show()
plt.close()


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
    print("Fractional difference = ",(cl[arg:arg+len(ells_bf_ar[i])] - cls_bf_ar[i]) / err[arg:arg+len(ells_bf_ar[i])])
    print(arg, ix)
    if want_ratio:
        ax[ix].plot(ell[arg:arg+len(ells_bf_ar[i])], cl[arg:arg+len(ells_bf_ar[i])]/cls_bf_ar[i], color='black', ls='--')
    else:
        ax[ix].errorbar(ell, cl, yerr=err, color='k', ls='--')
        ax[ix].loglog(ells_bf_ar[i], cls_bf_ar[i], color='orange', lw=2.5)
    ax[ix].text(0.9, 0.9, tr1+'-'+tr2, horizontalalignment='right', 
               verticalalignment='top', transform=ax[ix].transAxes)
    #ix += 1
plt.savefig("wlwl_fit.png")
#plt.show()
plt.close()

