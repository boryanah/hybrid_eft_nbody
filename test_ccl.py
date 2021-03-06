import os

import asdf
import pyccl as ccl
import numpy as np
from classy import Class
import matplotlib.pyplot as plt

def get_A_s(sigma8):
    A_s = (sigma8/fid_deriv_params['sigma8_cb'])**2*fid_A_s
    return A_s

# B.H. names of the bias EFT parameters
bias_eft_names = ['b1', 'b2', 'bn', 'bs']

# load the fiducial template and the derivatives
R_smooth = 0.
#data_dir = os.path.expanduser("~/repos/hybrid_eft_nbody/data/AbacusSummit_base_c000_ph006/z0.100/")
data_dir = os.path.expanduser("~/repos/hybrid_eft_nbody/data/AbacusSummit_base_c000_ph000/z0.100/")

want_rat = 1
if want_rat:
    fid_file = os.path.join(data_dir, "fid_rat_Pk_dPk_templates_%d.asdf"%(int(R_smooth)))
else:
    fid_file = os.path.join(data_dir, "fid_Pk_dPk_templates_%d.asdf"%(int(R_smooth)))
    
with asdf.open(fid_file, lazy_load=False, copy_arrays=True) as f:
    fid_dPk_Pk_templates = f['data']
    ks = f['data']['ks']
    header = f['header']

# get just the keys for the templates
template_comb_names = []
for i, bi in enumerate(bias_eft_names):
    for j, bj in enumerate(bias_eft_names):
        if j < i: continue
        template_comb_names.append(bi+'_'+bj)
        
for bi in bias_eft_names:
    template_comb_names.append('1'+'_'+bi)
    template_comb_names.append('1'+'_'+'1')
    template_comb_names = template_comb_names

# parameters used for the EFT approach: fid_deriv are used with the derivatives; other params are just to initialize CLASS
deriv_param_names = ['omega_b', 'omega_cdm', 'n_s', 'sigma8_cb']

# separate header items into cosmology parameters and redshift of the tempaltes
fid_cosmo = {}
fid_deriv_params = {}
z_templates = {}
for key in header.keys():
    if key in deriv_param_names:
        fid_cosmo[key] = header[key]
        fid_deriv_params[key] = header[key]
    elif 'ztmp' in key:
        z_templates[key] = header[key]
    elif 'A_s' == key:
        fid_A_s = header[key]
    elif 'theta_s_100' == key:
        theta_s_100 = header[key]
    else:
        #pass
        fid_cosmo[key] = header[key]

print(fid_cosmo.items())

#fid_cosmo['T_cmb'] = (ccl.physical_constants.T_CMB)
#fid_cosmo['recombination'] = 'RECFAST'
#fid_cosmo['tau_reio'] = 0.08

# remove the sigma8_cb parameter as CLASS uses A_s
fid_cosmo.pop('sigma8_cb')
fid_cosmo['A_s'] = fid_A_s

# initilize  the fiducial cosmology to check that you recover the theta_s from the header
class_cosmo = Class()
class_cosmo.set(fid_cosmo)
class_cosmo.compute()
fid_theta = class_cosmo.theta_s_100()
#assert np.abs(header['theta_s_100'] - fid_theta) < 1.e-6, "CLASS not initialized properly"
print(header['theta_s_100'],fid_theta)

# fiducial cosmology with all CLASS parameters
#fid_cosmo['100*theta_s'] = fid_theta
fid_cosmo['100*theta_s'] = header['theta_s_100']

# removing h since the parameter that is held fixed is 100*theta_s
try:
    fid_cosmo.pop('h')
except:
    pass





# B.H. parameters for the ccl object
param_dict = {'transfer_function': 'boltzmann_class'}

# setting values of cosmological parameters
fac = 64#16
#par = 'sigma8_cb'; h_small = 0.00406003#*4.
#h_small = 0.
h_small = 0.005*fac
param_dict['sigma8'] = fid_deriv_params['sigma8_cb'] + h_small

#h_small = 0.
h_small = 0.0005*16#fac
param_dict['omega_b'] = fid_deriv_params['omega_b'] + h_small

#par = 'omega_cdm'; h_small = 0.001000002*4.
#h_small = 0.0
h_small = 0.005*8
param_dict['omega_cdm'] = fid_deriv_params['omega_cdm'] + h_small

#par = 'n_s'; h_small = 0.0029999614#*4.
#h_small = 0.
h_small = 0.005*fac
param_dict['n_s'] = fid_deriv_params['n_s'] + h_small
# -16 bad; +16 ok; 0.05 for cdm ok

# get current values
sigma8_cb = param_dict['sigma8']
omega_cdm = param_dict['omega_cdm']
omega_b = param_dict['omega_b']
n_s = param_dict['n_s']
A_s = get_A_s(sigma8_cb)
print(omega_cdm, omega_b, sigma8_cb, A_s, n_s)

# updated dictionary without sigma8 because class takes only A_s and not sigma8
updated_dict = {'A_s': A_s, 'omega_b': omega_b, 'omega_cdm': omega_cdm, 'n_s': n_s}

# update the CLASS object with the current parameters
class_cosmo = Class()

# update the cosmology
new_cosmo = {**fid_cosmo, **updated_dict}
new_cosmo['output'] = 'mPk'
new_cosmo['z_max_pk'] = 1.1
class_cosmo.set(new_cosmo)
class_cosmo.compute()

# search for the corresponding value of H0 that keeps theta_s constant and update Omega_b and c
h = class_cosmo.h()
#h = H0_search(class_cosmo, fid_cosmo['100*theta_s'], prec=1.e4, tol_t=1.e-4)
print(h)
param_dict['h'] = h
param_dict['Omega_c'] = omega_cdm/h**2
param_dict['Omega_b'] = omega_b/h**2
param_dict['A_s'] = A_s

# remove parameters not recognized by ccl
param_not_ccl = ['100*theta_s', 'omega_cdm', 'omega_b', 'output', 'sigma8']
for p in param_not_ccl:
    if p in param_dict.keys(): param_dict.pop(p)

#param_dict['T_cmb'] = header['T_cmb']
# cosmology of the current step
cosmo_ccl = ccl.Cosmology(**param_dict)

# interpolate for the cosmological parameters that are being deriv
Pk_a_ij = {}
a_arr = np.zeros(len(z_templates))
# tuks
pk_a = np.zeros((len(z_templates), len(ks)))
# for a given redshift
for combo in template_comb_names:
    Pk_a = np.zeros((len(z_templates), len(ks)))
    for i in range(len(z_templates)):
        z_str = 'ztmp%d'%i
        a_arr[i] = 1./(1+z_templates[z_str])
        key = z_str+'_'+combo
        Pk = fid_dPk_Pk_templates[key] + \
            fid_dPk_Pk_templates[key+'_'+'omega_b'] * (omega_b - fid_deriv_params['omega_b']) + \
            fid_dPk_Pk_templates[key+'_'+'omega_cdm'] * (omega_cdm - fid_deriv_params['omega_cdm']) + \
            fid_dPk_Pk_templates[key+'_'+'n_s'] * (n_s - fid_deriv_params['n_s']) + \
            fid_dPk_Pk_templates[key+'_'+'sigma8_cb'] * (sigma8_cb - fid_deriv_params['sigma8_cb'])

        # tuks
        pk_a[i, :] = ccl.nonlin_matter_power(cosmo_ccl, ks*h, a=1./(1+z_templates[z_str]))
        
        if want_rat:
            Pk *= ccl.nonlin_matter_power(cosmo_ccl, ks*h, a=1./(1+z_templates[z_str]))*h**3
        # convert to Mpc^3 rather than [Mpc/h]^3
        Pk_a[i, :] = Pk/h**3.
    Pk_a_ij[combo] = Pk_a
# convert to Mpc^-1 rather than h/Mpc
Pk_a_ij['lk_arr'] = np.log(ks*h)


# tuks
pk_a *= 1.41
pk_tmp = ccl.Pk2D(a_arr=a_arr, lk_arr=np.log(ks*h), pk_arr=pk_a, is_logp=False)
ells = np.geomspace(2,1000,20)
z_arr = np.linspace(0,4,100)[::-1]
nz_arr = np.exp(-((z_arr-0.5)/0.05)**2/2)
bz_arr = np.ones(len(z_arr))*1.41
g_tracer = ccl.NumberCountsTracer(cosmo_ccl, has_rsd=False, dndz=(z_arr[::-1], nz_arr[::-1]), bias=(z_arr[::-1], bz_arr[::-1]))
g_1_tracer = ccl.NumberCountsTracer(cosmo_ccl, has_rsd=False, dndz=(z_arr[::-1], nz_arr[::-1]), bias=(z_arr[::-1], np.ones(len(z_arr))))
wl_tracer = ccl.WeakLensingTracer(cosmo_ccl, dndz=(z_arr[::-1], nz_arr[::-1]))
cl_gg = ccl.angular_cl(cosmo_ccl, g_tracer, wl_tracer, ells)
print(cl_gg)
cl_gg_tmp = ccl.angular_cl(cosmo_ccl, g_1_tracer, wl_tracer, ells, p_of_k_a=pk_tmp)
print(cl_gg_tmp)
print((cl_gg-cl_gg_tmp)/cl_gg)
plt.plot(ells, cl_gg)
plt.plot(ells, cl_gg_tmp)
plt.xscale('log')
plt.yscale('log')
plt.show()
quit()

for i in range(7):
    z_test = z_templates['ztmp%d'%i]
    NL = ccl.nonlin_matter_power(cosmo_ccl, ks*h, a=1./(1+z_test))
    print(z_test, h)

    class_ks = np.logspace(-5, np.log10(1), 1000)
    class_pk = np.array([class_cosmo.pk(ki, z_test) for ki in class_ks])
    lk = Pk_a_ij['lk_arr']
    Pk = Pk_a_ij['1_1'][i,:]

    plt.figure(i+1)
    '''
    plt.semilogy(lk, NL, label="CCL")
    plt.semilogy(lk, Pk, label="N-body")
    plt.plot(lk, np.interp(lk, np.log(class_ks), class_pk), label="CLASS")
    '''
    plt.plot(lk, np.ones(len(lk)), 'k--')
    plt.plot(lk, Pk/NL, label="N-body/CCL")
    try:
        Pk_11 = np.load("Pk_11_"+par+"_ztmp%d.npy"%i)/h**3
        plt.plot(lk, Pk_11/NL, label="N-body der/CCL")
    except:
        pass
    plt.plot(lk, np.interp(lk, np.log(class_ks), class_pk)/NL, label="CLASS/CCL")
    plt.ylim([0.9, 1.1])
    plt.legend()
plt.show()
