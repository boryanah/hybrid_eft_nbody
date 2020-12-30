import os

import numpy as np
import yaml
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

from getdist import plots, MCSamples

#import plotparams
#plotparams.default()

# names and choices
name_fid = "test"
yaml_dic = {'test': "../config/bias_pk.yaml"}
lab_fid = 'Hybrid EFT (4-parameter fit to $P^{hh}(k)$)'
home = os.path.expanduser("~")
chain_dir = home+"/repos/hybrid_eft_nbody/chains/"

# what are we plotting
bias_pars = ["f_1","f_2","f_3","f_4"]
filename = "triangle_Pk_hh.png"

def get_par_names(name):
    config = yaml.load(open(yaml_dic[name]))
    emcee_params = config['ch_config_params']
    assert emcee_params['chainsPrefix'] == name, "Names not matching"
    w_rat = emcee_params['walkersRatio']
    b_iter = emcee_params['burninIterations']
    n_iter = emcee_params['sampleIterations']
    par_names = ["f_0","f_1","f_2","f_3","f_4"]
    lab_names = ["F_0","F_1","F_2","F_3","F_4"]
    n_par = len(par_names)
    return n_iter, w_rat, n_par, b_iter, par_names, lab_names


# load the fiducial dataset
# walkers ratio, number of params and burn in iterations
fiducial_outfile = chain_dir+name_fid+".txt"
n_iter, w_rat, n_par, b_iter, par_names, lab_names = get_par_names(name_fid)
fiducial_chains = np.loadtxt(fiducial_outfile)
fiducial_probs = np.loadtxt(chain_dir+name_fid+"prob.txt")

'''
time = np.arange(n_iter+b_iter)
for i in range(w_rat*n_par):
    plt.figure(i+1)
    inds = np.arange(n_iter+b_iter)*w_rat*n_par+i
    plt.plot(time,fiducial_chains[inds,1])
    plt.plot(time,fiducial_chains[inds,2])
    plt.plot(time,fiducial_chains[inds,3])
    plt.plot(time,fiducial_chains[inds,4])
    #plt.plot(time,fiducial_probs[inds])
    #plt.savefig("test.png")
plt.show()
'''

max_probs = np.argmax(fiducial_probs)


# removing burn-in
fiducial_chains = fiducial_chains[w_rat*n_par*b_iter:]

# load samples
fiducial_hsc = MCSamples(samples=fiducial_chains,names=par_names,labels=lab_names,name_tag='Fid')

def print_bounds(ms):
    mean_vals = np.ones((len(par_names),len(ms)))
    for i in range(1,len(par_names)):
        string_all = ""
        for j in range(len(ms)):
            m = ms[j]
            f1 = m.getLatex(par_names,limit=1)
            p, v1 = f1
            f2 = m.getLatex(par_names,limit=2)
            p, v2 = f2
            value = v2[i]
            mean_vals[i,j] = float(value.split('^')[0])
            value = '^'+value.split('^')[-1]
            string_all += "$"+v1[i]+"$, $"+value+"$"
            if j != len(ms)-1:
                string_all += " & "
        print("$"+p[i]+"$ & "+string_all+" \\\\ [1ex]")
    return mean_vals

print("\\begin{table}")
print("\\begin{center}")
print("\\begin{tabular}{c | c c c c} ")
print(" \\hline\\hline")
print(" Parameter & HSC cov. [68\%, 95\%] & CV constraints [68\%, 95\%] & Marg. $N(z)$ [68\%, 95\%] \\\\ [0.5ex] ")
print(" \\hline")
#print(" $\\chi^2/\\nu$ & 87.49/80 & 88.29/80 & 88.54/82.32 \\\\ ")
margs = [fiducial_hsc]
means = print_bounds(margs)
print(" \\hline")
print(" \\hline")
print("\\end{tabular}")
print("\\end{center}")
print("\\label{tab:chi2_tests}")
print("\\caption{Table}")
print("\\end{table}")

print("F_mean = ",means[:,0])
print("F_best = np.array("+np.array2string(fiducial_chains[max_probs],separator = ', ')+")")
print("lnprob_best = ",fiducial_probs[max_probs])

# Triangle plot
g = plots.getSubplotPlotter()
g.settings.legend_fontsize = 14
g.settings.scaling_factor = 0.1
g.triangle_plot([fiducial_hsc],params=bias_pars,legend_labels=[lab_fid],colors=['#E03424'],filled=True)
plt.savefig("../figs/"+filename)
plt.show()
plt.close()
