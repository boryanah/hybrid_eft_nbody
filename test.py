from nbodykit.lab import ArrayCatalog, FFTPower
import matplotlib.pyplot as plt
import numpy as np


def CompensateTSC(w, v):
    for i in range(3):
        wi = w[i]
        tmp = (np.sinc(0.5 * wi / np.pi) ) ** 3
        v = v / tmp
        return v

machine = 'NERSC'#'alan'#'NERSC'
interlaced = True
window = 'TSC'
Lbox = 175.
N_dim = 256
gr_size = Lbox/N_dim
k_ny = np.pi * N_dim / Lbox

#pos_parts = np.vstack((np.random.rand(50000),np.random.rand(50000),np.random.rand(50000))).T*Lbox
#pos_parts_other = np.vstack((np.random.rand(50000),np.random.rand(50000),np.random.rand(50000))).T*Lbox


if machine == 'NERSC':
    directory = "/global/cscratch1/sd/damonge/NbodySims/Sim256/"
    data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/gadget/"
elif machine == 'alan':
    directory = "/mnt/gosling1/boryanah/small_box_damonge/"
    data_dir = "/mnt/gosling1/boryanah/small_box_damonge/output/"
pos_ic = np.load(data_dir+"pos_ic.npy")
#nabla_sq = np.load(data_dir+"density.npy")
nabla_sq = np.load(data_dir+"nabla_sq_4.npy")
#nabla_sq = np.load(data_dir+"delta_sq_4.npy")
pos_halo = np.load(data_dir+"pos_halo.npy")
pos_parts = np.load(data_dir+"pos_snap.npy")

pos_ijk = (pos_ic/gr_size).astype(int)%N_dim
weights = nabla_sq[pos_ijk[:,0],pos_ijk[:,1],pos_ijk[:,2]]
weights /= np.sum(weights)
print(weights[:5])

N_halo = pos_halo.shape[0]
n_halo = N_halo/Lbox**3.
P_sn = 1./n_halo

def get_mesh(pos_parts,weights):
    this_field = {}
    this_field['Position'] = pos_parts
    #this_field['Weights'] = weights
    this_field['Weight'] = weights
    cat = ArrayCatalog(this_field)

    # convert catalog to a mesh with desired window and interlacing
    mesh = cat.to_mesh(Nmesh=N_dim, window=window, BoxSize=Lbox, compensated=False, interlaced=interlaced)

    # apply correction for the window to the mesh
    compensation = mesh.CompensateCIC if window == 'CIC' else CompensateTSC
    mesh = mesh.apply(compensation, kind='circular', mode='complex')

    return mesh

mesh = get_mesh(pos_parts,weights=np.ones(pos_parts.shape[0]))
mesh_halo = get_mesh(pos_halo,weights=np.ones(pos_halo.shape[0]))
mesh_weighted = get_mesh(pos_parts,weights=weights)
#mesh_weighted = get_mesh(pos_parts_other,weights=pos_parts[:,0])

def get_Pk(mesh1,mesh2=None):
    # obtain the cross-correlation
    r = FFTPower(first=mesh1,second=mesh2, mode='1d')
    ks = r.power['k']
    Pk_cross = r.power['power']#.real
    return ks, Pk_cross

ks, Pk_hh = get_Pk(mesh_halo)
ks, Pk_mm = get_Pk(mesh)
ks, Pk_wmwm = get_Pk(mesh_weighted)
ks, Pk_hm = get_Pk(mesh_halo,mesh)

plt.figure(1)
plt.plot(ks,Pk_mm,label='m,m')
plt.plot(ks,Pk_wmwm,label='wm,wm')
plt.plot(ks,Pk_hm,label='h,m')
plt.plot(ks,Pk_hh,label='h,h')
plt.plot(ks,Pk_hh-P_sn,label='h,h-shot')
plt.axvline(x=k_ny, c='k', ls='--')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig("figs/Pk_test.png")
plt.show()

'''
# they are equivalent
karr = np.fft.fftfreq(N_dim, d=Lbox/(2*np.pi*N_dim))
ksq = np.zeros(shape=(N_dim, N_dim, N_dim),dtype=complex)
ksq = karr[None,None,:]**2+karr[None,:,None]**2+karr[:,None,None]**2

ksq_true = np.zeros(shape=(N_dim, N_dim, N_dim),dtype=np.float)
for a in range(N_dim):
    for b in range(N_dim):
        for c in range(N_dim):
            ksq_true[a,b,c] = karr[a]**2+karr[b]**2+karr[c]**2
'''
