from nbodykit.lab import ArrayCatalog, FFTPower
import matplotlib.pyplot as plt
import numpy as np


def CompensateTSC(w, v):
    for i in range(3):
        wi = w[i]
        tmp = (np.sinc(0.5 * wi / np.pi) ) ** 3
        v = v / tmp
        return v
    

interlaced = True
window = 'TSC'
Lbox = 175.
N_dim = 256
k_ny = np.pi * N_dim / Lbox


#pos_parts = np.vstack((np.random.rand(50000),np.random.rand(50000),np.random.rand(50000))).T*Lbox
#pos_parts_other = np.vstack((np.random.rand(50000),np.random.rand(50000),np.random.rand(50000))).T*Lbox
directory = "/mnt/gosling1/boryanah/small_box_damonge/"
pos_parts = np.load(directory+"pos_halo.npy")

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
mesh_weighted = get_mesh(pos_parts,weights=pos_parts[:,0])
#mesh_weighted = get_mesh(pos_parts_other,weights=pos_parts[:,0])

# obtain the "truth"
r_true = FFTPower(mesh, mode='1d')
ks = r_true.power['k']
Pk_true = r_true.power['power']#.real

# obtain the cross-correlation
r = FFTPower(first=mesh,second=mesh_weighted, mode='1d')
ks = r.power['k']
Pk_cross = r.power['power']#.real

r = FFTPower(first=mesh_weighted,second=mesh_weighted, mode='1d')
ks = r.power['k']
Pk_other = r.power['power']#.real

plt.figure(1)
plt.plot(ks,Pk_true,label='auto1')
plt.plot(ks,Pk_other,label='auto2')
plt.plot(ks,Pk_cross,label='cross')
plt.axvline(x=k_ny, c='k', ls='--', label="Nyquist frequency for Nmesh=256")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
