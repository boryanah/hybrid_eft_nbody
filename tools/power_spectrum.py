import numpy as np
from nbodykit.lab import *

def CompensateTSC(w, v):
    for i in range(3):
        wi = w[i]
        tmp = (np.sinc(0.5 * wi / np.pi) ) ** 3
        v = v / tmp
        return v

def get_mesh_list(pos_parts,ones,delta,delta_sq,nabla_sq,s_sq,lagr_pos,Lbox,N_dim,interlaced):
    # get i, j, k for position on the density array
    # TODO: fix
    #lagr_ijk = ((lagr_pos/Lbox+0.5)*N_dim).astype(int)%N_dim
    lagr_ijk = ((lagr_pos/Lbox)*N_dim).astype(int)%N_dim

    # compute the weights for each particle
    weights_ones     = ones[lagr_ijk[:,0],lagr_ijk[:,1],lagr_ijk[:,2]]
    weights_delta    = delta[lagr_ijk[:,0],lagr_ijk[:,1],lagr_ijk[:,2]]
    weights_delta_sq = delta_sq[lagr_ijk[:,0],lagr_ijk[:,1],lagr_ijk[:,2]]
    weights_nabla_sq = nabla_sq[lagr_ijk[:,0],lagr_ijk[:,1],lagr_ijk[:,2]]
    weights_s_sq     = s_sq[lagr_ijk[:,0],lagr_ijk[:,1],lagr_ijk[:,2]]
    
    # get all meshes
    mesh_ones     = get_mesh(pos_parts,weights_ones,N_dim,Lbox,interlaced)
    mesh_delta    = get_mesh(pos_parts,weights_delta,N_dim,Lbox,interlaced)
    mesh_delta_sq = get_mesh(pos_parts,weights_delta_sq,N_dim,Lbox,interlaced)
    mesh_nabla_sq = get_mesh(pos_parts,weights_nabla_sq,N_dim,Lbox,interlaced)
    mesh_s_sq     = get_mesh(pos_parts,weights_s_sq,N_dim,Lbox,interlaced)
    mesh_list = [mesh_ones,mesh_delta,mesh_delta_sq,mesh_nabla_sq,mesh_s_sq]

    return mesh_list


def get_mesh(pos_parts,weights_this,N_dim,Lbox,interlaced):
    # create nbodykit objects for all fields
    this_field = {}
    this_field['Position'] = pos_parts
    this_field['Weight'] = weights_this
    cat = ArrayCatalog(this_field)
    mesh_this = cat.to_mesh(window='tsc',Nmesh=N_dim,BoxSize=Lbox,interlaced=interlaced,compensated=False)
    compensation = CompensateTSC#mesh_this.CompensateTSC
    mesh_this = mesh_this.apply(compensation, kind='circular', mode='complex')
    
    return mesh_this

def get_cross_ps(first_mesh,second_mesh):
    r_first_second = FFTPower(first=first_mesh, second=second_mesh, mode='1d')#, dk=0.005, kmin=0.01)   
    Pk_first_second = r_first_second.power['power']#.real
    ks = r_first_second.power['k'] # [Mpc/h]^-1
    return ks, Pk_first_second

def predict_Pk(f_params,ks_all,Pk_all,k_lengths):
    k_starts = np.zeros(len(k_lengths),dtype=k_lengths.dtype)
    k_starts[1:] = np.cumsum(k_lengths)[:-1]
    Pk_predicted = np.zeros(k_lengths[0],dtype=np.float64) # assuming all are equal
    f_params = f_params.astype(np.float64)
    i_all = -1
    for i in range(len(f_params)):
        for j in range(len(f_params)):
            if j < i: continue
            
            i_all += 1
            start = k_starts[i_all]
            length = k_lengths[i_all]
            
            ks_ij = ks_all[start:start+length]
            Pk_ij = Pk_all[start:start+length]
            
            Pk_predicted += f_params[i]*f_params[j]*Pk_ij
            
    return Pk_predicted

def get_all_cross_ps(mesh_list):
    k_lengths = []
    for i in range(len(mesh_list)):
        for j in range(len(mesh_list)):
            if j < i: continue
            print(i,j)
            ks_ij, Pk_ij = get_cross_ps(mesh_list[i],mesh_list[j])
            k_lengths.append(len(ks_ij))
            try:
                Pk_all = np.hstack((Pk_all,Pk_ij))
                ks_all = np.hstack((ks_all,ks_ij))
            except:
                Pk_all = Pk_ij
                ks_all = ks_ij
    k_lengths = np.array(k_lengths,dtype=int)
    return ks_all, Pk_all, k_lengths

def get_Pk_true(pos_true,N_dim,Lbox,interlaced):
    # calculate power spectrum of the galaxies or halos
    galaxies = {}
    galaxies['Position'] = pos_true

    # create mesh object
    cat = ArrayCatalog(galaxies)
    mesh_true = cat.to_mesh(window='tsc',Nmesh=N_dim,BoxSize=Lbox,interlaced=interlaced,compensated=False)
    compensation = CompensateTSC#mesh_true.CompensateTSC
    mesh_true = mesh_true.apply(compensation, kind='circular', mode='complex')

    # obtain the "truth"
    r_true = FFTPower(mesh_true, mode='1d')
    ks = r_true.power['k']
    Pk_true = r_true.power['power']#.real

    return ks, Pk_true

