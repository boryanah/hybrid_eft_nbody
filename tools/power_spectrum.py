import numpy as np
from nbodykit.lab import *

from nbodykit.io.base import FileType
from nbodykit.source.catalog.file import FileCatalogFactory
from nbodykit.source.catalog import FITSCatalog


class NPYFile(FileType):
    """
    A file-like object to read np ``.npy`` files
    """
    def __init__(self, path):
        self.path = path
        self.attrs = {}
        self._data = np.load(self.path)
        self.size = len(self._data) # total size
        self.dtype = self._data.dtype # data dtype

    def read(self, columns, start, stop, step=1):
        """
        Read the specified column(s) over the given range
        """
        return self._data[start:stop:step]

#NPYCatalog = FileCatalogFactory('NPYCatalog', NPYFile)
#cat = NPYCatalog(pos_parts_fns)

def CompensateTSC(w, v):
    for i in range(3):
        wi = w[i]
        tmp = (np.sinc(0.5 * wi / np.pi) ) ** 3
        v = v / tmp
        return v

def get_mesh(pos_parts_fns,N_dim,Lbox,interlaced):

    # create catalog from fitsfile
    cat = FITSCatalog(pos_parts_fns, ext='Data') 
    
    mesh = cat.to_mesh(window='tsc',Nmesh=N_dim,BoxSize=Lbox,interlaced=interlaced,compensated=False)
    compensation = CompensateTSC # mesh.CompensateTSC not working
    mesh = mesh.apply(compensation, kind='circular', mode='complex')

    return mesh

def get_cross_ps(first_mesh,second_mesh,dk=None):
    r_cross = FFTPower(first=first_mesh, second=second_mesh, mode='1d', dk=dk)
    Pk_cross = r_cross.power['power']#.real
    ks = r_cross.power['k'] # [Mpc/h]^-1
    P_sn = r_cross.attrs['shotnoise']
    # TESTING perhaps don't subtract
    #Pk_cross -= P_sn
    return ks, Pk_cross

def predict_Pk(f_params,ks_all,Pk_all,k_lengths):
    k_starts = np.zeros(len(k_lengths),dtype=k_lengths.dtype)
    k_starts[1:] = np.cumsum(k_lengths)[:-1]
    Pk_predicted = np.zeros(k_lengths[0],dtype=np.float64) # assuming all are equal
    f_params = f_params.astype(np.float64)
    i_all = 0
    for i in range(len(f_params)):
        for j in range(len(f_params)):
            if j < i: continue
            
            start = k_starts[i_all]
            length = k_lengths[i_all]
            
            ks_ij = ks_all[start:start+length]
            Pk_ij = Pk_all[start:start+length]
            
            Pk_predicted += f_params[i]*f_params[j]*Pk_ij

            i_all += 1
    return Pk_predicted

def get_all_cross_ps(mesh_list,dk=None):
    k_lengths = []
    for i in range(len(mesh_list)):
        for j in range(len(mesh_list)):
            if j < i: continue
            print(i,j)
            ks_ij, Pk_ij = get_cross_ps(mesh_list[i],mesh_list[j],dk)
            k_lengths.append(len(ks_ij))
            try:
                Pk_all = np.hstack((Pk_all,Pk_ij))
                ks_all = np.hstack((ks_all,ks_ij))
            except:
                Pk_all = Pk_ij
                ks_all = ks_ij
    k_lengths = np.array(k_lengths,dtype=int)
    Pk_all = Pk_all.astype(np.float64)
    
    return ks_all, Pk_all, k_lengths

def get_Pk_arr(pos1,N_dim,Lbox,interlaced,dk=None,pos2=None):
    first = {}
    first['Position'] = pos1

    # create mesh object
    cat = ArrayCatalog(first)
    mesh1 = cat.to_mesh(window='tsc',Nmesh=N_dim,BoxSize=Lbox,interlaced=interlaced,compensated=False)
    compensation = CompensateTSC # mesh1.CompensateTSC not working
    mesh1 = mesh1.apply(compensation, kind='circular', mode='complex')

    if pos2 is None:
        mesh2 = None
    else:
        second = {}
        second['Position'] = pos2

        # create mesh object
        cat = ArrayCatalog(second)
        mesh2 = cat.to_mesh(window='tsc',Nmesh=N_dim,BoxSize=Lbox,interlaced=interlaced,compensated=False)
        compensation = CompensateTSC # mesh2.CompensateTSC not working
        mesh2 = mesh2.apply(compensation, kind='circular', mode='complex')

    # obtain the "truth"
    r = FFTPower(first=mesh1, second=mesh2, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    Pk -= P_sn
    return ks, Pk

def get_Pk(pos1_fns,N_dim,Lbox,interlaced,dk=None,pos2_fns=None):
    # calculate power spectrum of the galaxies or halos
    mesh1 = get_mesh(pos1_fns,N_dim,Lbox,interlaced)

    if pos2_fns is None:
        mesh2 = None
    else:
        mesh2 = get_mesh(pos2_fns,N_dim,Lbox,interlaced)
        
    # obtain the "truth"
    r = FFTPower(first=mesh1, second=mesh2, mode='1d', dk=dk)
    ks = r.power['k']
    Pk = r.power['power'].astype(np.float64)
    P_sn = r.attrs['shotnoise']
    Pk -= P_sn
    return ks, Pk

def resample_mesh(field_old,Lbox,N_dim_new):
    mesh_old = ArrayMesh(field_old, BoxSize=Lbox)
    mesh_new = mesh_old.paint(mode='real', Nmesh=N_dim_new)
    mesh_new = ArrayMesh(mesh_new, BoxSize=Lbox)

    return mesh_new

