import numpy as np
import matplotlib.pyplot as plt

def main():
    pos_snap = np.load("/mnt/gosling1/boryanah/small_box_damonge/pos_snap.npy")
    pos_ic = np.load("/mnt/gosling1/boryanah/small_box_damonge/pos_ic.npy")

    dens = np.load("../data/density_smooth_4.npy")
    #dens = np.load("../data/delta_sq_4.npy")

    Lbox = 175.
    N_dim = 256
    gr_size = Lbox/N_dim

    
    start = int(pos_snap.shape[0]//2)
    stop = start+100000
    pos_snap = pos_snap[::100]/gr_size#pos_snap[start:stop]/gr_size
    pos_ic = pos_ic[::100]/gr_size#pos_ic[start:stop]/gr_size
    
    slice_id = 20
    Dens_m = dens[:,:,slice_id]
    chosen_slice = (pos_snap[:,2] > slice_id) & (pos_snap[:,2] < (slice_id+1))

    plt.imshow(np.log10(Dens_m+1),cmap='Greys')
    plt.scatter(pos_snap[chosen_slice,1],pos_snap[chosen_slice,0],marker='*',color='red',s=15)
    plt.scatter(pos_ic[chosen_slice,1],pos_ic[chosen_slice,0],marker='*',color='blue',s=15)
    plt.savefig("tmp.png")
    plt.show()

main()
