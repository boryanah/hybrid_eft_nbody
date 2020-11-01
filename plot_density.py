import numpy as np
import matplotlib.pyplot as plt

def main():
    # user choices
    Lbox = 175.
    N_dim = 256
    simulation_code = 'gadget'
    machine = 'alan'
    z_nbody = 1.
    R_smooth = 2.
    
    if simulation_code == 'abacus':
        sim_name = "AbacusSummit_hugebase_c000_ph000"#small/AbacusSummit_small_c000_ph3046
        if machine == 'alan':
            data_dir = "/mnt/store1/boryanah/data_hybrid/abacus/"+sim_name+"/z%.3f/"%z_nbody
            dens_dir = "/mnt/store1/boryanah/data_hybrid/abacus/"+sim_name+"/"
        elif machine == 'NERSC':
            data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/abacus/"+sim_name+"/z%.3f/"%z_nbody
            dens_dir = "/global/cscratch1/sd/boryanah/data_hybrid/abacus/"+sim_name+"/"
            
    elif simulation_code == 'gadget':
        sim_name = "Sim256"    
        if machine == 'alan':
            data_dir = "/mnt/gosling1/boryanah/"+sim_name+"/z%.3f/"%z_nbody
            dens_dir = "/mnt/gosling1/boryanah/"+sim_name+"/"
        elif machine == 'NERSC':
            data_dir = "/global/cscratch1/sd/boryanah/data_hybrid/gadget/"+sim_name+"/z%.3f/"%z_nbody
            dens_dir = "/global/cscratch1/sd/boryanah/data_hybrid/gadget/"+sim_name+"/"
            
    pos_snap = np.load(data_dir+"pos_snap.npy")
    #pos_ic = np.load(data_dir+"pos_ic.npy")
    dens = np.load(dens_dir+"density_smooth_%d.npy"%(int(R_smooth)))
    #dens = np.load(dens_dir+"delta_sq_%d.npy"%(int(R_smooth)))

    gr_size = Lbox/N_dim    
    pos_snap = pos_snap[::100]/gr_size
    #pos_ic = pos_ic[::100]/gr_size
    
    slice_id = 20
    Dens_m = dens[:,:,slice_id]
    chosen_slice = (pos_snap[:,2] > slice_id) & (pos_snap[:,2] <= (slice_id+1))

    plt.imshow(np.log10(Dens_m+1),cmap='Greys')
    plt.scatter(pos_snap[chosen_slice,1],pos_snap[chosen_slice,0],marker='*',color='red',s=15)
    #plt.scatter(pos_ic[chosen_slice,1],pos_ic[chosen_slice,0],marker='*',color='blue',s=15)
    plt.savefig("figs/density.png")
    plt.show()

main()
