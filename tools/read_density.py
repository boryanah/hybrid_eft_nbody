import numpy as np

#ppd = 2304 #ICs
ppd = 1152 #ICs_low
fileName = "/mnt/store1/boryanah/ICs_low/density%d"%ppd

#with open(fileName, mode='rb') as file:
#    density = file.read()
density = np.fromfile(fileName, dtype=np.float32).reshape(ppd,ppd,ppd)
np.save("/mnt/gosling1/boryanah/AbacusSummit_hugebase_c000_ph000/density_%d.npy"%ppd,density)
print(density.shape)
