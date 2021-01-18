import sacc
from astropy.io import fits

filename = "data/cls_covG_covNG_DESgc_DESwl.fits"

s = sacc.Sacc.load_fits(filename)

print((s.mean).shape)
print((s.mean)[:100])


dt = s.get_data_types()

for i, tracer in enumerate(s.tracers):
    key = tracer
    print(key)
    #print(s.tracers[key].z)
    #print(s.tracers[key].nz)

for tr1, tr2 in s.get_tracer_combinations():
    print(tr1, tr2)
quit()
    
for i in range(len(dt)):
    t = s.get_tracer_combinations(dt[i])

    print(t[0][0],t[0][1])
    print(s.has_covariance())
    print(s.get_ell_cl(dt[i],t[0][0],t[0][1]))

    print(s.get_mean(dt[i],t[0]))
    quit()

hdu = fits.open(filename)

cov = sacc.covariance.BaseCovariance.from_hdu(hdu['covariance'])

icov = cov.inverse



#dt=['cl_00', 'cl_0e', 'cl_0b', 'cl_ee', 'cl_eb', 'cl_be', 'cl_bb']
