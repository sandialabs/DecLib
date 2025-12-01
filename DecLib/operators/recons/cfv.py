from numba import njit, prange
		
#THIS ALL LEANS HEAVILY ON UNIVERSAL BASIS FOR BVDFs...

@njit(parallel=True, cache=True)
def _apply_vform_recon_cfv2(vformarr, reconarr, velocity, nedges, ndofs, bundle_size, CE, cellareas):
    for e in prange(nedges):
        for l in range(ndofs):
            for d in range(bundle_size):
                reconarr[e,l,d] = (vformarr[CE[e,0],l,d]/cellareas[CE[e,0]] + vformarr[CE[e,1],l,d]/cellareas[CE[e,1]])/2.

@njit(parallel=True, cache=True)
def _apply_vform_recon_cfv2_scaled(vformarr, reconarr, velocity, scalearr, nedges, scaledoff, ndofs, bundle_size, CE, cellareas):
    for e in prange(nedges):
        scaleval = (scalearr[CE[e,0],scaledoff]/cellareas[CE[e,0]] + scalearr[CE[e,1],scaledoff]/cellareas[CE[e,1]])/2.
        for l in range(ndofs):
            for d in range(bundle_size):
                reconarr[e,l,d] = (vformarr[CE[e,0],l,d]/cellareas[CE[e,0]] + vformarr[CE[e,1],l,d]/cellareas[CE[e,1]])/(2. * scaleval)

@njit(parallel=True, cache=True)
def _apply_vform_recon_cfv4(vformarr, reconarr, velocity, nedges, ndofs, bundle_size, CE, cellareas):
    for e in prange(nedges):
        for l in range(ndofs):
            for d in range(bundle_size):
                reconarr[e,l,d] = -1./12. * vformarr[CE[e,0],l,d]/cellareas[CE[e,0]] + 2./3. * vformarr[CE[e,1],l,d]/cellareas[CE[e,1]] + \
                2./3. * vformarr[CE[e,2],l,d]/cellareas[CE[e,2]] + -1./12. * vformarr[CE[e,3],l,d]/cellareas[CE[e,3]]
           
@njit(parallel=True, cache=True)
def _apply_vform_recon_cfv4_scaled(vformarr, reconarr, velocity, scalearr, nedges, scaledoff, ndofs, bundle_size, CE, cellareas):
    for e in prange(nedges):
        scaleval = (scalearr[CE[e,1],scaledoff]/cellareas[CE[e,1]] + scalearr[CE[e,2],scaledoff]/cellareas[CE[e,2]])/2.
        for l in range(ndofs):
            for d in range(bundle_size):
                reconarr[e,l,d] = (-1./12. * vformarr[CE[e,0],l,d]/cellareas[CE[e,0]] + 2./3. * vformarr[CE[e,1],l,d]/cellareas[CE[e,1]] + \
                2./3. * vformarr[CE[e,2],l,d]/cellareas[CE[e,2]] + -1./12. * vformarr[CE[e,3],l,d]/cellareas[CE[e,3]])/scaleval

           
@njit(parallel=True, cache=True)
def _apply_vform_recon_cfv6(vformarr, reconarr, velocity, nedges, ndofs, bundle_size, CE, cellareas):
    for e in prange(nedges):
        for l in range(ndofs):
            for d in range(bundle_size):
                reconarr[e,l,d] = 1./60. * vformarr[CE[e,0],l,d]/cellareas[CE[e,0]] + -3./20. * vformarr[CE[e,1],l,d]/cellareas[CE[e,1]] + \
                23./30. * vformarr[CE[e,2],l,d]/cellareas[CE[e,2]] + 23./30. * vformarr[CE[e,3],l,d]/cellareas[CE[e,3]] + \
                + -3./20. * vformarr[CE[e,4],l,d]/cellareas[CE[e,4]] + 1./60. * vformarr[CE[e,5],l,d]/cellareas[CE[e,5]]

@njit(parallel=True, cache=True)
def _apply_vform_recon_cfv6_scaled(vformarr, reconarr, velocity, scalearr, nedges, scaledoff, ndofs, bundle_size, CE, cellareas):
    for e in prange(nedges):
        scaleval = (scalearr[CE[e,2],scaledoff]/cellareas[CE[e,2]] + scalearr[CE[e,3],scaledoff]/cellareas[CE[e,3]])/2.
        for l in range(ndofs):
            for d in range(bundle_size):
                reconarr[e,l,d] = (1./60. * vformarr[CE[e,0],l,d]/cellareas[CE[e,0]] + -3./20. * vformarr[CE[e,1],l,d]/cellareas[CE[e,1]] + \
                23./30. * vformarr[CE[e,2],l,d]/cellareas[CE[e,2]] + 23./30. * vformarr[CE[e,3],l,d]/cellareas[CE[e,3]] + \
                + -3./20. * vformarr[CE[e,4],l,d]/cellareas[CE[e,4]] + 1./60. * vformarr[CE[e,5],l,d]/cellareas[CE[e,5]])/scaleval
