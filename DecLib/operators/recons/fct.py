from numba import njit, prange

#@njit(parallel=True, cache=True)
def _fct_pd(dens, flux, phi, EC, nEC, cellorients, ncells, ndofs, bsize, dt):
    phi[:,:,:] = 1.0
    eps = 0.0
    for c in range(ncells):
        for l in range(ndofs):
            for b in range(bsize):
                flux_out = 0.0
                for j in range(nEC[c]):
                    e = EC[c,j]
                    orient = cellorients[c,j]
                    flux_out += max(orient * flux[e,l,b], 0.0)
                mass_available = max(dens[c,l,b], 0.0)
                mass_out = flux_out * dt
                if (mass_out > mass_available):
                    for j in range(nEC[c]):
                        e = EC[e,j]
                        orient = cellorients[c,j]
                        if (orient * flux[e,l,b] > 0): 
                            phi[e,l,b] = min(1., mass_available / (mass_out + eps))
                            #print(phi[e,l,b], mass_available, mass_out)
#THIS IS STILL A LITTLE BROKEN- DO SOME WORK/THINKING!

#does clever FCT rescaling of recon to enforce local BP- ie scaled density (specific quantity) stays bounded by local values ie within a Lagrangian volume!
#this is going to be a little tricky, need to think about it
#probably scaling outgoing flux so lower bound is satisfied, and incoming flux so upper bound is satisfied
@njit(parallel=True, cache=True)
def _fct_bp(dens, flux, phi, phi_min, phi_max, EC, nEC, NC, cellorients, ncells, ndofs, bsize, dt):
    phi[:,:,:] = 1.0
    for c in prange(ncells):
        for l in range(ndofs):
            for b in range(bsize):
                flux_out = 0.0
                flux_in = 0.0
                maxval = dens[c,l,b]
                minval = dens[c,l,b]
                for j in range(nEC[c]):
                    e = EC[c,j]
                    c1 = NC[c,j]
                    orient = cellorients[c,j]
                    flux_out += max(orient * flux[e,l,b], 0.0)
                    flux_in += min(orient * flux[e,l,b], 0.0)
                    #NOT SURE THIS MIN/MAX SETTING IS WHAT WE WANT, BUT IT IS A GOOD FIRST CRACK I THINK...
                    maxval = max(maxval, dens[c1,l,b])
                    minval = min(minval, dens[c1,l,b])
                mass_out = flux_out * dt
                mass_in = flux_in * dt
                mass = dens[c,l,b]
                if ((mass - mass_out) < minval):
                    for j in range(nEC[c]):
                        e = EC[e,j]
                        orient = cellorients[c,j]
                        if (orient * flux[e,l] > 0): phi_min[e,l,b] = (mass-minval)/mass_out
                if ((mass + mass_in) > maxval):
                    for j in range(nEC[c]):
                        e = EC[e,j]
                        orient = cellorients[c,j]
                        if (orient * flux[e,l] < 0): phi_max[e,l,b] = (maxval-mass)/mass_in
    for e in prange(nedges):
        for l in range(ndofs):
            for b in range(bsize):
                phi[e,l,b] = min(phi_min[e,l,b], phi_max[e,l,b])
    
 # 521      // q_{n+1} = q_{n} + dt*f(q_{n})
 # 522       real5d tracers_mult_x("tracers_mult_x",nz,ny,nx,nens) = 1;
 # 523       real5d tracers_mult_y("tracers_mult_y",nz,ny,nx,nens) = 1;
 # 524       real5d tracers_mult_z("tracers_mult_z",nz,ny,nx,nens) = 1;
 # 525       real lbound = 0;
 # 526       real ubound = 1;
 # 527       parallel_for( YAKL_AUTO_LABEL() , Bounds<5>(num_tracers,nz,ny,nx,nens) ,
 # 528                                         YAKL_LAMBDA (int tr, int k, int j, int i, int iens) {
 # 529         if (tracer_positive(tr)) {
 # 530           real mass_available = std::max(previous_mass(tr,k,j,i,iens)-lbound,0._fp) * dx * dy * dz;
 # 531           real flux_out_x = ( max(tracers_flux_x(tr,k,j,i+1,iens),0._fp) - min(tracers_flux_x(tr,k,j,i,iens),0._fp) ) / dx;
 # 532           real flux_out_y = ( max(tracers_flux_y(tr,k,j+1,i,iens),0._fp) - min(tracers_flux_y(tr,k,j,i,iens),0._fp) ) / dy;
 # 533           real flux_out_z = ( max(tracers_flux_z(tr,k+1,j,i,iens),0._fp) - min(tracers_flux_z(tr,k,j,i,iens),0._fp) ) / dz;
 # 534           real mass_out = (flux_out_x + flux_out_y + flux_out_z) * dt * dx * dy * dz;
 # 535           if (mass_out > mass_available) {
 # 536             real mult = mass_available / mass_out;
 # 537             if (tracers_flux_x(tr,k,j,i+1,iens) > 0) tracers_mult_x(tr,k,j,i+1,iens) = mult;
 # 538             if (tracers_flux_x(tr,k,j,i  ,iens) < 0) tracers_mult_x(tr,k,j,i  ,iens) = mult;
 # 539             if (tracers_flux_y(tr,k,j+1,i,iens) > 0) tracers_mult_y(tr,k,j+1,i,iens) = mult;
 # 540             if (tracers_flux_y(tr,k,j  ,i,iens) < 0) tracers_mult_y(tr,k,j  ,i,iens) = mult;
 # 541             if (tracers_flux_z(tr,k+1,j,i,iens) > 0) tracers_mult_z(tr,k+1,j,i,iens) = mult;
 # 542             if (tracers_flux_z(tr,k  ,j,i,iens) < 0) tracers_mult_z(tr,k  ,j,i,iens) = mult;
 # 543           }
 # 544         }
 # 545       });
 # 546       // Make sure domain edge fluxes are replaced by the minimum among neighboring tasks
 # 547       // Probably want to do lower bound FCT and upper bound FCT seperately
