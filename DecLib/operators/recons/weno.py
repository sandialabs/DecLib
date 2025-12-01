from numba import njit, prange
from math import copysign

#THIS ALL LEANS HEAVILY ON UNIVERSAL BASIS FOR BVDFs...

#really here for weno you build a left and right recon, and then do some sort of upwinding (maybe tanh blended)

#THIS MAKES SIGNIFICANT ASSUMPTIONS ABOUT DIRECTION OF EDGE FLUX IN COMPUTING UPWIND PARAMETER!
#Probably okay on uniform grids, less okay on other grids...

@njit(parallel=True, cache=True)
def _apply_vform_recon_weno1(vformarr, reconarr, velocity, nedges, ndofs, bundle_size, CE, cellareas):
    for e in prange(nedges):
        #compute upwind parameter
        eflux = velocity[e]
        upwind_param = copysign(1.0, eflux)
        #print(e, CE[e,0], CE[e,1], upwind_param)
        for l in range(ndofs):
            for d in range(bundle_size):
                left_recon = vformarr[CE[e,1],l,d]/cellareas[CE[e,1]]
                right_recon = vformarr[CE[e,0],l,d]/cellareas[CE[e,0]]
                reconarr[e,l,d] = 0.5 * (left_recon * (1. - upwind_param) + right_recon * (1. + upwind_param))

@njit(parallel=True, cache=True)
def _apply_vform_recon_weno1_scaled(vformarr, reconarr, velocity, scalearr, nedges, scaledoff, ndofs, bundle_size, CE, cellareas):
    for e in prange(nedges):
        scaleval = (scalearr[CE[e,0],scaledoff]/cellareas[CE[e,0]] + scalearr[CE[e,1],scaledoff]/cellareas[CE[e,1]])/2.
        #compute upwind parameter
        eflux = velocity[e]
        upwind_param = copysign(1.0, eflux)
        #print(e, CE[e,0], CE[e,1], upwind_param)
        for l in range(ndofs):
            for d in range(bundle_size):
                left_recon = vformarr[CE[e,1],l,d]/cellareas[CE[e,1]]
                right_recon = vformarr[CE[e,0],l,d]/cellareas[CE[e,0]]
                reconarr[e,l,d] = 0.5 * (left_recon * (1. - upwind_param) + right_recon * (1. + upwind_param)) / scaleval
@njit
def interp_weno3(phim1, phi, phip1):
    p0 = (-1.0 / 2.0) * phim1 + (3.0 / 2.0) * phi
    p1 = (1.0 / 2.0) * phi + (1.0 / 2.0) * phip1

    beta1 = (phip1 - phi) * (phip1 - phi)
    beta0 = (phi - phim1) * (phi - phim1)

    alpha0 = (1.0 / 3.0) / ((beta0 + 1e-10) * (beta0 + 1.0e-10))
    alpha1 = (2.0 / 3.0) / ((beta1 + 1e-10) * (beta1 + 1.0e-10))

    alpha_sum_inv = 1.0 / (alpha0 + alpha1)

    w0 = alpha0 * alpha_sum_inv
    w1 = alpha1 * alpha_sum_inv

    return w0 * p0 + w1 * p1

@njit
def interp_weno5( phim2,  phim1,  phi,  phip1,  phip2):

    p0 = (1.0 / 3.0) * phim2 - (7.0 / 6.0) * phim1 + (11.0 / 6.0) * phi
    p1 = (-1.0 / 6.0) * phim1 + (5.0 / 6.0) * phi + (1.0 / 3.0) * phip1
    p2 = (1.0 / 3.0) * phi + (5.0 / 6.0) * phip1 - (1.0 / 6.0) * phip2

    beta2 = (13.0 / 12.0 * (phi - 2.0 * phip1 + phip2) * (phi - 2.0 * phip1 + phip2) + 0.25 * (3.0 * phi - 4.0 * phip1 + phip2) * (3.0 * phi - 4.0 * phip1 + phip2))
    beta1 = (13.0 / 12.0 * (phim1 - 2.0 * phi + phip1) * (phim1 - 2.0 * phi + phip1) + 0.25 * (phim1 - phip1) * (phim1 - phip1))
    beta0 = (13.0 / 12.0 * (phim2 - 2.0 * phim1 + phi) * (phim2 - 2.0 * phim1 + phi) + 0.25 * (phim2 - 4.0 * phim1 + 3.0 * phi) * (phim2 - 4.0 * phim1 + 3.0 * phi))

    alpha0 = 0.1 / ((beta0 + 1e-10) * (beta0 + 1e-10))
    alpha1 = 0.6 / ((beta1 + 1e-10) * (beta1 + 1e-10))
    alpha2 = 0.3 / ((beta2 + 1e-10) * (beta2 + 1e-10))

    alpha_sum_inv = 1.0 / (alpha0 + alpha1 + alpha2)

    w0 = alpha0 * alpha_sum_inv
    w1 = alpha1 * alpha_sum_inv
    w2 = alpha2 * alpha_sum_inv

    return w0 * p0 + w1 * p1 + w2 * p2

@njit
def interp_weno7( phim3,  phim2,  phim1,  phi,  phip1,  phip2,  phip3): 
                            
    p0 = (-1.0 / 4.0) * phim3 + (13.0 / 12.0) * phim2 + (-23.0 / 12.0) * phim1 + (25.0 / 12.0) * phi
    p1 = (1.0 / 12.0) * phim2 + (-5.0 / 12.0) * phim1 + (13.0 / 12.0) * phi + (1.0 / 4.0) * phip1
    p2 = (-1.0 / 12.0) * phim1 + (7.0 / 12.0) * phi + (7.0 / 12.0) * phip1 + (-1.0 / 12.0) * phip2
    p3 = (1.0 / 4.0) * phi + (13.0 / 12.0) * phip1 + (-5.0 / 12.0) * phip2 + (1.0 / 12.0) * phip3

    beta0 = (phim3 * (547.0 * phim3 - 3882.0 * phim2 + 4642.0 * phim1 - 1854.0 * phi) + phim2 * (7043.0 * phim2 - 17246.0 * phim1 + 7042.0 * phi) + phim1 * (11003.0 * phim1 - 9402.0 * phi) + 2107.0 * phi * phi)
    beta1 = (phim2 * (267.0 * phim2 - 1642.0 * phim1 + 1602.0 * phi - 494.0 * phip1) + phim1 * (2843.0 * phim1 - 5966.0 * phi + 1922.0 * phip1) + phi * (3443.0 * phi - 2522.0 * phip1) + 547.0 * phip1 * phip1)
    beta2 = (phim1 * (547.0 * phim1 - 2522.0 * phi + 1922.0 * phip1 - 494.0 * phip2) + phi * (3443.0 * phi - 5966.0 * phip1 + 1602.0 * phip2) + phip1 * (2843.0 * phip1 - 1642.0 * phip2) + 267.0 * phip2 * phip2)
    beta3 = (phi * (2107.0 * phi - 9402.0 * phip1 + 7042.0 * phip2 - 1854.0 * phip3) + phip1 * (11003.0 * phip1 - 17246.0 * phip2 + 4642.0 * phip3) + phip2 * (7043.0 * phip2 - 3882.0 * phip3) + 547.0 * phip3 * phip3)

    alpha0 = (1.0 / 35.0) / ((beta0 + 1e-10) * (beta0 + 1e-10))
    alpha1 = (12.0 / 35.0) / ((beta1 + 1e-10) * (beta1 + 1e-10))
    alpha2 = (18.0 / 35.0) / ((beta2 + 1e-10) * (beta2 + 1e-10))
    alpha3 = (4.0 / 35.0) / ((beta3 + 1e-10) * (beta3 + 1e-10))

    alpha_sum_inv = 1.0 / (alpha0 + alpha1 + alpha2 + alpha3)

    w0 = alpha0 * alpha_sum_inv
    w1 = alpha1 * alpha_sum_inv
    w2 = alpha2 * alpha_sum_inv
    w3 = alpha3 * alpha_sum_inv

    return w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3


@njit(parallel=True, cache=True)
def _apply_vform_recon_weno3(vformarr, reconarr, velocity, nedges, ndofs, bundle_size, CE, cellareas):
    for e in prange(nedges):
        #compute upwind parameter
        eflux = velocity[e]
        upwind_param = copysign(1.0, eflux)
        for l in range(ndofs):
            for d in range(bundle_size):
                left_recon = interp_weno3(vformarr[CE[e,3],l,d]/cellareas[CE[e,3]], vformarr[CE[e,2],l,d]/cellareas[CE[e,2]], vformarr[CE[e,1],l,d]/cellareas[CE[e,1]])
                right_recon = interp_weno3(vformarr[CE[e,0],l,d]/cellareas[CE[e,0]], vformarr[CE[e,1],l,d]/cellareas[CE[e,1]], vformarr[CE[e,2],l,d]/cellareas[CE[e,2]])
                reconarr[e,l,d] = 0.5 * (left_recon * (1. - upwind_param) + right_recon * (1. + upwind_param))


@njit(parallel=True, cache=True)
def _apply_vform_recon_weno3_scaled(vformarr, reconarr, velocity, scalearr, nedges, scaledoff, ndofs, bundle_size, CE, cellareas):
    for e in prange(nedges):
        #compute upwind parameter
        eflux = velocity[e]
        upwind_param = copysign(1.0, eflux)
        scaleval = (scalearr[CE[e,1],scaledoff]/cellareas[CE[e,1]] + scalearr[CE[e,2],scaledoff]/cellareas[CE[e,2]])/2.
        for l in range(ndofs):
            for d in range(bundle_size):
                left_recon = interp_weno3(vformarr[CE[e,3],l,d]/cellareas[CE[e,3]], vformarr[CE[e,2],l,d]/cellareas[CE[e,2]], vformarr[CE[e,1],l,d]/cellareas[CE[e,1]])
                right_recon = interp_weno3(vformarr[CE[e,0],l,d]/cellareas[CE[e,0]], vformarr[CE[e,1],l,d]/cellareas[CE[e,1]], vformarr[CE[e,2],l,d]/cellareas[CE[e,2]])
                reconarr[e,l,d] = 0.5 * (left_recon * (1. - upwind_param) + right_recon * (1. + upwind_param)) / scaleval

@njit(parallel=True, cache=True)
def _apply_vform_recon_weno5(vformarr, reconarr, velocity, nedges, ndofs, bundle_size, CE, cellareas):
    for e in prange(nedges):
        #compute upwind parameter
        eflux = velocity[e]
        upwind_param = copysign(1.0, eflux)
        for l in range(ndofs):
            for d in range(bundle_size):
                left_recon = interp_weno5(vformarr[CE[e,5],l,d]/cellareas[CE[e,5]], vformarr[CE[e,4],l,d]/cellareas[CE[e,4]], vformarr[CE[e,3],l,d]/cellareas[CE[e,3]], vformarr[CE[e,2],l,d]/cellareas[CE[e,2]], vformarr[CE[e,1],l,d]/cellareas[CE[e,1]])
                right_recon = interp_weno5(vformarr[CE[e,0],l,d]/cellareas[CE[e,0]], vformarr[CE[e,1],l,d]/cellareas[CE[e,1]], vformarr[CE[e,2],l,d]/cellareas[CE[e,2]], vformarr[CE[e,3],l,d]/cellareas[CE[e,3]], vformarr[CE[e,4],l,d]/cellareas[CE[e,4]])
                reconarr[e,l,d] = 0.5 * (left_recon * (1. - upwind_param) + right_recon * (1. + upwind_param))

@njit(parallel=True, cache=True)
def _apply_vform_recon_weno5_scaled(vformarr, reconarr, velocity, scalearr, nedges, scaledoff, ndofs, bundle_size, CE, cellareas):
    for e in prange(nedges):
        #compute upwind parameter
        eflux = velocity[e]
        upwind_param = copysign(1.0, eflux)
        scaleval = (scalearr[CE[e,2],scaledoff]/cellareas[CE[e,2]] + scalearr[CE[e,3],scaledoff]/cellareas[CE[e,3]])/2.
        for l in range(ndofs):
            for d in range(bundle_size):
                left_recon = interp_weno5(vformarr[CE[e,5],l,d]/cellareas[CE[e,5]], vformarr[CE[e,4],l,d]/cellareas[CE[e,4]], vformarr[CE[e,3],l,d]/cellareas[CE[e,3]], vformarr[CE[e,2],l,d]/cellareas[CE[e,2]], vformarr[CE[e,1],l,d]/cellareas[CE[e,1]])
                right_recon = interp_weno5(vformarr[CE[e,0],l,d]/cellareas[CE[e,0]], vformarr[CE[e,1],l,d]/cellareas[CE[e,1]], vformarr[CE[e,2],l,d]/cellareas[CE[e,2]], vformarr[CE[e,3],l,d]/cellareas[CE[e,3]], vformarr[CE[e,4],l,d]/cellareas[CE[e,4]])
                reconarr[e,l,d] = 0.5 * (left_recon * (1. - upwind_param) + right_recon * (1. + upwind_param)) / scaleval

@njit(parallel=True, cache=True)
def _apply_vform_recon_weno7(vformarr, reconarr, velocity, nedges, ndofs, bundle_size, CE, cellareas):
    for e in prange(nedges):
        #compute upwind parameter
        eflux = velocity[e]
        upwind_param = copysign(1.0, eflux)
        for l in range(ndofs):
            for d in range(bundle_size):
                left_recon = interp_weno7(vformarr[CE[e,7],l,d]/cellareas[CE[e,7]], vformarr[CE[e,6],l,d]/cellareas[CE[e,6]], vformarr[CE[e,5],l,d]/cellareas[CE[e,5]], vformarr[CE[e,4],l,d]/cellareas[CE[e,4]], vformarr[CE[e,3],l,d]/cellareas[CE[e,3]], vformarr[CE[e,2],l,d]/cellareas[CE[e,2]], vformarr[CE[e,1],l,d]/cellareas[CE[e,1]])
                #right_recon = interp_weno7(vformarr[CE[e,6],l,d]/cellareas[CE[e,6]], vformarr[CE[e,5],l,d]/cellareas[CE[e,5]], vformarr[CE[e,4],l,d]/cellareas[CE[e,4]], vformarr[CE[e,3],l,d]/cellareas[CE[e,3]], vformarr[CE[e,2],l,d]/cellareas[CE[e,2]], vformarr[CE[e,1],l,d]/cellareas[CE[e,1]], vformarr[CE[e,0],l,d]/cellareas[CE[e,0]])
                #right_recon =  interp_weno7(vformarr[CE[e,0],l,d]/cellareas[CE[e,0]], vformarr[CE[e,1],l,d]/cellareas[CE[e,1]], vformarr[CE[e,2],l,d]/cellareas[CE[e,2]], vformarr[CE[e,3],l,d]/cellareas[CE[e,3]], vformarr[CE[e,4],l,d]/cellareas[CE[e,4]], vformarr[CE[e,5],l,d]/cellareas[CE[e,5]], vformarr[CE[e,6],l,d]/cellareas[CE[e,6]])
                #left_recon =  interp_weno7(vformarr[CE[e,1],l,d]/cellareas[CE[e,1]], vformarr[CE[e,2],l,d]/cellareas[CE[e,2]], vformarr[CE[e,3],l,d]/cellareas[CE[e,3]], vformarr[CE[e,4],l,d]/cellareas[CE[e,4]], vformarr[CE[e,5],l,d]/cellareas[CE[e,5]], vformarr[CE[e,6],l,d]/cellareas[CE[e,6]], vformarr[CE[e,7],l,d]/cellareas[CE[e,7]])
                right_recon =  interp_weno7(vformarr[CE[e,0],l,d]/cellareas[CE[e,0]], vformarr[CE[e,1],l,d]/cellareas[CE[e,1]], vformarr[CE[e,2],l,d]/cellareas[CE[e,2]], vformarr[CE[e,3],l,d]/cellareas[CE[e,3]], vformarr[CE[e,4],l,d]/cellareas[CE[e,4]], vformarr[CE[e,5],l,d]/cellareas[CE[e,5]], vformarr[CE[e,6],l,d]/cellareas[CE[e,6]])
                reconarr[e,l,d] = 0.5 * (left_recon * (1. - upwind_param) + right_recon * (1. + upwind_param))

@njit(parallel=True, cache=True)
def _apply_vform_recon_weno7_scaled(vformarr, reconarr, velocity, scalearr, nedges, scaledoff, ndofs, bundle_size, CE, cellareas):
    for e in prange(nedges):
        #compute upwind parameter
        eflux = velocity[e]
        upwind_param = copysign(1.0, eflux)
        scaleval = (scalearr[CE[e,3],scaledoff]/cellareas[CE[e,3]] + scalearr[CE[e,4],scaledoff]/cellareas[CE[e,4]])/2.
        for l in range(ndofs):
            for d in range(bundle_size):
                left_recon = interp_weno7(vformarr[CE[e,7],l,d]/cellareas[CE[e,7]], vformarr[CE[e,6],l,d]/cellareas[CE[e,6]], vformarr[CE[e,5],l,d]/cellareas[CE[e,5]], vformarr[CE[e,4],l,d]/cellareas[CE[e,4]], vformarr[CE[e,3],l,d]/cellareas[CE[e,3]], vformarr[CE[e,2],l,d]/cellareas[CE[e,2]], vformarr[CE[e,1],l,d]/cellareas[CE[e,1]])
                right_recon =  interp_weno7(vformarr[CE[e,0],l,d]/cellareas[CE[e,0]], vformarr[CE[e,1],l,d]/cellareas[CE[e,1]], vformarr[CE[e,2],l,d]/cellareas[CE[e,2]], vformarr[CE[e,3],l,d]/cellareas[CE[e,3]], vformarr[CE[e,4],l,d]/cellareas[CE[e,4]], vformarr[CE[e,5],l,d]/cellareas[CE[e,5]], vformarr[CE[e,6],l,d]/cellareas[CE[e,6]])
                reconarr[e,l,d] = 0.5 * (left_recon * (1. - upwind_param) + right_recon * (1. + upwind_param)) / scaleval
                
