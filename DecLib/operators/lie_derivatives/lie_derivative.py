from DecLib.operators.operators import BinaryOperator
from DecLib.common import ADD_MODE, INSERT_MODE
from DecLib.operators.ext_deriv import ExtDeriv

from numba import njit, prange
import numpy as np

class InteriorProductMLP():
    def __init__(self, meshes):
        self.meshes = meshes
        self.dim = meshes.dim
        self.ttopo = meshes.ttopo
        self.stopo = meshes.stopo
        self.stmapping = meshes.stmapping
        self.geom = meshes.dgeom

        self.nedges = self.ttopo.nkcells[self.dim - 1] - self.ttopo.nbkcells[self.dim-1]
        self.np0 = self.stopo.nkcells[0]
        self.CE = np.zeros((self.nedges, 2), dtype=np.int32)

#THIS IS AN UGLY HACK- SHOULD REALLY BE BASED ON A BASIS CHOICE FOR AN EDGE...
#ALSO ASSUMES THAT ALL QUADRATURE POINTS SHARE A BASIS- WHICH IS NOT NECCESARILY TRUE...
        self.edge_normals = np.zeros((self.nedges, self.dim))
        self.edge_normals[:,0] = 1.0
        #self.edge_normals = geom.facenormals.getArray()
        #nquad = self.edge_normals.shape[0]//nedges//topo.dim
        #self.edge_normals = self.edge_normals.reshape((nedges, nquad, topo.dim))
        #self.edge_normals = self.edge_normals[:,0,:]

    #JIT THIS PART ALSO
        deoff = self.ttopo.kcells_off[self.dim - 1]
        p0off = self.stopo.kcells_off[0]

        for de in range(self.ttopo.kcells[self.dim-1][0], self.ttopo.kcells[self.dim-1][1] - self.ttopo.nbkcells[self.dim-1]):
            pe = meshes.stmapping.dinmk_to_pk(de)
            pverts = meshes.stopo.lower_dim_TC(int(pe), 0)
            self.CE[de - deoff] = pverts - p0off

    def apply(self, velocity, flux):

        velocityarr = velocity.petsc_vec.getArray()
        velocityarr = velocityarr.reshape((self.np0, self.dim))
        fluxarr = flux.petsc_vec.getArray()
        fluxarr = fluxarr.reshape((flux.nelems, self.dim))

        _apply_interior_product_mlp(velocityarr, fluxarr, self.nedges, self.dim, self.CE, self.edge_normals)
        flux.petsc_vec.assemble()
#BE CAREFUL WITH GENERATION HERE- DON'T FILL UFLUX AT EDGES!

@njit(parallel=True, cache=True)
def _apply_interior_product_mlp(velocityarr, fluxarr, nedges, dim, CE, edge_normals):
    for e in prange(nedges):
        #THIS MAKES GROSS ASSUMPTIONS ABOUT A UNIVERSAL BASIS FOR COTANGENT SPACES, AND ALSO HOW THE DOT PRODUCT WORKS, ETC.
        edge_velocity = 0
        for d in range(dim):
            edge_velocity += (velocityarr[CE[e,0],d] + velocityarr[CE[e,1],d])/2. * edge_normals[e,d]
        fluxarr[e] = edge_velocity


#THIS LIE DERIVATIVE CAN BE ELIMINATED SINCE WE HAVE A UFLUX!!!
#THE CORRESPONDING DIAMOND, HOWEVER, CANNOT

#Does this pattern hold for all Lie derivatives ie compute UFLUX, use v1 Lie derivative? Unclear...



# class LieDerivativeVForm_MLP():
    # def __init__(self, meshes):
        # self.meshes = meshes
        # self.dim = meshes.dim
        # self.ttopo = meshes.ttopo
        # self.stopo = meshes.stopo
        # self.stmapping = meshes.stmapping
        # self.geom = meshes.dgeom
		# #self.Dbar = ExtDeriv(dim, topo, bundle)

        # self.ncells = self.ttopo.nkcells[self.dim]
        # self.nedges = self.ttopo.nkcells[self.dim - 1]
        # self.maxne = self.ttopo.petscmesh.getMaxSizes()[0]
        # self.np0 = self.stopo.nkcells[0]

        # self.EC = np.zeros((self.ncells, self.maxne), dtype=np.int32)
        # self.nEC = np.zeros(self.ncells, dtype=np.int32)
        # self.orientation = np.zeros((self.ncells, self.maxne), dtype=np.int32)

        # self.CE = np.zeros((self.nedges, 2), dtype=np.int32)
        # self.bnd = np.full(self.nedges, False, dtype=np.bool_)

# #THIS IS AN UGLY HACK- SHOULD REALLY BE BASED ON A BASIS CHOICE FOR AN EDGE...
# #ALSO ASSUMES THAT ALL QUADRATURE POINTS SHARE A BASIS- WHICH IS NOT NECCESARILY TRUE...
        # self.edge_normals = np.zeros((self.nedges, self.dim))
        # self.edge_normals[:,0] = 1.0
        # #self.edge_normals = geom.facenormals.getArray()
        # #nquad = self.edge_normals.shape[0]//nedges//topo.dim
        # #self.edge_normals = self.edge_normals.reshape((nedges, nquad, topo.dim))
        # #self.edge_normals = self.edge_normals[:,0,:]

    # #JIT THIS PART ALSO
        # dnoff = self.ttopo.kcells_off[self.dim]
        # deoff = self.ttopo.kcells_off[self.dim - 1]
        # p0off = self.stopo.kcells_off[0]

        # for dn in range(self.ttopo.kcells[self.dim][0], self.ttopo.kcells[self.dim][1]):
            # edges = self.ttopo.lower_dim_TC(dn, self.dim-1)
            # nedges = edges.shape[0]
            # orients = self.ttopo.lower_orientation(dn)
            # self.EC[dn - dnoff, :nedges] = edges - deoff
            # self.nEC[dn - dnoff] = nedges
            # self.orientation[dn - dnoff, :nedges] = orients

        # for de in range(self.ttopo.kcells[self.dim-1][0], self.ttopo.kcells[self.dim-1][1]):
            # pe = meshes.stmapping.dinmk_to_pk(de)
            # pverts = meshes.stopo.lower_dim_TC(int(pe), 0)
            # nCE = pverts.shape[0]
            # self.CE[de - deoff, :nCE] = pverts - p0off

        # if meshes.ttopo.has_boundary:
            # for de in range(meshes.ttopo.kcells[self.dim-1][0], meshes.ttopo.kcells[self.dim-1][1]):
                # self.bnd[de - deoff] = meshes.ttopo.petscmesh.getLabelValue('bnd', de)

    # def apply(self, recon, velocity, target, ubnd, mode=INSERT_MODE):
        # reconarr = recon.petsc_vec.getArray()
        # reconarr = reconarr.reshape((self.nedges, recon.ndofs))

        # velocityarr = velocity.petsc_vec.getArray()
        # velocityarr = velocityarr.reshape((self.np0, self.dim))
        # targetarr = target.petsc_vec.getArray()
        # targetarr = targetarr.reshape((self.ncells, target.ndofs))
        # ubndarr = ubnd.petsc_vec.getArray()
        # ubndarr = ubndarr.reshape((self.nedges, self.dim))
        # if mode == INSERT_MODE:
            # targetarr[:,:] = 0.0
        # _apply_lie_deriv_vform_mlp(reconarr, velocityarr, targetarr, ubndarr, target.ndofs, self.ncells, self.dim, self.EC, self.nEC, self.CE, self.edge_normals, self.orientation, self.bnd)
        # target.petsc_vec.assemble()

# @njit(parallel=True)
# def _apply_lie_deriv_vform_mlp(reconarr, velocityarr, targetarr, ubndarr, ndofs, ncells, dim, EC, nEC, CE, edge_normals, orientation, bnd):

# #loop over cells
# #loop over EC
# #compute edge_velocity
# #compute target += reconval * I * velocity

    # for i in prange(ncells):
        # nedges = nEC[i]
        # for j in range(nedges):
            # e = EC[i,j]
            # #THIS MAKES GROSS ASSUMPTIONS ABOUT A UNIVERSAL BASIS FOR COTANGENT SPACES, AND ALSO HOW THE DOT PRODUCT WORKS, ETC.

            # edge_velocity = 0
            # if not bnd[e]:
                # for d in range(dim):
                    # edge_velocity += (velocityarr[CE[e,0],d] + velocityarr[CE[e,1],d])/2. * edge_normals[e,d]
            # else:
                # for d in range(dim):
                    # edge_velocity += ubndarr[e,d] * edge_normals[e,d]
            # for l in range(ndofs):
                # targetarr[i,l] += reconarr[e,l] * edge_velocity * orientation[i,j]




class LieDerivativeM():
    def __init__(self, meshes):
        self.meshes = meshes
        self.dim = meshes.dim

        self.ncells = meshes.ttopo.nkcells[self.dim]
        self.nedges = meshes.ttopo.nkcells[self.dim - 1]
        self.np0 = meshes.stopo.nkcells[0]
        self.maxne = meshes.ttopo.petscmesh.getMaxSizes()[0]

        self.EC = np.zeros((self.ncells, self.maxne), dtype=np.int32)
        self.nEC = np.zeros(self.ncells, dtype=np.int32)
        self.cellorients = np.zeros((self.ncells, self.maxne), dtype=np.int32)
        self.edgeorients = np.zeros((self.nedges, 2), dtype=np.int32)
        self.CE = np.zeros((self.nedges, 2), dtype=np.int32)
        self.bnd = np.full(self.nedges, False, dtype=np.bool_)

#THIS IS AN UGLY HACK- SHOULD REALLY BE BASED ON A BASIS CHOICE FOR AN EDGE...
#ALSO ASSUMES THAT ALL QUADRATURE POINTS SHARE A BASIS- WHICH IS NOT NECCESARILY TRUE...
        self.edge_normals = np.zeros((self.nedges, self.dim))
        self.edge_normals[:,0] = 1.0
        #self.edge_normals = geom.facenormals.getArray()
        #nquad = self.edge_normals.shape[0]//nedges//topo.dim
        #self.edge_normals = self.edge_normals.reshape((nedges, nquad, topo.dim))
        #self.edge_normals = self.edge_normals[:,0,:]

    #JIT THIS PART ALSO
        dnoff = meshes.ttopo.kcells_off[self.dim]
        deoff = meshes.ttopo.kcells_off[self.dim - 1]
        p0off = meshes.stopo.kcells_off[0]

        for dn in range(meshes.ttopo.kcells[self.dim][0], meshes.ttopo.kcells[self.dim][1]):
            edges = meshes.ttopo.lower_dim_TC(dn, self.dim-1)
            nedges = edges.shape[0]
            self.EC[dn - dnoff, :nedges] = edges - deoff
            self.nEC[dn - dnoff] = nedges
            self.cellorients[dn - dnoff, :nedges] = meshes.ttopo.lower_orientation(dn)

        for de in range(meshes.ttopo.kcells[self.dim-1][0], meshes.ttopo.kcells[self.dim-1][1] - meshes.ttopo.nbkcells[self.dim-1]):
            pe = meshes.stmapping.dinmk_to_pk(de)
            pverts = meshes.stopo.lower_dim_TC(int(pe), 0)
            nCE = pverts.shape[0]
            self.CE[de - deoff, :nCE] = pverts - p0off
            self.edgeorients[de - deoff, :nCE] = meshes.stopo.lower_orientation(pe)
        if meshes.ttopo.has_boundary:
            for de in range(meshes.ttopo.kcells[self.dim-1][0], meshes.ttopo.kcells[self.dim-1][1]):
                self.bnd[de - deoff] = (meshes.ttopo.petscmesh.getLabelValue('bnd', de) == 1)

    def apply(self, recon, velocity, uflux, target, mode=INSERT_MODE):
        reconarr = recon.petsc_vec.getArray()
        reconarr = reconarr.reshape((self.nedges, self.dim))
        velocityarr = velocity.petsc_vec.getArray()
        velocityarr = velocityarr.reshape((self.np0, self.dim))
        targetarr = target.petsc_vec.getArray()
        targetarr = targetarr.reshape((self.ncells, self.dim))
        ufluxarr = uflux.petsc_vec.getArray()
        if mode == INSERT_MODE:
            targetarr[:,:] = 0.0
        _apply_lie_deriv_m(reconarr, velocityarr, ufluxarr, targetarr, self.ncells, self.dim, self.EC, self.nEC, self.CE, self.edge_normals, self.cellorients, self.edgeorients, self.bnd)
        target.petsc_vec.assemble()


@njit(parallel=True, cache=True)
def _apply_lie_deriv_m(reconarr, velocityarr, uflux, targetarr, ncells, dim, EC, nEC, CE, edge_normals, cellorients, edgeorients, bnd):
    for i in prange(ncells):
        nedges = nEC[i]
        for j in range(nedges):
            e = EC[i,j]

            # Dbarnm1 me I u
            #THIS MAKES GROSS ASSUMPTIONS ABOUT A UNIVERSAL BASIS FOR COTANGENT SPACES, AND ALSO HOW THE DOT PRODUCT WORKS, ETC.

            #edge_velocity = 0

            #if not bnd[e]:
            #    for d in range(dim):
            #        edge_velocity += (velocityarr[CE[e,0],d] + velocityarr[CE[e,1],d])/2. * edge_normals[e,d]
            #else:
            #    for d in range(dim):
            #        edge_velocity += ubnd[e,d] * edge_normals[e,d]

            for d1 in range(dim):
                targetarr[i,d1] += reconarr[e,d1] * uflux[e] * cellorients[i,j]

			#I^T me D1 u

            #THIS MAKES GROSS ASSUMPTIONS ABOUT A UNIVERSAL BASIS FOR COTANGENT SPACES, AND ALSO HOW THE DOT PRODUCT WORKS, ETC.

            eflux = 0
            if not bnd[e]:
                for d1 in range(dim):
                    eflux += reconarr[e,d1] * (velocityarr[CE[e,0],d1] * edgeorients[e,0] + velocityarr[CE[e,1], d1] * edgeorients[e,1])
            #else:
            #    for d1 in range(dim):
            #        eflux += reconarr[e,d1] * ubnd[e,d1] * edgeorients[e,0]
            for d in range(dim):
                targetarr[i,d] += eflux * edge_normals[e,d]/2.





class InteriorProductV():
    def __init__(self, meshes):
        self.meshes = meshes
        self.dim = meshes.dim


    def apply(self, recon, flux, target, mode=INSERT_MODE):
        reconarr = recon.petsc_vec.getArray()
        fluxarr = flux.petsc_vec.getArray()
        targetarr = target.petsc_vec.getArray()
        if mode == INSERT_MODE:
            targetarr[:] = 0.0
        #_apply_interior_product_v(reconarr, velocityarr, targetarr, self.ncells, self.dim, self.EC, self.nEC, self.CE, self.edge_normals, self.orientation)
        target.petsc_vec.assemble()


class LieDerivativeVForm():
    def __init__(self, meshes):
        self.meshes = meshes
        self.dim = meshes.dim
        self.ttopo = meshes.ttopo
        self.stopo = meshes.stopo
        self.stmapping = meshes.stmapping
        self.geom = meshes.dgeom

        self.ncells = self.ttopo.nkcells[self.dim]
        self.nedges = self.ttopo.nkcells[self.dim-1]
        self.nedges = self.ttopo.nkcells[self.dim - 1]
        self.maxne = self.ttopo.petscmesh.getMaxSizes()[0]

        self.EC = np.zeros((self.ncells, self.maxne), dtype=np.int32)
        self.nEC = np.zeros(self.ncells, dtype=np.int32)
        self.orientation = np.zeros((self.ncells, self.maxne), dtype=np.int32)

    #JIT THIS PART ALSO
        dnoff = self.ttopo.kcells_off[self.dim]
        deoff = self.ttopo.kcells_off[self.dim - 1]

        for dn in range(self.ttopo.kcells[self.dim][0], self.ttopo.kcells[self.dim][1]):
            edges = self.ttopo.lower_dim_TC(dn, self.dim-1)
            nedges = edges.shape[0]
            orients = self.ttopo.lower_orientation(dn)
            self.EC[dn - dnoff, :nedges] = edges - deoff
            self.nEC[dn - dnoff] = nedges
            self.orientation[dn - dnoff, :nedges] = orients

    def apply(self, recon, flux, target, mode=INSERT_MODE):
        reconarr = recon.petsc_vec.getArray()
        reconarr = reconarr.reshape((self.nedges, recon.ndofs))
        fluxarr = flux.petsc_vec.getArray()
        targetarr = target.petsc_vec.getArray()
        targetarr = targetarr.reshape((self.ncells, target.ndofs))

        if mode == INSERT_MODE:
            targetarr[:,:] = 0.0
        _apply_lie_deriv_vform(reconarr, fluxarr, targetarr, target.ndofs, self.ncells, self.EC, self.nEC, self.orientation)
        target.petsc_vec.assemble()

@njit(parallel=True, cache=True)
def _apply_lie_deriv_vform(reconarr, fluxarr, targetarr, ndofs, ncells, EC, nEC, orientation):

#loop over cells
#loop over EC
#compute target += reconval * I * flux

    for i in prange(ncells):
        nedges = nEC[i]
        for j in range(nedges):
            e = EC[i,j]
            for l in range(ndofs):
                targetarr[i,l] += reconarr[e,l] * fluxarr[e] * orientation[i,j]




class CovariantExteriorDerivativeVForm():
    def __init__(self, meshes):
        self.meshes = meshes
        self.dim = meshes.dim

        self.ncells = meshes.ttopo.nkcells[self.dim]
        self.nedges = meshes.ttopo.nkcells[self.dim - 1]
        self.maxne = meshes.ttopo.petscmesh.getMaxSizes()[0]

        self.EC = np.zeros((self.ncells, self.maxne), dtype=np.int32)
        self.nEC = np.zeros(self.ncells, dtype=np.int32)
        self.cellorients = np.zeros((self.ncells, self.maxne), dtype=np.int32)

    #JIT THIS PART ALSO
        dnoff = meshes.ttopo.kcells_off[self.dim]
        deoff = meshes.ttopo.kcells_off[self.dim - 1]

        for dn in range(meshes.ttopo.kcells[self.dim][0], meshes.ttopo.kcells[self.dim][1]):
            edges = meshes.ttopo.lower_dim_TC(dn, self.dim-1)
            nedges = edges.shape[0]
            self.EC[dn - dnoff, :nedges] = edges - deoff
            self.nEC[dn - dnoff] = nedges
            self.cellorients[dn - dnoff, :nedges] = meshes.ttopo.lower_orientation(dn)


    def apply(self, flux, target, mode=INSERT_MODE):
        fluxarr = flux.petsc_vec.getArray()
        fluxarr = fluxarr.reshape((flux.nelems, flux.bsize))
        targetarr = target.petsc_vec.getArray()
        targetarr = targetarr.reshape((self.ncells, self.dim))

        if mode == INSERT_MODE:
            targetarr[:,:] = 0.0
        _apply_covextderiv(fluxarr, targetarr, self.ncells, self.dim, self.EC, self.nEC, self.cellorients)
        target.petsc_vec.assemble()


@njit(parallel=True, cache=True)
def _apply_covextderiv(flux, target, ncells, dim, EC, nEC, cellorients):
    for i in prange(ncells):
        nedges = nEC[i]
        for j in range(nedges):
            e = EC[i,j]

            #THIS MAKES GROSS ASSUMPTIONS ABOUT A UNIVERSAL BASIS FOR COTANGENT SPACES, AND ALSO HOW THE DOT PRODUCT WORKS, ETC.

            for d1 in range(dim):
                target[i,d1] += flux[e,d1] * cellorients[i,j]

class ExteriorDerivativeVForm():
    def __init__(self, meshes):
        self.meshes = meshes
        self.dim = meshes.dim

        self.ncells = meshes.ttopo.nkcells[self.dim]
        self.nedges = meshes.ttopo.nkcells[self.dim - 1]
        self.maxne = meshes.ttopo.petscmesh.getMaxSizes()[0]

        self.EC = np.zeros((self.ncells, self.maxne), dtype=np.int32)
        self.nEC = np.zeros(self.ncells, dtype=np.int32)
        self.cellorients = np.zeros((self.ncells, self.maxne), dtype=np.int32)

    #JIT THIS PART ALSO
        dnoff = meshes.ttopo.kcells_off[self.dim]
        deoff = meshes.ttopo.kcells_off[self.dim - 1]

        for dn in range(meshes.ttopo.kcells[self.dim][0], meshes.ttopo.kcells[self.dim][1]):
            edges = meshes.ttopo.lower_dim_TC(dn, self.dim-1)
            nedges = edges.shape[0]
            self.EC[dn - dnoff, :nedges] = edges - deoff
            self.nEC[dn - dnoff] = nedges
            self.cellorients[dn - dnoff, :nedges] = meshes.ttopo.lower_orientation(dn)


    def apply(self, flux, target, mode=INSERT_MODE):
        fluxarr = flux.petsc_vec.getArray()
        fluxarr = fluxarr.reshape((flux.nelems, flux.ndofs))
        targetarr = target.petsc_vec.getArray()
        targetarr = targetarr.reshape((self.ncells, flux.ndofs))

        if mode == INSERT_MODE:
            targetarr[:,:] = 0.0
        _apply_extderiv(fluxarr, targetarr, self.ncells, flux.ndofs, self.EC, self.nEC, self.cellorients)
        target.petsc_vec.assemble()


@njit(parallel=True, cache=True)
def _apply_extderiv(flux, target, ncells, ndofs, EC, nEC, cellorients):
    for i in prange(ncells):
        nedges = nEC[i]
        for j in range(nedges):
            e = EC[i,j]
            for l in range(ndofs):
                target[i,l] += flux[e,l] * cellorients[i,j]
