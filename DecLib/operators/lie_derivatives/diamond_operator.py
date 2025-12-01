from DecLib.operators.operators import BinaryOperator
from DecLib.common import ADD_MODE, INSERT_MODE
from DecLib.operators.ext_deriv import ExtDeriv

from numba import njit, prange
import numpy as np

#THIS IS REALLY JUST THE VOLUME FORM DIAMOND...
#NEED TO PUT INTO BINARY OPERATOR FORM EVENTUALLY...
class DiamondVForm_MLP():
    def __init__(self, meshes):
        self.dim = meshes.dim

#LIKELY MOVE ALL THIS STUFF INTO THE MESH OBJECT?
#YES, BASICALLY ALL THIS STENCIL INFO IS WHAT DEFINES THE MESH...
#NEED TO JIT IT ALL TO MAKE IT FAST...

        self.nedges = meshes.ttopo.nkcells[self.dim-1]
        self.ncells = meshes.ttopo.nkcells[self.dim]
        self.nverts = meshes.stopo.nkcells[0]
        self.maxne = meshes.ttopo.petscmesh.getMaxSizes()[0]

        self.EC = np.zeros((self.ncells, self.maxne), dtype=np.int32)
        self.nEC = np.zeros(self.ncells, dtype=np.int32)

        self.CE = np.zeros((self.nedges, 2), dtype=np.int32)
        self.orientation = np.zeros((self.nedges, 2), dtype=np.int32)
        self.nCE = np.zeros(self.nedges, dtype=np.int32)

#THIS IS AN UGLY HACK- SHOULD REALLY BE BASED ON A BASIS CHOICE FOR AN EDGE...
#ALSO ASSUMES THAT ALL QUADRATURE POINTS SHARE A BASIS- WHICH IS NOT NECCESARILY TRUE...
        self.edge_normals = np.zeros((self.nedges, self.dim))
        self.edge_normals[:,0] = 1.0
        #self.edge_normals = geom.facenormals.getArray()
        #nquad = self.edge_normals.shape[0]//nedges//topo.dim
        #self.edge_normals = self.edge_normals.reshape((nedges, nquad, topo.dim))
        #self.edge_normals = self.edge_normals[:,0,:]

        self.bnd = np.full(self.nedges, False, dtype=np.bool_)

        dnoff = meshes.ttopo.kcells_off[self.dim]
        deoff = meshes.ttopo.kcells_off[self.dim - 1]

        for dn in range(meshes.ttopo.kcells[self.dim][0], meshes.ttopo.kcells[self.dim][1]):
            edges = meshes.ttopo.lower_dim_TC(dn, self.dim-1)
            nedges = edges.shape[0]
            self.EC[dn - dnoff, :nedges] = edges - deoff
            self.nEC[dn - dnoff] = nedges

#an alternative is to trim boundary edges in EC above...
        if meshes.ttopo.has_boundary:
            for de in range(meshes.ttopo.kcells[self.dim-1][0], meshes.ttopo.kcells[self.dim-1][1]):
                self.bnd[de - deoff] = (meshes.ttopo.petscmesh.getLabelValue('bnd', de) == 1)
                #print(de, meshes.ttopo.petscmesh.getLabelValue('bnd', de))
        #print(self.bnd)

        p0off = meshes.stopo.kcells_off[0]
        for de in range(meshes.ttopo.kcells[self.dim-1][0], meshes.ttopo.kcells[self.dim-1][1] - meshes.ttopo.nbkcells[self.dim-1]):
            pe = meshes.stmapping.dinmk_to_pk(de)
            pverts = meshes.stopo.lower_dim_TC(int(pe), 0)
            nCE = pverts.shape[0]
            orients = meshes.stopo.lower_orientation(pe)
            self.CE[de - deoff, :nCE] = pverts - p0off
            self.orientation[de - deoff, :nCE] = orients
            self.nCE[de - deoff] = nCE

    def apply(self, recon, dHdx, target, mode=INSERT_MODE):
        reconarr = recon.petsc_vec.getArray()
        reconarr = reconarr.reshape((self.nedges, recon.ndofs))
        dHdxarr = dHdx.petsc_vec.getArray()
        dHdxarr = dHdxarr.reshape((self.nverts, dHdx.ndofs))
        targetarr = target.petsc_vec.getArray()
        targetarr = targetarr.reshape((self.ncells, self.dim))
        if mode == INSERT_MODE:
            targetarr[:,:] = 0.0
        _apply_diamond_vform_mlp(reconarr, dHdxarr, targetarr, recon.ndofs, self.ncells, self.dim, self.EC, self.nEC, self.CE, self.nCE, self.orientation, self.edge_normals, self.bnd)
        target.petsc_vec.assemble()

@njit(parallel=True, cache=True)
def _apply_diamond_vform_mlp(reconarr, dHdxarr, targetarr, ndofs, ncells, dim, EC, nEC, CE, nCE, orients, edge_normals, bnd):
    for i in prange(ncells):
        nedges = nEC[i]
        for j in range(nedges):
            e = EC[i,j]
            if not bnd[e]:
                eflux = 0.0
                for k in range(nCE[e]):
                    for l in range(ndofs):
                        eflux += reconarr[e,l] * dHdxarr[CE[e,k],l] * orients[e,k]
                for d in range(dim):
                    targetarr[i,d] += eflux * edge_normals[e,d]/2.


class DiamondVForm_V():
    def __init__(self, meshes):
        self.dim = meshes.dim

#LIKELY MOVE ALL THIS STUFF INTO THE MESH OBJECT?
#YES, BASICALLY ALL THIS STENCIL INFO IS WHAT DEFINES THE MESH...
#NEED TO JIT IT ALL TO MAKE IT FAST...

        self.nedges = meshes.stopo.nkcells[1]
        self.nverts = meshes.stopo.nkcells[0]
        self.CE = np.zeros((self.nedges, 2), dtype=np.int32)
        self.orientation = np.zeros((self.nedges, 2), dtype=np.int32)
        self.nCE = np.zeros(self.nedges, dtype=np.int32)
        self.de = np.zeros(self.nedges, dtype=np.int32)

        p0off = meshes.stopo.kcells_off[0]
        peoff = meshes.stopo.kcells_off[1]
        deoff = meshes.ttopo.kcells_off[meshes.dim-1]

        for pe in range(meshes.stopo.kcells[1][0], meshes.stopo.kcells[1][1]):
            pverts = meshes.stopo.lower_dim_TC(int(pe), 0)
            orients = meshes.stopo.lower_orientation(pe)
            nCE = pverts.shape[0]
            self.CE[pe - peoff, :nCE] = pverts - p0off
            self.orientation[pe - peoff, :nCE] = orients
            self.nCE[pe - peoff] = nCE
            self.de[pe - peoff] = meshes.stmapping.pk_to_dinmk(pe) - deoff

    def apply(self, recon, dHdx, target, mode=INSERT_MODE):
        reconarr = recon.petsc_vec.getArray()
        reconarr = reconarr.reshape((recon.nelems, recon.ndofs))
        dHdxarr = dHdx.petsc_vec.getArray()
        dHdxarr = dHdxarr.reshape((self.nverts, dHdx.ndofs))
        targetarr = target.petsc_vec.getArray()
        if mode == INSERT_MODE:
            targetarr[:] = 0.0
        _apply_diamond_vform_v(reconarr, dHdxarr, targetarr, recon.ndofs, self.nedges, self.CE, self.nCE, self.de, self.orientation)
        target.petsc_vec.assemble()

@njit(parallel=True, cache=True)
def _apply_diamond_vform_v(reconarr, dHdxarr, targetarr, ndofs, nedges, CE, nCE, de, orients):
    for e in prange(nedges):
        for j in range(nCE[e]):
            for l in range(ndofs):
                targetarr[e] += reconarr[de[e],l] * dHdxarr[CE[e,j],l] * orients[e,j]
