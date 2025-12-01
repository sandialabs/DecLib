from DecLib.operators.operators import UnaryOperator
from DecLib.common import ADD_MODE

import numpy as np

from DecLib.operators.recons.weno import _apply_vform_recon_weno1, _apply_vform_recon_weno1_scaled
from DecLib.operators.recons.weno import _apply_vform_recon_weno3, _apply_vform_recon_weno3_scaled
from DecLib.operators.recons.weno import _apply_vform_recon_weno5, _apply_vform_recon_weno5_scaled
from DecLib.operators.recons.weno import _apply_vform_recon_weno7, _apply_vform_recon_weno7_scaled
from DecLib.operators.recons.cfv import _apply_vform_recon_cfv2, _apply_vform_recon_cfv2_scaled
from DecLib.operators.recons.cfv import _apply_vform_recon_cfv4, _apply_vform_recon_cfv4_scaled
from DecLib.operators.recons.cfv import _apply_vform_recon_cfv6, _apply_vform_recon_cfv6_scaled
from DecLib.operators.recons.fct import _fct_pd, _fct_bp

#NEED TO PUT INTO UNARY OPERATOR FORM EVENTUALLY...
#ALTHOUGH IT IS NOT REALLY A MATRIX, SO...

#RELIES ON PRESCRIBED BOUNDARY STUFF LIVING AT THE END OF KCELLS
#THIS IS REASONABLE, I THINK
class VolumeFormRecon():
    def __init__(self, topo, geom, recontype='cfv', reconorder=2, tanh_coeff=-1):
        self.topo = topo
        self.geom = geom
        
        self.ncells = topo.nkcells[topo.dim]
        self.nedges = topo.nkcells[topo.dim - 1]
        self.nbedges = topo.nbkcells[topo.dim-1]

        dnoff = topo.kcells_off[topo.dim]
        
        if recontype=='cfv': stencilsize = reconorder
        if recontype=='weno': stencilsize = reconorder+1
        
        self.CE = np.zeros((self.nedges, stencilsize), dtype=np.int32)

        if stencilsize == 2:
            for de in range(topo.kcells[topo.dim-1][0], topo.kcells[topo.dim-1][1] - topo.nbkcells[topo.dim-1]):
                cells = topo.higher_dim_TC(de, topo.dim)
                self.CE[de] = cells - dnoff
        
        if stencilsize >=4:
            for de in range(topo.kcells[topo.dim-1][0], topo.kcells[topo.dim-1][1] - topo.nbkcells[topo.dim-1]):
#THIS NEEDS TO BE MODIFIED FOR MULTI-DIMENSIONAL CASE
                start_cell = de - topo.kcells_off[topo.dim-1] - stencilsize//2
                for j in range(stencilsize):
                    cell = start_cell + j + 1
                    #this is a mirroring BC
                    if cell <0 and topo.has_boundary:
                        cell = -cell - 1
                    elif cell > topo.nkcells[topo.dim]-1 and topo.has_boundary:
                        cell = 2 * topo.nkcells[topo.dim] - cell - 1
                    #this is periodic BC
                    elif cell <0 and not topo.has_boundary:
                        cell = cell + topo.nkcells[topo.dim]
                    elif cell > topo.nkcells[topo.dim]-1 and not topo.has_boundary:
                        cell = cell - topo.nkcells[topo.dim]
                    self.CE[de,j] = cell
                        
        if recontype == 'cfv':
            if reconorder == 2:
                self._recon = _apply_vform_recon_cfv2
                self._scaled_recon = _apply_vform_recon_cfv2_scaled

#CFV4+ is uniform square grid specific
            if reconorder==4:
                self._recon = _apply_vform_recon_cfv4
                self._scaled_recon = _apply_vform_recon_cfv4_scaled

            if reconorder==6:
                self._recon = _apply_vform_recon_cfv6
                self._scaled_recon = _apply_vform_recon_cfv6_scaled

            #ADD MORE 1D CFV HERE
        
        elif recontype == 'weno':

            if reconorder==1:
                self._recon = _apply_vform_recon_weno1
                self._scaled_recon = _apply_vform_recon_weno1_scaled

#WENO3+ is uniform square grid specific
            if reconorder==3:
                self._recon = _apply_vform_recon_weno3
                self._scaled_recon = _apply_vform_recon_weno3_scaled
            if reconorder==5:
                self._recon = _apply_vform_recon_weno5
                self._scaled_recon = _apply_vform_recon_weno5_scaled
            if reconorder==7:
                self._recon = _apply_vform_recon_weno7
                self._scaled_recon = _apply_vform_recon_weno7_scaled
                
        self.cellareas = np.zeros(self.ncells)
        for i in range(self.ncells):
            self.cellareas[i] = self.geom.get_entity_size(topo.dim, i)
        
    def apply(self, vform, recon, velocity, scale=None, scaledof=None):
        vformarr = vform.petsc_vec.getArray()
        vformarr = vformarr.reshape((self.ncells, vform.ndofs, vform.bsize))
        reconarr = recon.petsc_vec.getArray()
        reconarr = reconarr.reshape((self.nedges, recon.ndofs, recon.bsize))
        
        velocityarr = velocity.petsc_vec.getArray()

        if scale is None:
            self._recon(vformarr, reconarr, velocityarr, self.nedges - self.nbedges, vform.ndofs, vform.bsize, self.CE, self.cellareas)
        else:
            scalearr = scale.petsc_vec.getArray()
            scalearr = scalearr.reshape((self.ncells, scale.ndofs))
            self._scaled_recon(vformarr, reconarr, velocityarr, scalearr, self.nedges - self.nbedges, scaledof, vform.ndofs, vform.bsize, self.CE, self.cellareas)
        
        recon.petsc_vec.assemble()


#at least initially this can be a simple approach, eventually do more clever things?
#ultimately it boils down to chosing a good scaling of the reconstruction operator, I think...
#see Kuzmin book for many ideas
class FCTVForm():
    def __init__(self, topo):
        
        self.ncells = topo.nkcells[topo.dim]
        self.maxne = topo.petscmesh.getMaxSizes()[0]

        self.EC = np.zeros((self.ncells, self.maxne), dtype=np.int32)
        self.NC = np.zeros((self.ncells, self.maxne), dtype=np.int32)
        self.nEC = np.zeros(self.ncells, dtype=np.int32)
        self.cellorients = np.zeros((self.ncells, self.maxne), dtype=np.int32)

        coff = topo.kcells_off[topo.dim]
        eoff = topo.kcells_off[topo.dim - 1]

        for c in range(topo.kcells[topo.dim][0], topo.kcells[topo.dim][1]):
            edges = topo.lower_dim_TC(c, topo.dim-1)
            nedges = edges.shape[0]
            self.EC[c - coff, :nedges] = edges - eoff
            self.nEC[c - coff] = nedges
            self.cellorients[c - coff, :nedges] = topo.lower_orientation(c)
            
    def apply(self, dens, flux, phi, dt, phi_min=None, phi_max=None, boundstype='pd'):
        
        densarr = dens.petsc_vec.getArray()
        densarr = densarr.reshape((dens.nelems, dens.ndofs, dens.bsize))
        fluxarr = flux.petsc_vec.getArray()
        fluxarr = fluxarr.reshape((flux.nelems, flux.ndofs, flux.bsize))
        phiarr = phi.petsc_vec.getArray()
        phiarr = phiarr.reshape((phi.nelems, phi.ndofs, phi.bsize))
        
        if boundstype == 'pd':
            _fct_pd(densarr, fluxarr, phiarr, self.EC, self.nEC, self.cellorients, self.ncells, dens.ndofs, dens.bsize, dt)
        if boundstype == 'bp':
            phiminarr = phi_min.petsc_vec.getArray()
            phiminarr = phiminarr.reshape((phi_min.nelems, phi_min.ndofs, phi_min.bsize))
            phimaxarr = phi_max.petsc_vec.getArray()
            phimaxarr = phimaxarr.reshape((phi_max.nelems, phi_max.ndofs, phi_max.bsize))
            _fct_bp(densarr, fluxarr, phiarr, phiminarr, phimaxarr, self.Ec, self.nEC, self.cellorients, self.ncells, dens.ndofs, dens.bsize, dt)
