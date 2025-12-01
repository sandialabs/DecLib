
from DecLib import KForm
from DecLib import VolumeFormRecon, LieDerivativeVForm, FCTVForm
from DecLib import ADD_MODE

from math import exp, log, pow

from numba import njit, prange
import numpy as np

from dynamics import Dynamics, Statistics, Diagnostics, Statistic


@njit(parallel=True, cache=True)
def _compute_edge_fluxes(densfluxarr, densarr, ufluxarr, nedges, ndofs):
        for e in prange(nedges):
            for l in range(ndofs):
                densfluxarr[e,l] = ufluxarr[e] * densarr[e,l]

@njit(parallel=True, cache=True)
def _scale_recons_by_phi(densarr, phiarr, nedges, ndofs):
        for e in prange(nedges):
            for l in range(ndofs):
                densarr[e,l] = phiarr[e,l] * densarr[e,l]
                
class AdvectionDynamics(Dynamics):
    def __init__(self, meshes, params, ic, construct=True):

        self.denslist = ['rho',]
        self.scaledof = 0
        for t in range(params['num_active_tracers']):
            self.denslist.append('TA' + str(t))
        
        if construct:
            Dynamics.__init__(self, meshes, params, ic)
            
            self.prog_vars['dens'] = KForm(meshes.dim, self.meshes.dtopo, meshes.dRBundle, 'dens', create_petsc=True, ndofs = len(self.denslist))
            self.aux_vars['dens_e'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'dens_e', create_petsc=True, ndofs = len(self.denslist))
            self.aux_vars['uflux'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'uflux', create_petsc=True)
            
            self.aux_vars['dens_flux'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'dens_e', create_petsc=True, ndofs = len(self.denslist))
            self.aux_vars['phi'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'phi', create_petsc=True, ndofs = len(self.denslist))
            self.aux_vars['phimin'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'phimin', create_petsc=True, ndofs = len(self.denslist))
            self.aux_vars['phimax'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'phimax', create_petsc=True, ndofs = len(self.denslist))

            #ADD SOME PRIMAL GRID RECON STUFF IE Q/F, utflux, etc.
            #ADD SOME MOMENTUM ADVECTION STUFF AS WELL!
            
            print('created dynamics vars')

            self.dens_recon = VolumeFormRecon(meshes.dtopo, meshes.dgeom, recontype=params['dens_recon_type'], reconorder=params['dens_recon_order'], tanh_coeff=params['dens_tanh_coeff'])
            self.lie_derivative = LieDerivativeVForm(meshes)
            
            self.fct = FCTVForm(meshes.dtopo)
            
            #ADD Q/F RECON STUFF!
            #ADD SOME MOMENTUM ADVECTION STUFF AS WELL!
        
    def compute_aux(self, state, t, dt):
        self.IC.set_uflux(self.aux_vars, t)
        self.dens_recon.apply(state['dens'], self.aux_vars['dens_e'], self.aux_vars['uflux'])
        if self.meshes.dtopo.has_boundary:
            self.IC.set_bnd_dhdxvars(self.aux_vars, t)
        
        if not (self.params['dens_fct'] == 'none'):
            densfluxarr = self.aux_vars['dens_flux'].petsc_vec.getArray()
            densfluxarr = densfluxarr.reshape((self.aux_vars['dens_flux'].nelems, self.aux_vars['dens_flux'].ndofs))
            
            densarr = self.aux_vars['dens_e'].petsc_vec.getArray()
            densarr = densarr.reshape((self.aux_vars['dens_e'].nelems, self.aux_vars['dens_e'].ndofs))
            
            phiarr = self.aux_vars['phi'].petsc_vec.getArray()
            phiarr = phiarr.reshape((self.aux_vars['phi'].nelems, self.aux_vars['phi'].ndofs))
            
            ufluxarr = self.aux_vars['uflux'].petsc_vec.getArray()
            _compute_edge_fluxes(densfluxarr, densarr, ufluxarr, self.meshes.dtopo.nkcells[self.meshes.dim-1], state['dens'].ndofs)
            self.fct.apply(state['dens'], self.aux_vars['dens_flux'], self.aux_vars['phi'], dt, phi_min = self.aux_vars['phimin'], phi_max=self.aux_vars['phimax'], boundstype=self.params['dens_fct'])
            _scale_recons_by_phi(densarr, phiarr, self.meshes.dtopo.nkcells[self.meshes.dim-1], state['dens'].ndofs)
                
        #ADD Q/F RECON STUFF
        #ADD SOME MOMENTUM ADVECTION STUFF AS WELL!
        
    def compute_rhs(self, rhs, t, dt):

		#compute dens tend
        self.lie_derivative.apply(self.aux_vars['dens_e'], self.aux_vars['uflux'], rhs['dens'])

        #ADD Q/F RECON STUFF

class AdvectionDiagnostics(Diagnostics):
    def __init__(self, dyn, construct=True):
        
        self.diaglist = ['uflux',]
        
        if construct:
            self.dyn = dyn
            self.vars = {}
            self.vars['dens0'] = KForm(0, dyn.meshes.ptopo, dyn.meshes.pRBundle, 'dens0', create_petsc=True, ndofs = len(dyn.denslist))
            self.vars['scalar0'] = KForm(0, dyn.meshes.ptopo, dyn.meshes.pRBundle, 'scalar0', create_petsc=True, ndofs = len(dyn.denslist))
            self.vars['uflux'] = KForm(dyn.meshes.dim-1, dyn.meshes.dtopo, dyn.meshes.dRBundle, 'uflux', create_petsc=True)

            #ADD SOME MOMENTUM ADVECTION STUFF AS WELL!
            #ADD Q/F RECON STUFF

#NEED TO JIT THIS!
    def compute(self, t):

        densarr = self.dyn.prog_vars['dens'].petsc_vec.getArray()
        densarr = densarr.reshape((self.dyn.prog_vars['dens'].nelems, self.dyn.prog_vars['dens'].ndofs))
        dens0arr = self.vars['dens0'].petsc_vec.getArray()
        dens0arr = dens0arr.reshape((self.vars['dens0'].nelems, self.vars['dens0'].ndofs))
        scalar0arr = self.vars['scalar0'].petsc_vec.getArray()
        scalar0arr = scalar0arr.reshape((self.vars['scalar0'].nelems, self.vars['scalar0'].ndofs))
        
        #self.vars['uflux'].petsc_vec.copy(self.dyn.aux_vars['uflux'].petsc_vec)
        self.dyn.aux_vars['uflux'].petsc_vec.copy(self.vars['uflux'].petsc_vec)

        dn_off = self.dyn.meshes.dtopo.kcells_off[self.dyn.meshes.dim]
        for dn in range(self.dyn.meshes.dtopo.kcells[self.dyn.meshes.dim][0] - dn_off, self.dyn.meshes.dtopo.kcells[self.dyn.meshes.dim][1] - dn_off):
            cellarea = self.dyn.meshes.dgeom.get_entity_size(self.dyn.meshes.dim, dn)
            
            for l in range(len(self.dyn.denslist)):
                dens0arr[dn,l] = densarr[dn,l] / cellarea
                scalar0arr[dn,l] = densarr[dn,l] / densarr[dn, self.dyn.scaledof]

        #DO WE NEED THESE ASSEMBLES?
        for k,v in self.vars.items():
            v.petsc_vec.assemble()
            


class AdvectionStatistics(Statistics):
    def __init__(self, dyn, compute_bnd_fluxes, construct=True):
        
        self.dyn = dyn
        self.stats = {}
        self.vars = {}

        self.energylist = []
        
        if construct:
            self.stats['dens_total'] = Statistic('dens_total', dyn.params['Nsteps'], dyn.params['nstat'], ndofs=len(self.dyn.denslist))
            self.stats['dens_max'] = Statistic('dens_max', dyn.params['Nsteps'], dyn.params['nstat'], ndofs=len(self.dyn.denslist))
            self.stats['dens_min'] = Statistic('dens_min', dyn.params['Nsteps'], dyn.params['nstat'], ndofs=len(self.dyn.denslist))

            if self.dyn.meshes.dtopo.has_boundary and compute_bnd_fluxes:
                self.stats['mass_fluxes'] = Statistic('mass_fluxes', dyn.params['Nsteps'], 1, ndofs=len(self.dyn.denslist))

        #ADD SOME MOMENTUM ADVECTION STUFF AS WELL!
        #ADD Q/F RECON STUFF
        
#BROKEN FOR MPI- NEEDS A REDUCTION!
    def compute_bnd_fluxes(self, alpha, k, t, dt):
        if self.dyn.meshes.dtopo.has_boundary:

            massflux_arr = self.stats['mass_fluxes'].petsc_vec.getArray()
            massflux_arr = massflux_arr.reshape((self.stats['mass_fluxes'].statsize, self.stats['mass_fluxes'].ndofs))
        
            ufluxarr = self.dyn.aux_vars['uflux'].petsc_vec.getArray()
            reconarr = self.dyn.aux_vars['dens_e'].petsc_vec.getArray()
            reconarr = reconarr.reshape((self.dyn.meshes.dtopo.nkcells[self.dyn.meshes.dim-1], self.stats['mass_fluxes'].ndofs))
                        
            doffset = self.dyn.meshes.dtopo.kcells_off[self.dyn.meshes.dim-1]

            mflux = np.zeros(self.stats['mass_fluxes'].ndofs)

        #ADD SOME MOMENTUM ADVECTION STUFF AS WELL!
        #ADD Q/F RECON STUFF
        
            #JIT THIS
            for dbs in self.dyn.meshes.dtopo.bkcells[self.dyn.meshes.dim-1]:
                orient = self.dyn.meshes.dtopo.higher_orientation(dbs)[0]
                for l in range(self.stats['mass_fluxes'].ndofs):
                    mflux[l] += ufluxarr[dbs - doffset] * orient * reconarr[dbs - doffset, l]

#This covers both EC2 (1 call with alpha=1) and s-stage RK (s calls with alpha=bs)
            for l in range(self.stats['mass_fluxes'].ndofs):
                massflux_arr[k,l] += alpha * dt * mflux[l]
                
#BROKEN FOR MPI- NEEDS A REDUCATION!
    def compute(self, ind, t):
        #print('stats computation')
        
        #JIT THIS
        densarr = self.dyn.prog_vars['dens'].petsc_vec.getArray()
        densarr = densarr.reshape((self.dyn.prog_vars['dens'].nelems, self.dyn.prog_vars['dens'].ndofs))
        dens_stat_arr = self.stats['dens_total'].petsc_vec.getArray()
        dens_stat_arr = dens_stat_arr.reshape((self.stats['dens_total'].statsize, self.stats['dens_total'].ndofs))
        
        dens_max_arr = self.stats['dens_max'].petsc_vec.getArray()
        dens_max_arr = dens_max_arr.reshape((self.stats['dens_max'].statsize, self.stats['dens_max'].ndofs))
        dens_min_arr = self.stats['dens_min'].petsc_vec.getArray()
        dens_min_arr = dens_min_arr.reshape((self.stats['dens_min'].statsize, self.stats['dens_min'].ndofs))
                
        #Marr = self.dyn.prog_vars['M'].petsc_vec.getArray()
        #Marr = Marr.reshape((self.dyn.meshes.dtopo.nkcells[self.dyn.meshes.dim], self.dyn.meshes.dim))
        #Mstats = self.stats['M_total'].petsc_vec.getArray()
        #Mstats = Mstats.reshape((Mstats.shape[0]//self.dyn.meshes.dim, self.dyn.meshes.dim))
        
        dn_off = self.dyn.meshes.dtopo.kcells_off[self.dyn.meshes.dim]
        for dn in range(self.dyn.meshes.dtopo.kcells[self.dyn.meshes.dim][0] - dn_off, self.dyn.meshes.dtopo.kcells[self.dyn.meshes.dim][1] - dn_off):
            for l in range(len(self.dyn.denslist)):
                dens_stat_arr[ind,l] += densarr[dn,l]
        for l in range(len(self.dyn.denslist)):
            dens_max_arr[ind,l] = np.max(densarr[:,l])
            dens_min_arr[ind,l] = np.min(densarr[:,l])

        #    for i in range(self.dyn.meshes.dim):
        #        Mstats[ind,i] += Marr[dn,i]

        #DO WE NEED THESE ASSEMBLES?
        for k,v in self.stats.items():
            v.petsc_vec.assemble()
