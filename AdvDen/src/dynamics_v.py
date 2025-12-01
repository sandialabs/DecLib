
from DecLib import KForm
from DecLib import VolumeFormRecon, LieDerivativeVForm, DiamondVForm_V, InteriorProductV
from DecLib import ADD_MODE

from math import exp, log, pow

from numba import njit, prange
import numpy as np

from dynamics import Dynamics, Statistics, Diagnostics, Statistic
from thermodynamics import getThermo
from hamiltonians import getHamiltonian

class AdvDensVDynamics(Dynamics):
    def __init__(self, meshes, params, ic, construct=True):

        thermo = getThermo(params, ic)
        hamiltonian = getHamiltonian(params, meshes, thermo, construct=construct)
        
        self.denslist = hamiltonian.denslist
        self.hamiltonian = hamiltonian
        self.scaledof = hamiltonian.scaledof
        
        if construct:
            Dynamics.__init__(self, meshes, params, ic)

            self.prog_vars['dens'] = KForm(meshes.dim, self.meshes.dtopo, meshes.dRBundle, 'dens', create_petsc=True, ndofs = len(self.denslist))
            self.prog_vars['v'] = KForm(1, self.meshes.ptopo, meshes.pRBundle, 'v', create_petsc=True)

            self.aux_vars['dens_e'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'dens_e', create_petsc=True, ndofs = len(self.denslist))
            self.aux_vars['q_e'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'q_e', create_petsc=True)
            self.aux_vars['f_e'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'f_e', create_petsc=True)

            self.aux_vars['F'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'F', create_petsc=True)
            self.aux_vars['B'] = KForm(0, self.meshes.ptopo, meshes.pRBundle, 'B', create_petsc=True, ndofs = len(self.denslist))

            print('created dynamics vars')

            self.dens_recon = VolumeFormRecon(meshes.dtopo, meshes.dgeom, recontype=params['dens_recon_type'], reconorder=params['dens_recon_order'], tanh_coeff=params['dens_tanh_coeff'])
            self.q_recon = VolumeFormRecon(meshes.ptopo, meshes.pgeom, recontype=params['q_recon_type'], reconorder=params['q_recon_order'], tanh_coeff=params['q_tanh_coeff'])
            self.f_recon = VolumeFormRecon(meshes.ptopo, meshes.pgeom, recontype=params['f_recon_type'], reconorder=params['f_recon_order'], tanh_coeff=params['q_tanh_coeff'])
            self.lie_derivative = LieDerivativeVForm(meshes)
            self.diamond = DiamondVForm_V(meshes)
            self.v_interior_product = InteriorProductV(meshes)

    def compute_dHdx(self, state, t, dt):
		#compute functional derivatives
        self.hamiltonian.compute_dHdx(state, self.aux_vars)
        if self.meshes.dtopo.has_boundary:
            self.IC.set_bnd_dhdxvars(self.aux_vars, t, scaledof=self.scaledof)
            
    def compute_Jvars(self, state, t, dt):
        #compute recons
#FIX 2D RECON
        #self.q_recon.apply(state['v'], self.aux_vars['q_e'], self.aux_vars['F'])
        #self.f_recon.apply(state['v'], self.aux_vars['f_e'], self.aux_vars['F'])
        self.dens_recon.apply(state['dens'], self.aux_vars['dens_e'], self.aux_vars['F'], scale=state['dens'], scaledof=self.scaledof)
        if self.meshes.dtopo.has_boundary:
            self.IC.set_bnd_jvars(self.aux_vars, t, scaledof=self.scaledof) 
            
    def compute_aux(self, state, t, dt):
        self.compute_Jvars(state, t, dt)
        self.compute_dHdx(state, t, dt)
            
    def compute_rhs(self, rhs, t, dt):
        #compute V tend
        self.v_interior_product.apply(self.aux_vars['q_e'], self.aux_vars['F'], rhs['v'])
        self.v_interior_product.apply(self.aux_vars['f_e'], self.aux_vars['F'], rhs['v'], mode=ADD_MODE)
        self.diamond.apply(self.aux_vars['dens_e'], self.aux_vars['B'], rhs['v'], mode=ADD_MODE)

		#compute dens tend
        self.lie_derivative.apply(self.aux_vars['dens_e'], self.aux_vars['F'], rhs['dens'])


#REALLY p and T are varset specific....
class AdvDensVDiagnostics(Diagnostics):
    def __init__(self, dyn, construct=True):
        self.dyn = dyn
        self.diaglist = ['v',]
        for hamildiag in self.dyn.hamiltonian.diaglist:
            self.diaglist.append(hamildiag)
        
        if (construct):
            self.vars = {}
            self.vars['dens0'] = KForm(0, dyn.meshes.ptopo, dyn.meshes.pRBundle, 'dens0', create_petsc=True, ndofs = len(dyn.denslist))
            self.vars['scalar0'] = KForm(0, dyn.meshes.ptopo, dyn.meshes.pRBundle, 'scalar0', create_petsc=True, ndofs = len(dyn.denslist))
            self.vars['v'] = KForm(1, dyn.meshes.ptopo, dyn.meshes.pRBundle, 'v', create_petsc=True)
            self.dyn.hamiltonian.add_diagnostic_vars(self.vars)
        

#NEED TO JIT THIS!
    def compute(self, t):

        self.dyn.hamiltonian.compute_diagnostic_vars(self.dyn.prog_vars, self.vars)

        densarr = self.dyn.prog_vars['dens'].petsc_vec.getArray()
        densarr = densarr.reshape((self.dyn.prog_vars['dens'].nelems, self.dyn.prog_vars['dens'].ndofs))
        varr =  self.dyn.prog_vars['v'].petsc_vec.getArray()
        dens0arr = self.vars['dens0'].petsc_vec.getArray()
        dens0arr = dens0arr.reshape((self.vars['dens0'].nelems, self.vars['dens0'].ndofs))
        scalar0arr = self.vars['scalar0'].petsc_vec.getArray()
        scalar0arr = scalar0arr.reshape((self.vars['scalar0'].nelems, self.vars['scalar0'].ndofs))

        v0arr = self.vars['v'].petsc_vec.getArray()

        pe_off = self.dyn.meshes.ptopo.kcells_off[1]
        for pe in range(self.dyn.meshes.ptopo.kcells[1][0] - pe_off, self.dyn.meshes.ptopo.kcells[1][1] - pe_off):
            edgelength = self.dyn.meshes.pgeom.get_entity_size(self.dyn.meshes.dim, pe)
            v0arr[pe] = varr[pe] / edgelength
            
        dn_off = self.dyn.meshes.dtopo.kcells_off[self.dyn.meshes.dim]
        for dn in range(self.dyn.meshes.dtopo.kcells[self.dyn.meshes.dim][0] - dn_off, self.dyn.meshes.dtopo.kcells[self.dyn.meshes.dim][1] - dn_off):
            cellarea = self.dyn.meshes.dgeom.get_entity_size(self.dyn.meshes.dim, dn)
            
            for l in range(len(self.dyn.denslist)):
                dens0arr[dn,l] = densarr[dn,l] / cellarea
                scalar0arr[dn,l] = densarr[dn,l] / densarr[dn, self.dyn.scaledof]
                            
        #DO WE NEED THESE ASSEMBLES?
        for k,v in self.vars.items():
            v.petsc_vec.assemble()
            
                
class AdvDensVStatistics(Statistics):
    def __init__(self, dyn, compute_bnd_fluxes, construct=True):
        self.dyn = dyn
        self.stats = {}
        self.vars = {}
        
        self.energylist = dyn.hamiltonian.energylist
        
        if construct:
            self.stats['energies'] = Statistic('energies', dyn.params['Nsteps'], dyn.params['nstat'], ndofs=len(dyn.hamiltonian.energylist))
            self.stats['dens_total'] = Statistic('dens_total', dyn.params['Nsteps'], dyn.params['nstat'], ndofs=len(self.dyn.denslist))

            if self.dyn.meshes.dtopo.has_boundary and compute_bnd_fluxes:
                self.stats['mass_fluxes'] = Statistic('mass_fluxes', dyn.params['Nsteps'], 1, ndofs=len(self.dyn.denslist))
                self.stats['energy_flux'] = Statistic('energy_flux', dyn.params['Nsteps'], 1, ndofs=1)

#BROKEN FOR MPI- NEEDS A REDUCTION!
    def compute_bnd_fluxes(self, alpha, k, t, dt):
        if self.dyn.meshes.dtopo.has_boundary:

            massflux_arr = self.stats['mass_fluxes'].petsc_vec.getArray()
            massflux_arr = massflux_arr.reshape((self.stats['mass_fluxes'].statsize, self.stats['mass_fluxes'].ndofs))
            energyflux_arr = self.stats['energy_flux'].petsc_vec.getArray()
            energyflux_arr = energyflux_arr.reshape((self.stats['energy_flux'].statsize, self.stats['energy_flux'].ndofs))
        
            Farr = self.dyn.aux_vars['F'].petsc_vec.getArray()
            reconarr = self.dyn.aux_vars['dens_e'].petsc_vec.getArray()
            reconarr = reconarr.reshape((self.dyn.meshes.dtopo.nkcells[self.dyn.meshes.dim-1], self.stats['mass_fluxes'].ndofs))
            Barr = self.dyn.aux_vars['B'].petsc_vec.getArray()
            Barr = Barr.reshape((self.dyn.aux_vars['B'].nelems, self.dyn.aux_vars['B'].ndofs))
        
            doffset = self.dyn.meshes.dtopo.kcells_off[self.dyn.meshes.dim-1]
            poffset = self.dyn.meshes.ptopo.kcells_off[0]

            mflux = np.zeros(self.stats['mass_fluxes'].ndofs)
            eflux = 0.0

            #JIT THIS
            for dbs in self.dyn.meshes.dtopo.bkcells[self.dyn.meshes.dim-1]:
                pbe = self.dyn.meshes.pdmapping.dbnmk_to_pbk(dbs)
                orient = self.dyn.meshes.dtopo.higher_orientation(dbs)[0]
                for l in range(self.stats['mass_fluxes'].ndofs):
                    mflux[l] += Farr[dbs - doffset] * orient * reconarr[dbs - doffset, l]
                    eflux += Farr[dbs - doffset] * orient * reconarr[dbs - doffset, l] * Barr[pbe - poffset, l]

#This covers both EC2 (1 call with alpha=1) and s-stage RK (s calls with alpha=bs)
            energyflux_arr[k,0] += alpha * dt * eflux
            for l in range(self.stats['mass_fluxes'].ndofs):
                massflux_arr[k,l] += alpha * dt * mflux[l]
                

#BROKEN FOR MPI- NEEDS A REDUCTION!
    def compute(self, ind, t):
        #print('stats computation')
        
        self.dyn.hamiltonian.compute_energy(self.dyn.prog_vars, self.stats, ind)

        #JIT THIS
        densarr = self.dyn.prog_vars['dens'].petsc_vec.getArray()
        densarr = densarr.reshape((self.dyn.prog_vars['dens'].nelems, self.dyn.prog_vars['dens'].ndofs))
        
        dens_stat_arr = self.stats['dens_total'].petsc_vec.getArray()
        dens_stat_arr = dens_stat_arr.reshape((self.stats['dens_total'].statsize, self.stats['dens_total'].ndofs))

        dn_off = self.dyn.meshes.dtopo.kcells_off[self.dyn.meshes.dim]
        for dn in range(self.dyn.meshes.dtopo.kcells[self.dyn.meshes.dim][0] - dn_off, self.dyn.meshes.dtopo.kcells[self.dyn.meshes.dim][1] - dn_off):
            for l in range(len(self.dyn.denslist)):
                dens_stat_arr[ind,l] += densarr[dn,l]
            
        #DO WE NEED THESE ASSEMBLES?
        for k,v in self.stats.items():
            v.petsc_vec.assemble()
