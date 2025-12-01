
from DecLib import KForm
from DecLib import VolumeFormRecon, LieDerivativeVForm, DiamondVForm_MLP, LieDerivativeM, InteriorProductMLP
from DecLib import ADD_MODE
from DecLib import CovariantExteriorDerivativeVForm, ExteriorDerivativeVForm

from math import exp, log, pow

from numba import njit, prange
import numpy as np

from dynamics import Dynamics, Statistics, Diagnostics, Statistic
from thermodynamics import getThermo
from hamiltonians import getHamiltonian
from operators import getRegularization

class AdvDensMLPDynamics(Dynamics):
    def __init__(self, meshes, params, ic, construct=True):

        thermo = getThermo(params, ic)
        self.hamiltonian = getHamiltonian(params, meshes, thermo, construct=construct)
        self.denslist = self.hamiltonian.denslist
        self.scaledof = self.hamiltonian.scaledof
        self.viscosity = getRegularization(params, self.hamiltonian, meshes, self.hamiltonian.entropydof, len(self.denslist), construct=construct)

        if construct:
            Dynamics.__init__(self, meshes, params, ic)

            self.prog_vars['dens'] = KForm(meshes.dim, self.meshes.dtopo, meshes.dRBundle, 'dens', create_petsc=True, ndofs = len(self.denslist))
            self.prog_vars['M'] = KForm(meshes.dim, self.meshes.dtopo, meshes.dCTBundle, 'M', create_petsc=True)

            self.aux_vars['dens_e'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'dens_e', create_petsc=True, ndofs = len(self.denslist))
            self.aux_vars['M_e'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dCTBundle, 'M_e', create_petsc=True)

            self.aux_vars['u'] = KForm(0, self.meshes.ptopo, meshes.pTBundle, 'u', create_petsc=True)
            self.aux_vars['uflux'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'uflux', create_petsc=True)

            self.aux_vars['B'] = KForm(0, self.meshes.ptopo, meshes.pRBundle, 'B', create_petsc=True, ndofs = len(self.denslist))
            self.aux_vars['h'] = KForm(meshes.dim, self.meshes.dtopo, meshes.dRBundle, 'h', create_petsc=True)

            print('created dynamics vars')

            self.dens_recon = VolumeFormRecon(meshes.dtopo, meshes.dgeom, recontype=params['dens_recon_type'], reconorder=params['dens_recon_order'], tanh_coeff=params['dens_tanh_coeff'])
            self.M_recon = VolumeFormRecon(meshes.dtopo, meshes.dgeom, recontype=params['M_recon_type'], reconorder=params['M_recon_order'], tanh_coeff=params['M_tanh_coeff'])
            self.interior_product = InteriorProductMLP(meshes)
            self.lie_derivative = LieDerivativeVForm(meshes)
            self.diamond = DiamondVForm_MLP(meshes)
            self.m_lie_derivative = LieDerivativeM(meshes)

            self.aux_vars['dens_flux'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'dens_flux', create_petsc=True, ndofs = len(self.denslist))
            self.aux_vars['M_flux'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dCTBundle, 'M_flux', create_petsc=True)
            self.aux_vars['dens_source'] = KForm(meshes.dim, self.meshes.dtopo, meshes.dRBundle, 'dens_source', create_petsc=True, ndofs = len(self.denslist))
            self.aux_vars['epsilon'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'epsilon', create_petsc=True)
            self.aux_vars['alpha'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'alpha', create_petsc=True)

            self.m_covextderiv = CovariantExteriorDerivativeVForm(meshes)
            self.dens_covextderiv = ExteriorDerivativeVForm(meshes)

    def compute_dHdx(self, state, t, dt):
		#compute functional derivatives
        self.hamiltonian.compute_dHdx(state, self.aux_vars)
        self.interior_product.apply(self.aux_vars['u'], self.aux_vars['uflux'])
        if self.meshes.dtopo.has_boundary:
            self.IC.set_bnd_dhdxvars(self.aux_vars, t)
        #print(self.aux_vars['uflux'].petsc_vec.getArray())

    def compute_Jvars(self, state, t, dt):
        #compute recons
        self.M_recon.apply(state['M'], self.aux_vars['M_e'], self.aux_vars['uflux'])
        self.dens_recon.apply(state['dens'], self.aux_vars['dens_e'], self.aux_vars['uflux'])
        if self.meshes.dtopo.has_boundary:
            self.IC.set_bnd_jvars(self.aux_vars, t)

    def pre_step(self):
        self.viscosity.pre_step(self.aux_vars)

    def post_step(self):
        self.viscosity.post_step(self.aux_vars)

    def compute_aux(self, state, t, dt):
        self.compute_Jvars(state, t, dt)
        self.compute_dHdx(state, t, dt)

        self.hamiltonian.compute_pointwise_energy(state, self.aux_vars, self.aux_vars['h'])

        #set boundary dens/M values
#RENAME/SWITCH THIS!
        self.IC.set_bnd_fluxvars(self.aux_vars, t)

        self.viscosity.compute_fluxes(state, self.aux_vars)


    def compute_rhs(self, rhs, t, dt):
        #compute M tend
        self.m_lie_derivative.apply(self.aux_vars['M_e'], self.aux_vars['u'], self.aux_vars['uflux'], rhs['M'])
        #print(rhs['M'].petsc_vec.getArray())
        self.diamond.apply(self.aux_vars['dens_e'], self.aux_vars['B'], rhs['M'], mode=ADD_MODE)
        #print(rhs['M'].petsc_vec.getArray())

		#compute dens tend
        self.lie_derivative.apply(self.aux_vars['dens_e'], self.aux_vars['uflux'], rhs['dens'])

        #add viscosity
        #compute M tend
        self.m_covextderiv.apply(self.aux_vars['M_flux'], rhs['M'], mode=ADD_MODE)

	    #compute dens tend
        self.dens_covextderiv.apply(self.aux_vars['dens_flux'], rhs['dens'], mode=ADD_MODE)
        rhs['dens'].petsc_vec.axpy(-1.0, self.aux_vars['dens_source'].petsc_vec)




class AdvDensMLPDiagnostics(Diagnostics):
    def __init__(self, dyn, construct=True):
        self.dyn = dyn
        self.diaglist = ['M0', 'u']
        for hamildiag in self.dyn.hamiltonian.diaglist:
            self.diaglist.append(hamildiag)
        for viscdiag in self.dyn.viscosity.diaglist:
            self.diaglist.append(viscdiag)
#ADD entropy generation rate ie Pi (part of regularization)
#ADD pointwise total energy (in Hamiltonian)

#add energy related stuff

        if (construct):
            self.vars = {}
            self.vars['dens0'] = KForm(0, dyn.meshes.ptopo, dyn.meshes.pRBundle, 'dens0', create_petsc=True, ndofs = len(dyn.denslist))
            self.vars['scalar0'] = KForm(0, dyn.meshes.ptopo, dyn.meshes.pRBundle, 'scalar0', create_petsc=True, ndofs = len(dyn.denslist))
            self.dyn.hamiltonian.add_diagnostic_vars(self.vars)
            self.dyn.viscosity.add_diagnostic_vars(self.vars)
            self.vars['M0'] = KForm(0, dyn.meshes.ptopo, dyn.meshes.pTBundle, 'M0', create_petsc=True)
            self.vars['u'] = KForm(0, dyn.meshes.ptopo, dyn.meshes.pTBundle, 'u', create_petsc=True)
            self.vars['h'] = KForm(0, dyn.meshes.ptopo, dyn.meshes.pRBundle, 'h', create_petsc=True)

#NEED TO JIT THIS!
    def compute(self, t):

        self.dyn.hamiltonian.compute_diagnostic_vars(self.dyn.prog_vars, self.vars)
        self.dyn.viscosity.compute_diagnostic_vars(self.dyn.prog_vars, self.vars)

        densarr = self.dyn.prog_vars['dens'].petsc_vec.getArray()
        densarr = densarr.reshape((self.dyn.prog_vars['dens'].nelems, self.dyn.prog_vars['dens'].ndofs))
        Marr =  self.dyn.prog_vars['M'].petsc_vec.getArray()
        Marr = Marr.reshape((self.dyn.meshes.dtopo.nkcells[self.dyn.meshes.dim], self.dyn.meshes.dim))
        dens0arr = self.vars['dens0'].petsc_vec.getArray()
        dens0arr = dens0arr.reshape((self.vars['dens0'].nelems, self.vars['dens0'].ndofs))
        scalar0arr = self.vars['scalar0'].petsc_vec.getArray()
        scalar0arr = scalar0arr.reshape((self.vars['scalar0'].nelems, self.vars['scalar0'].ndofs))

        M0arr = self.vars['M0'].petsc_vec.getArray()
        M0arr = M0arr.reshape((self.dyn.meshes.dtopo.nkcells[self.dyn.meshes.dim], self.dyn.meshes.dim))
        uarr = self.vars['u'].petsc_vec.getArray()
        uarr = uarr.reshape((self.dyn.meshes.dtopo.nkcells[self.dyn.meshes.dim], self.dyn.meshes.dim))

        dn_off = self.dyn.meshes.dtopo.kcells_off[self.dyn.meshes.dim]
        for dn in range(self.dyn.meshes.dtopo.kcells[self.dyn.meshes.dim][0] - dn_off, self.dyn.meshes.dtopo.kcells[self.dyn.meshes.dim][1] - dn_off):
            cellarea = self.dyn.meshes.dgeom.get_entity_size(self.dyn.meshes.dim, dn)

            for l in range(len(self.dyn.denslist)):
                dens0arr[dn,l] = densarr[dn,l] / cellarea
                scalar0arr[dn,l] = densarr[dn,l] / densarr[dn, self.dyn.scaledof]

            for j in range(self.dyn.meshes.dim):
                M0arr[dn,j] = Marr[dn] / cellarea
                uarr[dn,j] = Marr[dn] / densarr[dn, self.dyn.scaledof]

        #DO WE NEED THESE ASSEMBLES?
        for k,v in self.vars.items():
            v.petsc_vec.assemble()



class AdvDensMLPStatistics(Statistics):
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
                self.stats['momentum_flux'] = Statistic('momentum_flux', dyn.params['Nsteps'], 1, ndofs=1)

            self.stats['M_total'] =  Statistic('M_total', dyn.params['Nsteps'], dyn.params['nstat'], ndofs = self.dyn.meshes.dim)

#BROKEN FOR MPI- NEEDS A REDUCTION!
    def compute_bnd_fluxes(self, alpha, k, t, dt):
        if self.dyn.meshes.dtopo.has_boundary:

            massflux_arr = self.stats['mass_fluxes'].petsc_vec.getArray()
            massflux_arr = massflux_arr.reshape((self.stats['mass_fluxes'].statsize, self.stats['mass_fluxes'].ndofs))
            energyflux_arr = self.stats['energy_flux'].petsc_vec.getArray()
            Mflux_arr = self.stats['momentum_flux'].petsc_vec.getArray()

            ufluxarr = self.dyn.aux_vars['uflux'].petsc_vec.getArray()
            reconarr = self.dyn.aux_vars['dens_e'].petsc_vec.getArray()
            reconarr = reconarr.reshape((self.dyn.meshes.dtopo.nkcells[self.dyn.meshes.dim-1], self.stats['mass_fluxes'].ndofs))
            Barr = self.dyn.aux_vars['B'].petsc_vec.getArray()
            Barr = Barr.reshape((self.dyn.aux_vars['B'].nelems, self.dyn.aux_vars['B'].ndofs))

            mearr = self.dyn.aux_vars['M_e'].petsc_vec.getArray()
            mearr = mearr.reshape((self.dyn.aux_vars['M_e'].nelems, self.dyn.aux_vars['M_e'].bsize))
            uarr = self.dyn.aux_vars['u'].petsc_vec.getArray()
            uarr = uarr.reshape((self.dyn.aux_vars['u'].nelems, self.dyn.aux_vars['u'].bsize))

            doffset = self.dyn.meshes.dtopo.kcells_off[self.dyn.meshes.dim-1]
            poffset = self.dyn.meshes.ptopo.kcells_off[0]

            mflux = np.zeros(self.stats['mass_fluxes'].ndofs)
            eflux = 0.0
            Mflux = 0.0

            #JIT THIS
            for dbs in self.dyn.meshes.dtopo.bkcells[self.dyn.meshes.dim-1]:
                pbe = self.dyn.meshes.pdmapping.dbnmk_to_pbk(dbs)
                orient = self.dyn.meshes.dtopo.higher_orientation(dbs)[0]
                for l in range(self.stats['mass_fluxes'].ndofs):
                    mflux[l] += ufluxarr[dbs - doffset] * orient * reconarr[dbs - doffset, l]
                    eflux += ufluxarr[dbs - doffset] * orient * reconarr[dbs - doffset, l] * Barr[pbe - poffset, l]
                for d in range(self.dyn.meshes.dim):
                    eflux += ufluxarr[dbs - doffset] * orient * mearr[dbs - doffset, d] * uarr[pbe - poffset, d]
                    Mflux += ufluxarr[dbs - doffset] * orient * mearr[dbs - doffset, d]

#This covers both EC2 (1 call with alpha=1) and s-stage RK (s calls with alpha=bs)
            energyflux_arr[k] += alpha * dt * eflux
            Mflux_arr[k] += alpha * dt * Mflux
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
        Marr = self.dyn.prog_vars['M'].petsc_vec.getArray()
        Marr = Marr.reshape((self.dyn.meshes.dtopo.nkcells[self.dyn.meshes.dim], self.dyn.meshes.dim))
        Mstats = self.stats['M_total'].petsc_vec.getArray()
        Mstats = Mstats.reshape((Mstats.shape[0]//self.dyn.meshes.dim, self.dyn.meshes.dim))

        dn_off = self.dyn.meshes.dtopo.kcells_off[self.dyn.meshes.dim]
        for dn in range(self.dyn.meshes.dtopo.kcells[self.dyn.meshes.dim][0] - dn_off, self.dyn.meshes.dtopo.kcells[self.dyn.meshes.dim][1] - dn_off):
            for l in range(len(self.dyn.denslist)):
                dens_stat_arr[ind,l] += densarr[dn,l]
            for i in range(self.dyn.meshes.dim):
                Mstats[ind,i] += Marr[dn,i]

        #DO WE NEED THESE ASSEMBLES?
        for k,v in self.stats.items():
            v.petsc_vec.assemble()
