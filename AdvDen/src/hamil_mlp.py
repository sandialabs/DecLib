
from numba import njit, prange
import numpy as np
from math import exp, log, pow
from DecLib import KForm

class HamiltonianM():
    def __init__(self, meshes, params, construct=True):
        self.meshes = meshes
        self.params = params

        self.diaglist = []
        tracer_list = []
        for t in range(params['num_active_tracers']):
            tracer_list.append('TA' + str(t))
        for t in range(params['num_inactive_tracers']):
            tracer_list.append('TI' + str(t))
        self.denslist = ['rho','S', *tracer_list]
        self.scaledof = 0
        self.entropydof = 1
        self.ntracers = params['num_active_tracers']

        if (construct):
            self.dim = meshes.dim
            self.ncells = meshes.dtopo.nkcells[meshes.dim]
            self.cellareas = np.zeros(self.ncells)
            self.dn_to_p0 = np.zeros(self.ncells, dtype=np.int32)
            self.dn_identity = np.zeros(self.ncells, dtype=np.int32)
            self.bedges = np.zeros(self.meshes.dtopo.nbkcells[self.meshes.dim-1], dtype=np.int32)

            for i in range(self.ncells):
                self.cellareas[i] = meshes.dgeom.get_entity_size(meshes.dim, i)
                self.dn_to_p0[i] = meshes.pdmapping.dinmk_to_pk(i + meshes.dtopo.kcells_off[meshes.dim])
                self.dn_identity[i] = i

            self.bedges[:] = meshes.dtopo.bkcells[meshes.dim-1]

    def add_diagnostic_vars(self, dvars):
        pass

    def compute_diagnostic_vars(self, prog_vars, dvars):
        pass

    def compute_dHdx(self, state, dHdx, do_in_dn=False):
        Marr = state['M'].petsc_vec.getArray()
        Marr = Marr.reshape((state['M'].nelems, state['M'].bsize))

        densarr = state['dens'].petsc_vec.getArray()
        densarr = densarr.reshape((state['dens'].nelems, state['dens'].ndofs))

        Barr = dHdx['B'].petsc_vec.getArray()
        Barr = Barr.reshape((dHdx['B'].nelems, dHdx['B'].ndofs))
        uarr = dHdx['u'].petsc_vec.getArray()
        uarr = uarr.reshape((dHdx['u'].nelems, dHdx['u'].bsize))

        if (do_in_dn):
            self._dHdx(densarr, Marr, Barr, uarr, self.ntracers, self.ncells, self.dim, self.cellareas, self.dn_identity, *self.dHdx_params)
        else:
            self._dHdx(densarr, Marr, Barr, uarr, self.ntracers, self.ncells, self.dim, self.cellareas, self.dn_to_p0, *self.dHdx_params)

    def compute_pointwise_energy(self, state, aux, energy):

        densarr = state['dens'].petsc_vec.getArray()
        densarr = densarr.reshape((state['dens'].nelems, state['dens'].ndofs))
        Marr = state['M'].petsc_vec.getArray()
        Marr = Marr.reshape((state['M'].nelems, state['M'].bsize))
        energyarr = energy.petsc_vec.getArray()

        self._pointwise_energy(densarr, Marr, energyarr, self.ntracers, self.ncells, self.dim, self.cellareas, *self.energy_params)

    def compute_energy(self, state, stats, ind):

        densarr = state['dens'].petsc_vec.getArray()
        densarr = densarr.reshape((state['dens'].nelems, state['dens'].ndofs))

        energyarr = stats['energies'].petsc_vec.getArray()
        energyarr = energyarr.reshape((stats['energies'].statsize, stats['energies'].ndofs))

        Marr = state['M'].petsc_vec.getArray()
        Marr = Marr.reshape((state['M'].nelems, state['M'].bsize))

        self._energy(densarr, Marr, energyarr, self.ntracers, self.ncells, self.dim, self.cellareas, ind, *self.energy_params)

    def set_dHdx_bound(self, aux, t):

        Marr = aux['Mbound'].petsc_vec.getArray()
        Marr = Marr.reshape((aux['Mbound'].nelems, aux['Mbound'].bsize))

        densarr = aux['densbound'].petsc_vec.getArray()
        densarr = densarr.reshape((aux['densbound'].nelems, aux['densbound'].ndofs))

        Barr = aux['Bbound'].petsc_vec.getArray()
        Barr = Barr.reshape((aux['Bbound'].nelems, aux['Bbound'].ndofs))

        uarr = aux['ubound'].petsc_vec.getArray()
        uarr = uarr.reshape((aux['ubound'].nelems, aux['ubound'].bsize))

        harr = aux['hbound'].petsc_vec.getArray()
        harr = harr.reshape((aux['hbound'].nelems, aux['hbound'].bsize))

        return self._bnd_dHdx(densarr, Marr, Barr, uarr, harr, self.bedges, self.dim, self.ntracers, *self.dHdx_params)

class TSWE_M(HamiltonianM):
    def __init__(self, meshes, params, hs=None, R=None, construct=True):
        HamiltonianM.__init__(self, meshes, params, construct=construct)

        self.energylist = ['TE', 'KE', 'PE']
        self.hs = hs
        self.R = R
        self.name = 'tswe'

#THESE ARE MISSING HS TERMS...
        self._dHdx = _dHdx_tswe_mlp
        self._energy = _energy_tswe_mlp
        self._pointwise_energy = _pointwise_energy_tswe_mlp
        self._bnd_dHdx = _bnd_dHdx_tswe_mlp

        self.dHdx_params = []
        self.energy_params = []
        self.pressure_params = []

class CompressibleEuler_M(HamiltonianM):
    def __init__(self, meshes, params, thermo, R = None, geop = None, construct=True):
        HamiltonianM.__init__(self, meshes, params, construct=construct)

        self.diaglist = ['p', 'T', 'inte', 'h']
        self.energylist = ['TE', 'KE', 'IE']
        self.geop = geop
        self.R = R
        self.thermo = thermo
        if not (geop is None):
            self.energylist.append('PE')
        self.name = 'ce'

        self._dHdx = _dHdx_ce_mlp
        self._energy = _energy_ce_mlp
        self._pointwise_energy = _pointwise_energy_ce_mlp
        self._bnd_dHdx = _bnd_dHdx_ce_mlp

        self.dHdx_params = [self.thermo.Cv, self.thermo.gamma]
        self.energy_params = [self.thermo.Cv, self.thermo.gamma]
        self.pressure_params = [self.thermo.Cv, self.thermo.gamma]

 #H = 1/2 * m^2/rho + rho * u(alpha,eta) = 1/2 * m \wedgedot \star m \wedge 1/rho
 #dHdm = u = m/rho = \star m \wedge 1/rho
 #dHdrho = B_rho = -1/2 * m^2 / rho^2 + u + p\alpha - eta * T
 #dHdS = B_S = dudeta = T

#NEED TO FIGURE OUT HOW TO JIT CLASSES...
    def add_diagnostic_vars(self, dvars):
        dvars['p'] = KForm(0, self.meshes.ptopo, self.meshes.pRBundle, 'p', create_petsc=True)
        dvars['T'] = KForm(0, self.meshes.ptopo, self.meshes.pRBundle, 'T', create_petsc=True)
        dvars['inte'] = KForm(0, self.meshes.ptopo, self.meshes.pRBundle, 'inte', create_petsc=True)
        dvars['h'] = KForm(0, self.meshes.ptopo, self.meshes.pRBundle, 'h', create_petsc=True)

#THIS IS MISSING DN <-> P0 MAP
#PROBABLY OKAY THOUGH
    def compute_diagnostic_vars(self, prog_vars, dvars):
        parr = dvars['p'].petsc_vec.getArray()
        Tarr = dvars['T'].petsc_vec.getArray()
        intearr = dvars['inte'].petsc_vec.getArray()
        densarr = prog_vars['dens'].petsc_vec.getArray()
        densarr = densarr.reshape((prog_vars['dens'].nelems, prog_vars['dens'].ndofs))

        harr = dvars['h'].petsc_vec.getArray()
        Marr = prog_vars['M'].petsc_vec.getArray()
        Marr = Marr.reshape((prog_vars['M'].nelems, prog_vars['M'].bsize))

        dn_off = self.meshes.dtopo.kcells_off[self.meshes.dim]
        for dn in range(self.meshes.dtopo.kcells[self.meshes.dim][0] - dn_off, self.meshes.dtopo.kcells[self.meshes.dim][1] - dn_off):
            cellarea = self.meshes.dgeom.get_entity_size(self.meshes.dim, dn)

            rho0 = densarr[dn,0] / cellarea
            eta = densarr[dn,1] / cellarea / rho0
            parr[dn] = self.thermo.get_p(rho0, eta)
            Tarr[dn] = self.thermo.get_T(rho0, eta)
            inte = self.thermo.compute_u(rho0, eta)
            intearr[dn] = inte
            ke = 0
            for d in range(self.meshes.dim):
                ke += Marr[dn,d] * Marr[dn,d] / (cellarea * cellarea * rho0 * rho0 * 2.)
            harr[dn] = rho0 * inte + rho0 * ke


@njit
def _compute_k_u(i, dim, u, dn_to_p0, M, cellarea, rho0):
    k = 0
    for d in range(dim):
        u[dn_to_p0[i],d] = M[i,d] / cellarea / rho0
        k += M[i,d] * M[i,d] / (cellarea * cellarea * rho0 * rho0 * 2.)
    return k

@njit
def _compute_M_ke2(i, dim, M, cellarea, rho0):
    ke2 = 0
    for d in range(dim):
        ke2 += M[i,d] * M[i,d] / (cellarea * rho0 * 2.)
    return ke2

@njit(parallel=True, cache=True)
def _dHdx_tswe_mlp(dens, M, B, u, ntracers, ncells, dim, cellareas, dn_to_p0):
    for i in prange(ncells):

        #THIS IS LOWEST ORDER HODGE STAR!!!
        cellarea = cellareas[i]
        rho0 = dens[i,0] / cellarea
        S0 = dens[i,1] / cellarea

        k = _compute_k_u(i, dim, u, dn_to_p0, M, cellarea, rho0)

        B[dn_to_p0[i],0] = S0/2. - k
        for t in range(ntracers):
            T0 = dens[i,2+t] / cellarea
            B[dn_to_p0[i],0] += T0/2.
            B[dn_to_p0[i],2+t] = rho0/2.
        B[dn_to_p0[i],1] = rho0/2.

@njit(parallel=True, cache=True)
def _bnd_dHdx_tswe_mlp(densbnd, Mbnd, Bbnd, ubnd, hbnd, bedges, ndim, ntracers):

    for i in prange(bedges.shape[0]):
        be = bedges[i]
        rho = densbnd[be,0]
        S = densbnd[be,1]

        ke = 0.0
        for d in range(ndim):
            u = Mbnd[be,d] / rho
            ke += u*u / 2.0
            ubnd[be,d] = u

        hbnd[be] = rho * ke

        Bbnd[be,0] = S/2. - ke
        for t in range(ntracers):
            Bbnd[be,0] += densbnd[be,2+t]/2.
            Bbnd[be,2+t] = rho/2.
            hbnd[be] += dens[be,2+t] * rho / 2.
        Bbnd[be,1] = rho/2.
        hbnd[be] += rho * S / 2.

@njit
def _energy_tswe_mlp(dens, M, energies, ntracers, ncells, dim, cellareas, ind):
    for i in range(ncells):

        #THIS IS LOWEST ORDER HODGE STAR!!!
        cellarea = cellareas[i]
        rho0 = dens[i,0] / cellarea

        rho_ke2 = _compute_M_ke2(i, dim, M, cellarea, rho0)
        pot_energy = rho0 * dens[i,1] / 2.
        for t in range(ntracers):
            pot_energy += rho0 * dens[i,2+t] / 2.

        energies[ind,0] += rho_ke2 + pot_energy
        energies[ind,1] += rho_ke2
        energies[ind,2] += pot_energy

@njit(parallel=True, cache=True)
def _pointwise_energy_tswe_mlp(dens, M, h, ntracers, ncells, dim, cellareas):
    for i in prange(ncells):

        #THIS IS LOWEST ORDER HODGE STAR!!!
        cellarea = cellareas[i]
        rho0 = dens[i,0] / cellarea

        rho_ke2 = _compute_M_ke2(i, dim, M, cellarea, rho0)
        pot_energy = rho0 * dens[i,1] / 2.
        for t in range(ntracers):
            pot_energy += rho0 * dens[i,2+t] / 2.

        h[i] = rho_ke2 + pot_energy

#NEED TO FIGURE OUT HOW TO JIT CLASSES...
#IE THESE SHOULD COME FROM THERMO!!!

@njit
def compute_dudrho(rho, eta, Cv, gamma):
    return pow(rho, gamma - 2.0) * exp(eta / Cv)

@njit
def compute_dudeta(rho, eta, Cv, gamma):
    return pow(rho, gamma - 1.0) * exp(eta / Cv) / (gamma - 1.0) / Cv

@njit
def compute_u(rho, eta, Cv, gamma):
    return pow(rho, gamma - 1.0) * exp(eta / Cv) / (gamma - 1.0)


@njit(parallel=True, cache=True)
def _dHdx_ce_mlp(dens, M, B, u, ntracers, ncells, dim, cellareas, dn_to_p0, Cv, gamma):
    for i in prange(ncells):

        #THIS IS LOWEST ORDER HODGE STAR!!!
        cellarea = cellareas[i]
        rho0 = dens[i,0] / cellarea
        eta = dens[i,1] / cellarea / rho0

        int_energy = compute_u(rho0, eta, Cv, gamma)
        dudrho = compute_dudrho(rho0, eta, Cv, gamma)
        dudeta = compute_dudeta(rho0, eta, Cv, gamma)

        k = _compute_k_u(i, dim, u, dn_to_p0, M, cellarea, rho0)

        B[dn_to_p0[i],0] = int_energy + rho0 * dudrho - eta * dudeta - k
        B[dn_to_p0[i],1] = dudeta

@njit(parallel=True, cache=True)
def _bnd_dHdx_ce_mlp(densbnd, Mbnd, Bbnd, ubnd, hbnd, bedges, ndim, ntracers, Cv, gamma):
    for i in prange(bedges.shape[0]):
        be = bedges[i]
        rho = densbnd[be,0]
        eta = densbnd[be,1] / rho

        int_energy = compute_u(rho, eta, Cv, gamma)
        dudrho = compute_dudrho(rho, eta, Cv, gamma)
        dudeta = compute_dudeta(rho, eta, Cv, gamma)

        ke = 0.0
        for d in range(ndim):
            u = Mbnd[be,d] / rho
            ke += u*u / 2.0
            ubnd[be,d] = u

        hbnd[be] = rho * ke
        Bbnd[be,0] = int_energy + rho * dudrho - eta * dudeta - ke
        Bbnd[be,1] = dudeta
        hbnd[be] += rho * int_energy

@njit
def _energy_ce_mlp(dens, M, energies, ntracers, ncells, dim, cellareas, ind, Cv, gamma):
    for i in range(ncells):

        #THIS IS LOWEST ORDER HODGE STAR!!!
        cellarea = cellareas[i]
        rho0 = dens[i,0] / cellarea
        eta = dens[i,1] / cellarea / rho0

        ke2 = _compute_M_ke2(i, dim, M, cellarea, rho0)
        int_energy = compute_u(rho0, eta, Cv, gamma)

        energies[ind,0] += ke2 + dens[i,0] * int_energy
        energies[ind,1] += ke2
        energies[ind,2] += dens[i,0] * int_energy

@njit(parallel=True, cache=True)
def _pointwise_energy_ce_mlp(dens, M, h, ntracers, ncells, dim, cellareas, Cv, gamma):
    for i in prange(ncells):

        #THIS IS LOWEST ORDER HODGE STAR!!!
        cellarea = cellareas[i]
        rho0 = dens[i,0] / cellarea
        eta = dens[i,1] / cellarea / rho0

        ke2 = _compute_M_ke2(i, dim, M, cellarea, rho0)
        int_energy = compute_u(rho0, eta, Cv, gamma)

        h[i] = ke2 + dens[i,0] * int_energy
