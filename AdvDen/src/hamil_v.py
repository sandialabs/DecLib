
from numba import njit, prange
import numpy as np
from math import exp, log, pow
from DecLib import KForm

class HamiltonianV():
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
            self.ncells = meshes.dtopo.nkcells[meshes.dim]
            self.cellareas = np.zeros(self.ncells)
            for i in range(self.ncells):
                self.cellareas[i] = meshes.dgeom.get_entity_size(meshes.dim, i)
        
            self.nedges = meshes.dtopo.nkcells[meshes.dim-1]
            self.nbedges = meshes.dtopo.nbkcells[meshes.dim-1]
            self.maxne = meshes.dtopo.petscmesh.getMaxSizes()[0]
            self.EC = np.zeros((self.ncells, self.maxne), dtype=np.int32)
            self.nEC = np.zeros(self.ncells, dtype=np.int32)
            self.dn_to_p0 = np.zeros(self.ncells, dtype=np.int32)
            self.de_to_pe = np.zeros(self.nedges, dtype=np.int32)
            self.dedgelengths = np.zeros(self.nedges)
            self.pedgelengths = np.zeros(self.nedges)
            self.CE = np.zeros((self.nedges, 2), dtype=np.int32)

            dnoff = meshes.dtopo.kcells_off[meshes.dim]
            deoff = meshes.dtopo.kcells_off[meshes.dim - 1]
            p0off = meshes.ptopo.kcells_off[0]
            for dn in range(meshes.dtopo.kcells[meshes.dim][0], meshes.dtopo.kcells[meshes.dim][1]):
                edges = meshes.dtopo.lower_dim_TC(dn, meshes.dim-1)
                tEC = list(edges)
                if meshes.dtopo.has_boundary and meshes.dtopo.petscmesh.getLabelValue('bndcell', dn) == 1:
                    tEC = []
                    for e in edges:
                        if not meshes.dtopo.petscmesh.getLabelValue('bnd', e) == 1: tEC.append(e)
                nedges = len(tEC)
                self.EC[dn - dnoff, :nedges] = tEC - deoff
                self.nEC[dn - dnoff] = nedges
                self.dn_to_p0[dn - dnoff] = meshes.pdmapping.dinmk_to_pk(dn) - p0off

            peoff = meshes.ptopo.kcells_off[1]
            for de in range(meshes.dtopo.kcells[meshes.dim-1][0], meshes.dtopo.kcells[meshes.dim -1][1] - self.nbedges):
                pe = meshes.pdmapping.dinmk_to_pk(de)
                self.dedgelengths[de - deoff] = meshes.dgeom.get_entity_size(meshes.dim -1, de - deoff)
                self.pedgelengths[pe - peoff] = meshes.pgeom.get_entity_size(1, pe- peoff)
                self.de_to_pe[de - deoff] = pe - peoff
                cells = meshes.dtopo.higher_dim_TC(de, meshes.dim)
                self.CE[de - deoff] = cells - dnoff    
    
    def add_diagnostic_vars(self, dvars):
        pass

    def compute_diagnostic_vars(self, prog_vars, dvars):
        pass
        
#THESE ARE MISSING HS AND GEOP TERMS...
    def compute_dHdx(self, state, dHdx):
        densarr = state['dens'].petsc_vec.getArray()
        densarr = densarr.reshape((state['dens'].nelems, state['dens'].ndofs))
        varr = state['v'].petsc_vec.getArray()
        
        Barr = dHdx['B'].petsc_vec.getArray()
        Barr = Barr.reshape((dHdx['B'].nelems, dHdx['B'].ndofs))
        Farr = dHdx['F'].petsc_vec.getArray()
        self._dHdx(densarr, varr, Barr, Farr, self.ntracers, self.ncells, self.nedges - self.nbedges, self.EC, self.nEC, self.CE, self.de_to_pe, self.cellareas, self.pedgelengths, self.dedgelengths, self.dn_to_p0, *self.dHdx_params)


    def compute_energy(self, state, stats, ind):

        densarr = state['dens'].petsc_vec.getArray()
        densarr = densarr.reshape((state['dens'].nelems, state['dens'].ndofs))
        varr = state['v'].petsc_vec.getArray()

        energyarr = stats['energies'].petsc_vec.getArray()
        energyarr = energyarr.reshape((stats['energies'].statsize, stats['energies'].ndofs))

        self._energy(densarr, varr, energyarr, self.ntracers, self.ncells, self.nedges, self.EC, self.nEC, self.de_to_pe, self.cellareas, self.pedgelengths, self.dedgelengths, ind, *self.energy_params)


class TSWE_V(HamiltonianV):
    def __init__(self, meshes, params, hs=None, R=None, construct=True):
        HamiltonianV.__init__(self, meshes, params, construct=construct)
        self.energylist = ['TE', 'KE', 'PE']
        self.hs = hs
        self.R = R
        self.name = 'tswe'
        
        self._dHdx = _dHdx_tswe_v
        self._energy = _energy_tswe_v

        self.energy_params = []
        self.dHdx_params = []


#THIS IS ACTUALLY SPECIFIC TO Abgrall form of internal energy ie based on rho and eta instead of alpha and eta...
class CompressibleEuler_V(HamiltonianV):
    def __init__(self, meshes, params, thermo, R = None, geop = None, construct=True):
        HamiltonianV.__init__(self, meshes, params, construct=construct)

        self.diaglist = ['p','T','inte']
        self.energylist = ['TE', 'KE', 'IE']
        self.geop = geop
        self.R = R
        self.thermo = thermo
        if not (geop is None):
            self.energylist.append('PE')
        self.name = 'ce'

        self._dHdx = _dHdx_ce_v
        self._energy = _energy_ce_v
        
        self.dHdx_params = [self.thermo.Cv, self.thermo.gamma]
        self.energy_params = [self.thermo.Cv, self.thermo.gamma]
        
#NEED TO FIGURE OUT HOW TO JIT CLASSES...
    def add_diagnostic_vars(self, dvars):
        dvars['p'] = KForm(0, self.meshes.ptopo, self.meshes.pRBundle, 'p', create_petsc=True)
        dvars['T'] = KForm(0, self.meshes.ptopo, self.meshes.pRBundle, 'T', create_petsc=True)
        dvars['inte'] = KForm(0, self.meshes.ptopo, self.meshes.pRBundle, 'inte', create_petsc=True)

    def compute_diagnostic_vars(self, prog_vars, dvars):
        parr = dvars['p'].petsc_vec.getArray()
        Tarr = dvars['T'].petsc_vec.getArray()
        intearr = dvars['inte'].petsc_vec.getArray()
        densarr = prog_vars['dens'].petsc_vec.getArray()
        densarr = densarr.reshape((prog_vars['dens'].nelems, prog_vars['dens'].ndofs))
        
        dn_off = self.meshes.dtopo.kcells_off[self.meshes.dim]
        for dn in range(self.meshes.dtopo.kcells[self.meshes.dim][0] - dn_off, self.meshes.dtopo.kcells[self.meshes.dim][1] - dn_off):
            cellarea = self.meshes.dgeom.get_entity_size(self.meshes.dim, dn)

            rho0 = densarr[dn,0] / cellarea
            eta = densarr[dn,1] / cellarea / rho0
            parr[dn] = self.thermo.get_p(rho0, eta)
            Tarr[dn] = self.thermo.get_T(rho0, eta)
            intearr[dn] = self.thermo.compute_u(rho0, eta)

@njit
def _compute_v_ke2(i, v, EC, nEC, de_to_pe, dedgelengths, pedgelengths):
    #THIS IS LOWEST ORDER HODGE STAR AND TOPOLOGICAL PHI!!!

    ke2 = 0
    for j in range(nEC[i]):
        de = EC[i,j]
        pe = de_to_pe[de]
        U = dedgelengths[de] / pedgelengths[pe] * v[pe]
        ke2 += v[pe] * U / 2.0
    ke2 = ke2 / 2.0
    return ke2

@njit(parallel=True, cache=True)
def _compute_F(nedges, de_to_pe, dens, cellareas, v, dedgelengths, pedgelengths, F, CE):
    #THIS IS LOWEST ORDER HODGE STAR AND TOPOLOGICAL PHI!!!
    for de in prange(nedges):
        pe = de_to_pe[de]
        he = (dens[CE[de,0],0] / cellareas[CE[de,0]] + dens[CE[de,1],0] / cellareas[CE[de,1]])/2.
        U = dedgelengths[de] / pedgelengths[pe] * v[pe]
        F[de] = he * U

@njit(parallel=True, cache=True)
def _dHdx_tswe_v(dens, v, B, F, ntracers, ncells, nedges, EC, nEC, CE, de_to_pe, cellareas, pedgelengths, dedgelengths, dn_to_p0):
    for i in prange(ncells):

        #THIS IS LOWEST ORDER HODGE STAR AND TOPOLOGICAL PHI!!!
        cellarea = cellareas[i]
        rho0 = dens[i,0] / cellarea
        S0 = dens[i,1] / cellarea

        ke2 = _compute_v_ke2(i, v, EC, nEC, de_to_pe, dedgelengths, pedgelengths)
        
        B[dn_to_p0[i],0] = S0/2. + ke2/cellarea
        for t in range(ntracers):
            T0 = dens[i,2+t] / cellarea
            B[dn_to_p0[i],0] += T0/2.
            B[dn_to_p0[i],2+t] = rho0/2.
        B[dn_to_p0[i],1] = rho0/2.
        
    _compute_F(nedges, de_to_pe, dens, cellareas, v, dedgelengths, pedgelengths, F, CE)


@njit
def _energy_tswe_v(dens, v, energies, ntracers, ncells, nedges, EC, nEC, de_to_pe, cellareas, pedgelengths, dedgelengths, ind):
    for i in range(ncells):

        #THIS IS LOWEST ORDER HODGE STARS AND TOPOLOGICAL PHI!!!
        cellarea = cellareas[i]
        rho0 = dens[i,0] / cellarea

        ke2 = _compute_v_ke2(i, v, EC, nEC, de_to_pe, dedgelengths, pedgelengths)

        pot_energy = rho0 * dens[i,1] / 2.
        for t in range(ntracers):
            pot_energy += rho0 * dens[i,2+t] / 2.

        energies[ind,0] += rho0 * ke2 + pot_energy
        energies[ind,1] += rho0 * ke2
        energies[ind,2] += pot_energy   
                    
@njit(parallel=True, cache=True)
def _dHdx_ce_v(dens, v, B, F, ntracers, ncells, nedges, EC, nEC, CE, de_to_pe, cellareas, pedgelengths, dedgelengths, dn_to_p0, Cv, gamma):
    for i in prange(ncells):

        #THIS IS LOWEST ORDER HODGE STAR AND TOPOLOGICAL PHI!!!
        cellarea = cellareas[i]
        rho0 = dens[i,0] / cellarea
        eta = dens[i,1] / cellarea / rho0
        
        int_energy = compute_u(rho0, eta, Cv, gamma)
        dudrho = compute_dudrho(rho0, eta, Cv, gamma)
        dudeta = compute_dudeta(rho0, eta, Cv, gamma)

        ke2 = _compute_v_ke2(i, v, EC, nEC, de_to_pe, dedgelengths, pedgelengths)
        
        B[dn_to_p0[i],0] = int_energy + rho0 * dudrho - eta * dudeta + ke2 / cellarea
        B[dn_to_p0[i],1] = dudeta

    _compute_F(nedges, de_to_pe, dens, cellareas, v, dedgelengths, pedgelengths, F, CE)

#THESE ARE VERY THERMO SPECIFIC...
#IDEALLY THERMO CHOICE EXPOSES THESE PROPERLY!!!

@njit
def compute_dudrho(rho, eta, Cv, gamma):
    return pow(rho, gamma - 2.0) * exp(eta / Cv)

@njit        
def compute_dudeta(rho, eta, Cv, gamma):
    return Cv * pow(rho, gamma - 1.0) * exp(eta / Cv) / (gamma - 1.0)
        
@njit
def compute_u(rho, eta, Cv, gamma):
    return pow(rho, gamma - 1.0) * exp(eta / Cv) / (gamma - 1.0)

@njit
def _energy_ce_v(dens, v, energy, ntracers, ncells, nedges, EC, nEC, de_to_pe, cellareas, pedgelengths, dedgelengths, ind, Cv, gamma):
    for i in range(ncells):

        #THIS IS LOWEST ORDER HODGE STAR AND TOPOLOGICAL PHI!!!
        cellarea = cellareas[i]
        rho0 = dens[i,0] / cellarea
        eta = dens[i,1] / cellarea / rho0

        ke2 = _compute_v_ke2(i, v, EC, nEC, de_to_pe, dedgelengths, pedgelengths)

        int_energy = compute_u(rho0, eta, Cv, gamma)
        energy[ind,0] += rho0 * ke2 + dens[i,0] * int_energy
        energy[ind,1] += rho0 * ke2
        energy[ind,2] += dens[i,0] * int_energy
