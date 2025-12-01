
from DecLib import KForm
from DecLib import CovariantExteriorDerivativeVForm, ExteriorDerivativeVForm
from DecLib import VolumeFormRecon, InteriorProductMLP
from DecLib import ADD_MODE

from math import exp, log, pow

from numba import njit, prange
import numpy as np

from dynamics import Dynamics, Statistics, Diagnostics, Statistic
from thermodynamics import getThermo
from hamiltonians import getHamiltonian
from operators import getRegularization


#three options here: constant, alpha-based, and Rusanov-type
#IMPLEMENT ALPHA AND RUSANOV VARIANTS!
#@njit(cache=True)
#def _compute_epsilon(densalphas, Malphas, epsilon, nedges, epsilon_type, epsilon_const):
#    if epsilon_type == 'const':
#        for e in prange(nedges):
#            epsilon[e] = epsilon_const
#    elif epsilon_type == 'rusanov':
#        pass
#    elif epsilon_type == 'alpha':
#        pass

@njit(cache=True)
def _compute_cell_fluxes(dens, B, M, u, h, ndim, ndofs):
    densflux = np.zeros((ndim, ndofs))
    Mflux = np.zeros((ndim, ndim))
    Eflux = np.zeros(ndim)
    for d in range(ndim):
        for l in range(ndofs):
            densflux[d,l] = dens[l] * u[d]

        #compute generalized pressure
        p = 0.0
        for d2 in range(ndim):
            p += u[d2] * M[d2]
        for l in range(ndofs):
            p += B[l] * dens[l]
        p -= h

        for d1 in range(ndim):
            Mflux[d,d1] = u[d1] * M[d]
        Mflux[d, d] += p

        Eflux[d] = u[d] * (h + p)

    return densflux, Mflux, Eflux

@njit(cache=True)
def _compute_alpha(denscellflux0, denscellflux1, Mcellflux0, Mcellflux1, Ecellflux0, Ecellflux1, densFtilde, MFtilde, B0, B1, u0, u1, edge_normal, edgeorient, ndim, ndofs):
    alpha = 0.0
    Bdiff = B0 * edgeorient[0] + B1 * edgeorient[1]
    udiff = u0 * edgeorient[0] + u1 * edgeorient[1]
    denom = 0.0
    for l in range(ndofs):
        denom += Bdiff[l] * Bdiff[l]
    for d1 in range(ndim):
        denom += udiff[d1] * udiff[d1]
    if denom > 0.0:
        Ediff = Ecellflux0 * edgeorient[0] + Ecellflux1 * edgeorient[1]
        for d in range(ndim):
            alpha += Ediff[d] * edge_normal[d]
            for l in range(ndofs):
                alpha += densFtilde[d,l] * Bdiff[l] * edge_normal[d]
                alpha -= (denscellflux0[d,l] * B0[l] * edgeorient[0] + denscellflux1[d,l] * B1[l] * edgeorient[1]) * edge_normal[d]
            for d1 in range(ndim):
                alpha += MFtilde[d,d1] * udiff[d1] * edge_normal[d]
                alpha -= (Mcellflux0[d,d1] * u0[d1] * edgeorient[0] + Mcellflux1[d,d1] * u1[d1] * edgeorient[1]) * edge_normal[d]
        alpha /= denom
    return alpha


# @njit(cache=True)
# def _compute_epsilon(epsilon_type, epsilon_param):
# #ADD MORE OPTIONS HERE!
#     if epsilon_type=='const':
#         eps = epsilon_param
#     elif epsilon_type == 'rusanov':
#         pass
#     elif epsilon_type == 'alpha':
#         pass
#     return eps


#STRAIGHT WENO ON FLUXES IS BAD HERE
#NEED TO BE MORE CLEVER IN FTILDE CHOICE!
#THERE IS A LOT OF LITERATURE ON THIS, NEED TO INVESTIGATE MORE!
#there is tons of literature about weno-based finite-volume schemes; also various flux splitting options

#Can we further do scale scalings/adjustments to support local bounds preservation and/or flux limiting?
#YES, SOME SORT OF FCT-TYPE LIMITERS, AND/OR INVARIANT DOMAIN PRESERVING LIMITERS
#HOW DOES THIS INTERACT WITH ALPHA CALCULATION?
#Does this help with T spike in RP3 (123 problem)?


@njit(cache=True)
def _compute_Ftilde(flux0, flux1):
    return (flux0 + flux1)/2.0

@njit(parallel=True, cache=True)
def _compute_edge_fluxes(dens, B, M, u, h, densflux, Eflux, Mflux, epsilon, \
dn_to_p0, CE, edgeorients, edge_lengths, primal_edge_lengths, edge_normals, cellareas, bedges, \
nedges, ndofs, ndim):
    for e in prange(nedges):
        c0 = CE[e,0]
        c1 = CE[e,1]
        v0 = dn_to_p0[c0]
        v1 = dn_to_p0[c1]
        edgeorient = edgeorients[e,:]
        edge_length = edge_lengths[e]
        primal_edge_length = primal_edge_lengths[e]
        cellarea0 = cellareas[c0]
        cellarea1 = cellareas[c1]
        edge_normal = edge_normals[e,:]

        dens0 = dens[c0,:]/cellarea0
        dens1 = dens[c1,:]/cellarea1
        B0 = B[v0,:]
        B1 = B[v1,:]
        M0 = M[c0,:]/cellarea0
        M1 = M[c1,:]/cellarea1
        u0 = u[v0,:]
        u1 = u[v1,:]
        h0 = h[c0]/cellarea0
        h1 = h[c1]/cellarea1

        #compute cell fluxes
        denscellflux0, Mcellflux0, Ecellflux0 = _compute_cell_fluxes(dens0, B0, M0, u0, h0, ndim, ndofs)
        denscellflux1, Mcellflux1, Ecellflux1 = _compute_cell_fluxes(dens1, B1, M1, u1, h1, ndim, ndofs)

        #compute Ftilde
        densFtilde = _compute_Ftilde(denscellflux0, denscellflux1)
        MFtilde = _compute_Ftilde(Mcellflux0, Mcellflux1)
        EFtilde = _compute_Ftilde(Ecellflux0, Ecellflux1)

        #compute alpha
        alpha = _compute_alpha(denscellflux0, denscellflux1, Mcellflux0, Mcellflux1, Ecellflux0, Ecellflux1, densFtilde, MFtilde, B0, B1, u0, u1, \
        edge_normal, edgeorient, ndim, ndofs)

        #compute inviscid flux
        for l in range(ndofs):
            temp = 0.0
            for d in range(ndim):
                temp += densFtilde[d,l] * edge_normal[d]
            densflux[e,l] += (temp - alpha * (B0[l] * edgeorient[0] + B1[l] * edgeorient[1])) * edge_length

        for d1 in range(ndim):
            temp = 0.0
            for d in range(ndim):
                temp += MFtilde[d,d1] * edge_normal[d]
            Mflux[e,d1] += (temp - alpha * (u0[d1] * edgeorient[0] + u1[d1] * edgeorient[1])) * edge_length

        temp = 0.0
        for d in range(ndim):
            temp += EFtilde[d] * edge_normal[d]
        Eflux[e] = temp * edge_length

        # #compute epsilon
        # eps = _compute_epsilon(epsilon_type, epsilon_param)
        # epsilon[e] = eps
        #
        #compute viscous flux
        # dens_jump = (dens0 * edgeorient[0] + dens1 * edgeorient[1]) / primal_edge_length
        # M_jump = (M0 * edgeorient[0] + M1 * edgeorient[1]) / primal_edge_length
        E_jump = (h0 * edgeorient[0] + h1 * edgeorient[1]) / primal_edge_length
        #
        # densflux[e,:] += dens_jump * eps * edge_length
        # Mflux[e,:] += M_jump * eps * edge_length
        Eflux[e] += E_jump * epsilon[e] * edge_length

@njit(parallel=True, cache=True)
def _compute_boundary_edge_fluxes(dens, B, M, u, h, densfluxarr, Efluxarr, Mfluxarr, bedges, edge_normals, edge_lengths, ndim, ndofs):

    for m in prange(bedges.shape[0]):
        densflux = np.zeros((ndim, ndofs))
        Mflux = np.zeros((ndim, ndim))
        Eflux = np.zeros(ndim)

        be = bedges[m]
        densbnd = dens[be,:]
        Mbnd = M[be,:]
        ubnd = u[be,:]
        hbnd = h[be]
        Bbnd = B[be,:]
        edge_normal = edge_normals[be,:]
        edge_length = edge_lengths[be]

        p = 0.0
        for d2 in range(ndim):
            p += ubnd[d2] * Mbnd[d2]
        for l in range(ndofs):
            p += Bbnd[l] * densbnd[l]
        p -= hbnd

        for d in range(ndim):

            for l in range(ndofs):
                densflux[d,l] = densbnd[l] * ubnd[d]

            for d1 in range(ndim):
                Mflux[d,d1] = ubnd[d1] * Mbnd[d]
            Mflux[d,d] += p

            Eflux[d] = ubnd[d] * (hbnd + p)

        for l in range(ndofs):
            dflux = 0.0
            for d in range(ndim):
                dflux += densflux[d,l] * edge_normal[d]
            densfluxarr[be,l] = dflux  * edge_length

        for d in range(ndim):
            mflux = 0.0
            for d1 in range(ndim):
                mflux += Mflux[d1,d] * edge_normal[d1]
            Mfluxarr[be,d] = mflux * edge_length

        eflux = 0.0
        for d in range(ndim):
            eflux += Eflux[d] * edge_normal[d]
        Efluxarr[be] = eflux  * edge_length
        #print(m, be, edge_length, edge_normal, Mfluxarr)

                # #ADD VISCOUS FLUXES
            #THIS IS ANOTHER TYPE OF BC TO SET
            #FOR NOW WE JUST ASSUME GRAD M = GRAD D = 0 at the boundaries


class AdvDensConsDynamics(Dynamics):
    def __init__(self, meshes, params, ic, construct=True):

        thermo = getThermo(params, ic)
        hamiltonian = getHamiltonian(params, meshes, thermo, construct=construct)

        self.denslist = hamiltonian.denslist
        self.hamiltonian = hamiltonian
        self.scaledof = hamiltonian.scaledof
        self.viscosity = getRegularization(params, self.hamiltonian, meshes, self.hamiltonian.entropydof, len(self.denslist), construct=construct)

        if construct:
            Dynamics.__init__(self, meshes, params, ic)

            self.prog_vars['dens'] = KForm(meshes.dim, self.meshes.dtopo, meshes.dRBundle, 'dens', create_petsc=True, ndofs = len(self.denslist))
            self.prog_vars['M'] = KForm(meshes.dim, self.meshes.dtopo, meshes.dCTBundle, 'M', create_petsc=True)
            self.prog_vars['E'] = KForm(meshes.dim, self.meshes.dtopo, meshes.dRBundle, 'E', create_petsc=True)

            self.aux_vars['dens_flux'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'dens_flux', create_petsc=True, ndofs = len(self.denslist))
            self.aux_vars['M_flux'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dCTBundle, 'M_flux', create_petsc=True)
            self.aux_vars['E_flux'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'E_flux', create_petsc=True)

            self.aux_vars['epsilon'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'epsilon', create_petsc=True)
            self.aux_vars['alpha'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'alpha', create_petsc=True)

            self.aux_vars['u'] = KForm(meshes.dim, self.meshes.dtopo, meshes.dTBundle, 'u', create_petsc=True)
            self.aux_vars['B'] = KForm(meshes.dim, self.meshes.dtopo, meshes.dRBundle, 'B', create_petsc=True, ndofs = len(self.denslist))

            self.aux_vars['h'] = KForm(meshes.dim, self.meshes.dtopo, meshes.dRBundle, 'h', create_petsc=True)

            self.aux_vars['dens_source'] = KForm(meshes.dim, self.meshes.dtopo, meshes.dRBundle, 'dens_source', create_petsc=True, ndofs = len(self.denslist))

            #EVENTUALLY THESE SHOULD BE SIZE B
            #REQUIRES NEW DECLIB CAPABILITIES
            self.aux_vars['hbound'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'hbound', create_petsc=True)
            self.aux_vars['Mbound'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dCTBundle, 'Mbound', create_petsc=True)
            self.aux_vars['ubound'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dTBundle, 'ubound', create_petsc=True)
            self.aux_vars['densbound'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'densbound', create_petsc=True, ndofs = len(self.denslist))
            self.aux_vars['Bbound'] = KForm(meshes.dim-1, self.meshes.dtopo, meshes.dRBundle, 'Bbound', create_petsc=True, ndofs = len(self.denslist))

            print('created dynamics vars')

            self.m_covextderiv = CovariantExteriorDerivativeVForm(meshes)
            self.dens_covextderiv = ExteriorDerivativeVForm(meshes)
            self.E_covextderiv = ExteriorDerivativeVForm(meshes)

            self.cellareas = np.zeros(self.meshes.dtopo.nkcells[self.meshes.dim])
            self.edge_normals = np.zeros((self.meshes.dtopo.nkcells[self.meshes.dim-1], self.meshes.dim))
            self.edge_orients = np.zeros((self.meshes.dtopo.nkcells[self.meshes.dim-1], 2), dtype=np.int32)
            self.edge_lengths = np.zeros(self.meshes.dtopo.nkcells[self.meshes.dim-1])
            self.primal_edge_lengths = np.zeros(self.meshes.dtopo.nkcells[self.meshes.dim-1])
            self.CE = np.zeros((self.meshes.dtopo.nkcells[self.meshes.dim-1], 2), dtype=np.int32)
            self.bedges = np.zeros(self.meshes.dtopo.nbkcells[self.meshes.dim-1], dtype=np.int32)
            self.dn_to_p0 = np.zeros(self.meshes.dtopo.nkcells[self.meshes.dim], dtype=np.int32)

            maxne = meshes.dtopo.petscmesh.getMaxSizes()[0]

            self.EC = np.zeros((self.meshes.dtopo.nkcells[self.meshes.dim], maxne), dtype=np.int32)
            self.nEC = np.zeros(self.meshes.dtopo.nkcells[self.meshes.dim], dtype=np.int32)
            self.bnd = np.full(self.meshes.dtopo.nkcells[self.meshes.dim-1], False, dtype=np.bool_)

            #THIS IS AN UGLY HACK- SHOULD REALLY BE BASED ON A BASIS CHOICE FOR AN EDGE...
            #ALSO ASSUMES THAT ALL QUADRATURE POINTS SHARE A BASIS- WHICH IS NOT NECESSARILY TRUE...
            self.edge_normals[:,0] = 1.0

            dcoff = meshes.dtopo.kcells_off[meshes.dim]
            deoff = meshes.dtopo.kcells_off[meshes.dim - 1]
            peoff = meshes.ptopo.kcells_off[1]

            for dc in range(meshes.dtopo.kcells[meshes.dim][0], meshes.dtopo.kcells[meshes.dim][1]):
                self.cellareas[dc - dcoff] = meshes.dgeom.entitysizes[meshes.dim][dc - dcoff]
                edges = meshes.dtopo.lower_dim_TC(dc, self.dim-1)
                nedges = edges.shape[0]
                self.EC[dc - dcoff, :nedges] = edges - deoff
                self.nEC[dc - dcoff] = nedges
                self.dn_to_p0[dc - dcoff] = meshes.pdmapping.dinmk_to_pk(dc) - meshes.ptopo.kcells_off[0]

            for de in range(meshes.dtopo.kcells[meshes.dim-1][0], meshes.dtopo.kcells[meshes.dim-1][1] - meshes.dtopo.nbkcells[meshes.dim-1]):
                pe = meshes.pdmapping.dinmk_to_pk(de)
                self.CE[de - deoff, :] = meshes.dtopo.higher_dim_TC(de, meshes.dim) - dcoff
                self.edge_orients[de - deoff, :] = meshes.dtopo.higher_orientation(de)
                self.primal_edge_lengths[de - deoff] = meshes.pgeom.entitysizes[1][pe - peoff]

            for de in range(meshes.dtopo.kcells[meshes.dim-1][0], meshes.dtopo.kcells[meshes.dim-1][1]):
                self.edge_lengths[de - deoff] = meshes.dgeom.entitysizes[meshes.dim-1][de - deoff]

            if meshes.dtopo.has_boundary:
                self.bedges[:] = meshes.dtopo.bkcells[meshes.dim-1]
                for de in range(meshes.dtopo.kcells[meshes.dim-1][0], meshes.dtopo.kcells[meshes.dim-1][1]):
                    self.bnd[de - deoff] = (meshes.dtopo.petscmesh.getLabelValue('bnd', de) == 1)

#THERE IS PROBABLY A LOGICAL SPLITTING OF THIS INTO DHDX AND J CALCS, TO EVENTUALLY SUPPORT EC2...
#ALSO THERE IS A BRACKET HIDING HERE!!!
    def set_IC(self):
        Dynamics.set_IC(self)
        self.hamiltonian.compute_pointwise_energy(self.prog_vars, self.aux_vars, self.prog_vars['E'])

    def pre_step(self):
        self.viscosity.pre_step(self.aux_vars)

    def post_step(self):
        self.viscosity.post_step(self.aux_vars)

    def compute_aux(self, state, t, dt):

        #compute p = B, u; and h
        self.hamiltonian.compute_dHdx(state, self.aux_vars, do_in_dn=True)
        self.hamiltonian.compute_pointwise_energy(state, self.aux_vars, self.aux_vars['h'])

        #extract needed arrays
        Barr = self.aux_vars['B'].petsc_vec.getArray()
        Barr = Barr.reshape((self.aux_vars['B'].nelems, self.aux_vars['B'].ndofs))
        densarr = state['dens'].petsc_vec.getArray()
        densarr = densarr.reshape((state['dens'].nelems, state['dens'].ndofs))
        uarr = self.aux_vars['u'].petsc_vec.getArray()
        uarr = uarr.reshape((self.aux_vars['u'].nelems, self.aux_vars['u'].bsize))
        Marr = state['M'].petsc_vec.getArray()
        Marr = Marr.reshape((state['M'].nelems, state['M'].bsize))
        harr = self.aux_vars['h'].petsc_vec.getArray()

        densfluxarr = self.aux_vars['dens_flux'].petsc_vec.getArray()
        densfluxarr = densfluxarr.reshape((self.aux_vars['dens_flux'].nelems, self.aux_vars['dens_flux'].ndofs))
        Efluxarr = self.aux_vars['E_flux'].petsc_vec.getArray()
        Mfluxarr = self.aux_vars['M_flux'].petsc_vec.getArray()
        Mfluxarr = Mfluxarr.reshape((self.aux_vars['M_flux'].nelems, self.aux_vars['M_flux'].bsize))

        denssourcearr = self.aux_vars['dens_source'].petsc_vec.getArray()
        denssourcearr = denssourcearr.reshape((self.aux_vars['dens_source'].nelems, self.aux_vars['dens_source'].ndofs))

        epsilonarr = self.aux_vars['epsilon'].petsc_vec.getArray()

        Bboundarr = self.aux_vars['Bbound'].petsc_vec.getArray()
        Bboundarr = Bboundarr.reshape((self.aux_vars['Bbound'].nelems, self.aux_vars['Bbound'].ndofs))
        densboundarr = self.aux_vars['densbound'].petsc_vec.getArray()
        densboundarr = densboundarr.reshape((self.aux_vars['densbound'].nelems, self.aux_vars['densbound'].ndofs))
        uboundarr = self.aux_vars['ubound'].petsc_vec.getArray()
        uboundarr = uboundarr.reshape((self.aux_vars['ubound'].nelems, self.aux_vars['ubound'].bsize))
        Mboundarr = self.aux_vars['Mbound'].petsc_vec.getArray()
        Mboundarr = Mboundarr.reshape((self.aux_vars['Mbound'].nelems, self.aux_vars['Mbound'].bsize))
        hboundarr = self.aux_vars['hbound'].petsc_vec.getArray()

        #set boundary dens/M values
        self.IC.set_bnd_vars(self.aux_vars, t)
        self.hamiltonian.set_dHdx_bound(self.aux_vars, t)

#SPLIT INTO INVISCID AND REGULARIZATION
        #compute viscous fluxes
        self.viscosity.compute_fluxes(state, self.aux_vars)

        #compute inviscid edge fluxes
        _compute_edge_fluxes(densarr, Barr, Marr, uarr, harr, densfluxarr, Efluxarr, Mfluxarr, epsilonarr, \
        self.dn_to_p0, self.CE, self.edge_orients, self.edge_lengths, self.primal_edge_lengths, self.edge_normals, self.cellareas, self.bedges, \
        self.meshes.dtopo.nkcells[self.meshes.dim-1] - self.meshes.dtopo.nbkcells[self.meshes.dim-1], state['dens'].ndofs, self.meshes.dim)

        _compute_boundary_edge_fluxes(densboundarr, Bboundarr, Mboundarr, uboundarr, hboundarr, densfluxarr, Efluxarr, Mfluxarr, \
        self.meshes.dtopo.bkcells[self.meshes.dim-1], self.edge_normals, self.edge_lengths, self.meshes.dim, state['dens'].ndofs)

        #compute entropy production term
#        _compute_entropy_source(densarr, Marr, Barr, uarr, denssourcearr, epsilonarr, \
#        self.EC, self.nEC, self.CE, self.dn_to_p0, self.edge_orients, self.edge_lengths, self.primal_edge_lengths, self.cellareas, self.bnd, \
#        self.meshes.dtopo.nkcells[self.meshes.dim], self.meshes.dim, state['dens'].ndofs, 1)

        #print('denssourcearr',denssourcearr)

    def compute_rhs(self, rhs, t, dt):
        #compute M tend
        self.m_covextderiv.apply(self.aux_vars['M_flux'], rhs['M'])

		#compute dens tend
        self.dens_covextderiv.apply(self.aux_vars['dens_flux'], rhs['dens'])
        rhs['dens'].petsc_vec.axpy(-1.0, self.aux_vars['dens_source'].petsc_vec)

		#compute E tend
        self.E_covextderiv.apply(self.aux_vars['E_flux'], rhs['E'])

class AdvDensConsDiagnostics(Diagnostics):
    def __init__(self, dyn, construct=True):
        self.dyn = dyn
        self.diaglist = ['M0', 'u', 'E']
        for hamildiag in self.dyn.hamiltonian.diaglist:
            self.diaglist.append(hamildiag)
        for viscdiag in self.dyn.viscosity.diaglist:
            self.diaglist.append(viscdiag)

        if (construct):
            self.vars = {}
            self.vars['dens0'] = KForm(0, dyn.meshes.ptopo, dyn.meshes.pRBundle, 'dens0', create_petsc=True, ndofs = len(dyn.denslist))
            self.vars['scalar0'] = KForm(0, dyn.meshes.ptopo, dyn.meshes.pRBundle, 'scalar0', create_petsc=True, ndofs = len(dyn.denslist))
            self.dyn.hamiltonian.add_diagnostic_vars(self.vars)
            self.dyn.viscosity.add_diagnostic_vars(self.vars)
            self.vars['M0'] = KForm(0, dyn.meshes.ptopo, dyn.meshes.pTBundle, 'M0', create_petsc=True)
            self.vars['u'] = KForm(0, dyn.meshes.ptopo, dyn.meshes.pTBundle, 'u', create_petsc=True)
            self.vars['E'] = KForm(0, dyn.meshes.ptopo, dyn.meshes.pRBundle, 'E', create_petsc=True)

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

        Earr = self.dyn.prog_vars['E'].petsc_vec.getArray()
        E0arr = self.vars['E'].petsc_vec.getArray()

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
            E0arr[dn] = Earr[dn] / cellarea

        #DO WE NEED THESE ASSEMBLES?
        for k,v in self.vars.items():
            v.petsc_vec.assemble()



class AdvDensConsStatistics(Statistics):
    def __init__(self, dyn, compute_bnd_fluxes, construct=True):
        self.dyn = dyn
        self.stats = {}
        self.vars = {}

        self.energylist = dyn.hamiltonian.energylist

        if construct:
            self.stats['energies'] = Statistic('energies', dyn.params['Nsteps'], dyn.params['nstat'], ndofs=len(dyn.hamiltonian.energylist))
            self.stats['dens_total'] = Statistic('dens_total', dyn.params['Nsteps'], dyn.params['nstat'], ndofs=len(self.dyn.denslist))
            self.stats['E_total'] = Statistic('E_total', dyn.params['Nsteps'], dyn.params['nstat'])
            self.stats['M_total'] =  Statistic('M_total', dyn.params['Nsteps'], dyn.params['nstat'], ndofs = self.dyn.meshes.dim)

            if self.dyn.meshes.dtopo.has_boundary and compute_bnd_fluxes:
                self.stats['mass_fluxes'] = Statistic('mass_fluxes', dyn.params['Nsteps'], 1, ndofs=len(self.dyn.denslist))
                self.stats['energy_flux'] = Statistic('energy_flux', dyn.params['Nsteps'], 1, ndofs=1)
                self.stats['E_flux'] = Statistic('E_flux', dyn.params['Nsteps'], 1, ndofs=1)
                self.stats['momentum_flux'] = Statistic('momentum_flux', dyn.params['Nsteps'], 1, ndofs=1)


#BROKEN FOR MPI- NEEDS A REDUCTION!
    def compute_bnd_fluxes(self, alpha, k, t, dt):
        if self.dyn.meshes.dtopo.has_boundary:

            massflux_arr = self.stats['mass_fluxes'].petsc_vec.getArray()
            massflux_arr = massflux_arr.reshape((self.stats['mass_fluxes'].statsize, self.stats['mass_fluxes'].ndofs))
            energyflux_arr = self.stats['energy_flux'].petsc_vec.getArray()
            Eflux_arr = self.stats['E_flux'].petsc_vec.getArray()
            momflux_arr = self.stats['momentum_flux'].petsc_vec.getArray()

            densfluxarr = self.dyn.aux_vars['dens_flux'].petsc_vec.getArray()
            densfluxarr = densfluxarr.reshape((self.dyn.aux_vars['dens_flux'].nelems, self.dyn.aux_vars['dens_flux'].ndofs))
            Efluxarr = self.dyn.aux_vars['E_flux'].petsc_vec.getArray()
            Mfluxarr = self.dyn.aux_vars['M_flux'].petsc_vec.getArray()
            Mfluxarr = Mfluxarr.reshape((self.dyn.aux_vars['M_flux'].nelems, self.dyn.aux_vars['M_flux'].bsize))

            doffset = self.dyn.meshes.dtopo.kcells_off[self.dyn.meshes.dim-1]
            poffset = self.dyn.meshes.ptopo.kcells_off[0]

            mflux = np.zeros(self.stats['mass_fluxes'].ndofs)
            eflux = 0.0
            momflux = 0.0
            Eflux = 0.0

            #JIT THIS
            for dbs in self.dyn.meshes.dtopo.bkcells[self.dyn.meshes.dim-1]:
                pbe = self.dyn.meshes.pdmapping.dbnmk_to_pbk(dbs)
                orient = self.dyn.meshes.dtopo.higher_orientation(dbs)[0]
                for l in range(self.stats['mass_fluxes'].ndofs):
                    mflux[l] += orient * densfluxarr[dbs - doffset, l]
#FIX THIS
#                   eflux += orient * densfluxarr[dbs - doffset, l] * XXX
                for d in range(self.dyn.meshes.dim):
                    momflux += orient * Mfluxarr[dbs - doffset, d]
#                   eflux += XXX
                Eflux += orient * Efluxarr[dbs - doffset]

#This covers both EC2 (1 call with alpha=1) and s-stage RK (s calls with alpha=bs)
            energyflux_arr[k] += alpha * dt * eflux
            Eflux_arr[k] += alpha * dt * Eflux
            momflux_arr[k] += alpha * dt * momflux
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
        Estats = self.stats['E_total'].petsc_vec.getArray()
        Earr = self.dyn.prog_vars['E'].petsc_vec.getArray()

        dn_off = self.dyn.meshes.dtopo.kcells_off[self.dyn.meshes.dim]
        for dn in range(self.dyn.meshes.dtopo.kcells[self.dyn.meshes.dim][0] - dn_off, self.dyn.meshes.dtopo.kcells[self.dyn.meshes.dim][1] - dn_off):
            for l in range(len(self.dyn.denslist)):
                dens_stat_arr[ind,l] += densarr[dn,l]
            for i in range(self.dyn.meshes.dim):
                Mstats[ind,i] += Marr[dn,i]
            Estats[ind] += Earr[dn]

        #DO WE NEED THESE ASSEMBLES?
        for k,v in self.stats.items():
            v.petsc_vec.assemble()
