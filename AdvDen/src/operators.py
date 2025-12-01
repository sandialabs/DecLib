from numba import njit, prange
import numpy as np
from DecLib import KForm
from math import exp, log, pow, sqrt

@njit(parallel=True, cache=True)
def _compute_edge_fluxes(dens, M, B, u, h, densflux, Mflux, epsilon, alpha, \
CE, neighbor_edges, dn_to_p0, edgeorients, edge_normals, edge_lengths, primal_edge_lengths, bedges, cellareas, \
nedges, ndim, ndofs, epsilon_type, epsilon_param, alphamin, alphamax):

    for e in prange(nedges):
        c0 = CE[e,0]
        c1 = CE[e,1]
        v0 = dn_to_p0[c0]
        v1 = dn_to_p0[c1]
        edgeorient = edgeorients[e,:]
        edge_normal = edge_normals[e,:]
        edge_length = edge_lengths[e]
        primal_edge_length = primal_edge_lengths[e]
        cellarea0 = cellareas[c0]
        cellarea1 = cellareas[c1]

        dens0 = dens[c0,:]/cellarea0
        dens1 = dens[c1,:]/cellarea1
        M0 = M[c0,:]/cellarea0
        M1 = M[c1,:]/cellarea1
        B0 = B[v0,:]
        B1 = B[v1,:]
        u0 = u[v0,:]
        u1 = u[v1,:]
        h0 = h[c0]/cellarea0
        h1 = h[c1]/cellarea1

        em = neighbor_edges[e,0]
        cm0 = CE[em,0]
        cm1 = CE[em,1]
        cellaream0 = cellareas[cm0]
        cellaream1 = cellareas[cm1]
        edgeorientm = edgeorients[em,:]
        hm0 = h[cm0]/cellaream0
        hm1 = h[cm1]/cellaream1
        densm0 = dens[cm0]/cellaream0
        densm1 = dens[cm1]/cellaream1

        ep = neighbor_edges[e,1]
        cp0 = CE[ep,0]
        cp1 = CE[ep,1]
        cellareap0 = cellareas[cp0]
        cellareap1 = cellareas[cp1]
        edgeorientp = edgeorients[ep,:]
        hp0 = h[cp0]/cellareap0
        hp1 = h[cp1]/cellareap1
        densp0 = dens[cp0]/cellareap0
        densp1 = dens[cp1]/cellareap1

        hjump_m = hm0*edgeorientm[0] + hm1*edgeorientm[1]
        hjump_p = hp0*edgeorientp[0] + hp1*edgeorientp[1]
        hjump = h0*edgeorient[0] + h1*edgeorient[1]

        densjump_m = densm0*edgeorientm[0] + densm1*edgeorientm[1]
        densjump_p = densp0*edgeorientp[0] + densp1*edgeorientp[1]
        densjump = dens0*edgeorient[0] + dens1*edgeorient[1]

#EVENTUALLY WE COULD HAVE VARIABLE-SPECIFIC EPSILON CHOICES HERE AS WELL
        #compute epsilon
        eps, alphaval = _compute_epsilon(epsilon_type, epsilon_param, alphamin, alphamax, dens0, dens1, M0, M1, B0, B1, u0, u1, h0, h1, \
        hjump_m, hjump_p, hjump, densjump_m, densjump_p, densjump, primal_edge_length, edge_normal, edgeorient, ndim, ndofs)
        epsilon[e] = eps
        alpha[e] = alphaval

        #compute viscous flux
        dens_jump = (dens0 * edgeorient[0] + dens1 * edgeorient[1]) / primal_edge_length
        M_jump = (M0 * edgeorient[0] + M1 * edgeorient[1]) / primal_edge_length

        densflux[e,:] = dens_jump * eps * edge_length
        Mflux[e,:] = M_jump * eps * edge_length


    for m in prange(bedges.shape[0]):
        be = bedges[m]
        densbnd = densflux[be,:]
        Mbnd = Mflux[be,:]
        edgeorient = edgeorients[be,0]
        edge_length = edge_lengths[be]

#THIS SHOULD BE FIXABLE TO DO SOMETHING REALISTIC
        #compute epsilon
        #eps = _compute_epsilon(epsilon_type, epsilon_param)
        #epsilon[be] = eps

#THIS IS AN UGLY HACK- SHOULD BE SOME SORT OF CORRECTLY SET BOUNDARY VALUE...
        densflux[be,:] = 0 #densbnd * edgeorient * eps * edge_length
        Mflux[be,:] = 0 #-Mbnd * edgeorient * eps * edge_length

#THIS IS HIGHLY COMPRESSIBLE EULER + IDEAL GAS SPECIFIC
#IT SHOULD REALLY BE PART OF HAMILTONIAN, AND THROUGH THAT CONNECTION WITH THERMODYNAMICS

#SPECIFIC TO IDEAL GAS
@njit(cache=True)
def _compute_cs(rho, eta, gamma, Cv):
    p = pow(rho, gamma) * exp(eta / Cv)
    return sqrt(gamma * p / rho)

#SPECIFIC TO COMPRESSIBLE EULER
#SPECIFIC TO 1D
@njit(cache=True)
def _compute_max_signal_speed(dens, M):
    rho = dens[0]
    S = dens[1]
    eta = S / rho
    u = M[0] /rho
    cs = _compute_cs(rho, eta, 1.4, 1.0)
    return cs + abs(u)

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
def _compute_Ftilde(flux0, flux1):
    return (flux0 + flux1)/2.0

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

@njit(cache=True)
def _compute_alpha_edge(dens0, dens1, M0, M1, B0, B1, u0, u1, h0, h1, edge_normal, edgeorient, ndim, ndofs):

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

    return alpha

@njit(cache=True)
def _compute_epsilon(epsilon_type, epsilon_const, alphamin, alphamax, dens0, dens1, M0, M1, B0, B1, u0, u1, h0, h1, \
hjump_m, hjump_p, hjump, densjump_m, densjump_p, densjump, edgelen, edge_normal, edgeorient, ndim, ndofs):
    if epsilon_type == 'const':
        return epsilon_const, 0.0
    elif epsilon_type == 'rusanov':
        smax = max(_compute_max_signal_speed(dens0, M0), _compute_max_signal_speed(dens1, M1))
        return epsilon_const * edgelen * smax, 0.0
    elif epsilon_type == 'alpha':
        smax = max(_compute_max_signal_speed(dens0, M0), _compute_max_signal_speed(dens1, M1))
        alpha = _compute_alpha_edge(dens0, dens1, M0, M1, B0, B1, u0, u1, h0, h1, edge_normal, edgeorient, ndim, ndofs)
        if (alphamax - alphamin) > 0.0:
            alphabar = min(1.0, max(0.0, (abs(alpha) - alphamin) / (alphamax - alphamin)))
        else:
            alphabar = 0.0
        return epsilon_const * alphabar * edgelen * smax, alpha

    elif epsilon_type == 'minbee':
        smax = max(_compute_max_signal_speed(dens0, M0), _compute_max_signal_speed(dens1, M1))
        if hjump > 0.0:
            d0 = hjump_m/hjump
            d1 = hjump_p/hjump
            phi0 = max(0,min(1,d0))
            phi1 = max(0,min(1,d1))
            phi = min(phi0, phi1)
        else:
            phi = 0.0
        return epsilon_const * edgelen * smax * (1.0 -phi), 0.0

    elif epsilon_type == 'minbee_rho':
        smax = max(_compute_max_signal_speed(dens0, M0), _compute_max_signal_speed(dens1, M1))
        if densjump[0] > 0.0:
            d0 = densjump_m[0]/densjump[0]
            d1 = densjump_p[0]/densjump[0]
            phi0 = max(0,min(1,d0))
            phi1 = max(0,min(1,d1))
            phi = min(phi0, phi1)
        else:
            phi = 0.0
        return epsilon_const * edgelen * smax * (1.0 -phi), 0.0

#EVENTUALLY WE COULD HAVE VARIABLE-SPECIFIC EPSILON CHOICES HERE AS WELL
@njit(parallel=True, cache=True)
def _compute_entropy_source(dens, M, B, u, denssource, epsilon, \
EC, nEC, CE, dn_to_p0, edge_orients, edge_lengths, primal_edge_lengths, cellareas, bnd, \
ncells, ndim, ndofs, entropy_loc):
    for i in prange(ncells):
        prod_term = 0.0
        for j in range(nEC[i]):
            e = EC[i,j]
            if not bnd[e]:
                edgeorient = edge_orients[e,:]
                c0 = CE[e,0]
                c1 = CE[e,1]
                v0 = dn_to_p0[c0]
                v1 = dn_to_p0[c1]
                dens_jump = (dens[c0,:] / cellareas[c0] * edgeorient[0] + dens[c1,:] / cellareas[c1] * edgeorient[1])
                M_jump = (M[c0,:] / cellareas[c0] * edgeorient[0] + M[c1,:] / cellareas[c1] * edgeorient[1])
                B_jump = (B[v0,:] * edgeorient[0] + B[v1,:] * edgeorient[1])
                u_jump = (u[v0,:] * edgeorient[0] + u[v1,:] * edgeorient[1])
                prod = 0.0
                for l in range(ndofs):
                    prod += dens_jump[l] * B_jump[l]
                for d1 in range(ndim):
                    prod += M_jump[d1] * u_jump[d1]
                #print(i,j,e,epsilon,prod,primal_edge_lengths[e],edge_lengths[e])
 #the 1/2 term here is so that there is a product rule that reconstructs a viable energy regularization term
                prod_term += 0.5 * epsilon[e] * prod / primal_edge_lengths[e] * edge_lengths[e]
            else:
                pass
#ADD VISCOUS BOUNDARY TERMS, IF THEY EXIST?
#YES MUST DO THIS
#I THINK THEY ARE JUST THE DENSFLUX TIMES B AT THE BND? etc.
        T = B[i,entropy_loc]
        denssource[i,entropy_loc] = prod_term / T


class NoRegularization():
    def __init__(self, ):
        self.diaglist = []

    def add_diagnostic_vars(self, dvars):
        pass

    def compute_diagnostic_vars(self, prog_vars, diag_vars):
        pass

    def compute_fluxes(self, prog_vars, aux_vars):
        pass

class ThermodynamicallyCompatibleRegularization():
    def __init__(self, hamiltonian, meshes, epsilon_type, epsilon_value, alphamin, entropydof, ndofs, construct=True):
        self.meshes = meshes
        self.epsilon_type = epsilon_type
        self.epsilon_value = epsilon_value
        self.alphamin = alphamin
#PROBABLY WANT TO DO SOMETHING A LITTLE MORE CLEVER HERE IE COMPUTE SOME INITIAL ALPHA...
        self.alphamax = 0.0
        self._alpha_max_int = 0.0
        self.entropydof = entropydof
        self.ndofs = ndofs
        self.hamiltonian = hamiltonian

        self.diaglist = ['Pi', 'eps', 'alpha']

        if construct:
    #THIS SHOULD ALL BE PART OF THE TOPOLOGY/GEOMETRY CLASSES!
            self.cellareas = np.zeros(self.meshes.dtopo.nkcells[self.meshes.dim])
            self.edge_normals = np.zeros((self.meshes.dtopo.nkcells[self.meshes.dim-1], self.meshes.dim))
            self.edge_orients = np.zeros((self.meshes.dtopo.nkcells[self.meshes.dim-1], 2), dtype=np.int32)
            self.edge_lengths = np.zeros(self.meshes.dtopo.nkcells[self.meshes.dim-1])
            self.primal_edge_lengths = np.zeros(self.meshes.dtopo.nkcells[self.meshes.dim-1])
            self.CE = np.zeros((self.meshes.dtopo.nkcells[self.meshes.dim-1], 2), dtype=np.int32)
#THIS IS HIGHLY 1D SPECIFIC
            self.neighbor_edges = np.zeros((self.meshes.dtopo.nkcells[self.meshes.dim-1], 2), dtype=np.int32)
            self.bedges = np.zeros(self.meshes.dtopo.nbkcells[self.meshes.dim-1], dtype=np.int32)
            self.dn_to_p0 = np.zeros(self.meshes.dtopo.nkcells[self.meshes.dim], dtype=np.int32)

            maxne = meshes.dtopo.petscmesh.getMaxSizes()[0]

            self.EC = np.zeros((self.meshes.dtopo.nkcells[self.meshes.dim], maxne), dtype=np.int32)
            self.nEC = np.zeros(self.meshes.dtopo.nkcells[self.meshes.dim], dtype=np.int32)
            self.bnd = np.full(self.meshes.dtopo.nkcells[self.meshes.dim-1], False, dtype=np.bool_)

            #THIS IS AN UGLY HACK- SHOULD REALLY BE BASED ON A BASIS CHOICE FOR AN EDGE...
            #ALSO ASSUMES THAT ALL QUADRATURE POINTS SHARE A BASIS- WHICH IS NOT NECCESARILY TRUE...
            self.edge_normals[:,0] = 1.0

            dcoff = meshes.dtopo.kcells_off[meshes.dim]
            deoff = meshes.dtopo.kcells_off[meshes.dim - 1]
            peoff = meshes.ptopo.kcells_off[1]

            for dc in range(meshes.dtopo.kcells[meshes.dim][0], meshes.dtopo.kcells[meshes.dim][1]):
                self.cellareas[dc - dcoff] = meshes.dgeom.entitysizes[meshes.dim][dc - dcoff]
                edges = meshes.dtopo.lower_dim_TC(dc, self.meshes.dim-1)
                nedges = edges.shape[0]
                self.EC[dc - dcoff, :nedges] = edges - deoff
                self.nEC[dc - dcoff] = nedges
                self.dn_to_p0[dc - dcoff] = meshes.pdmapping.dinmk_to_pk(dc) - meshes.ptopo.kcells_off[0]

            for de in range(meshes.dtopo.kcells[meshes.dim-1][0], meshes.dtopo.kcells[meshes.dim-1][1] - meshes.dtopo.nbkcells[meshes.dim-1]):
                pe = meshes.pdmapping.dinmk_to_pk(de)
                self.CE[de - deoff, :] = meshes.dtopo.higher_dim_TC(de, meshes.dim) - dcoff
                self.edge_orients[de - deoff, :] = meshes.dtopo.higher_orientation(de)
                #print(de,self.CE[de - deoff, :],self.edge_orients[de - deoff, :])
                self.primal_edge_lengths[de - deoff] = meshes.pgeom.entitysizes[1][pe - peoff]

#THIS NEEDS TO BE MODIFIED FOR MULTI-DIMENSIONAL CASE
                cell0 = self.CE[de - deoff, 0]
                cell1 = self.CE[de - deoff, 1]
                EC0 = meshes.dtopo.lower_dim_TC(cell0 + dcoff, meshes.dim-1) - deoff
                EC1 = meshes.dtopo.lower_dim_TC(cell1 + dcoff, meshes.dim-1) - deoff
                for e0 in EC0:
                    if not (e0 == de):
                        self.neighbor_edges[de- deoff,0] = e0 - deoff
                for e1 in EC1:
                    if not (e1 == de):
                        self.neighbor_edges[de- deoff,1] = e1 - deoff
                if meshes.dtopo.has_boundary and (meshes.dtopo.petscmesh.getLabelValue('bnd', self.neighbor_edges[de - deoff,0] + deoff) == 1):
                    self.neighbor_edges[de - deoff,0] = self.neighbor_edges[de - deoff,1]
                if meshes.dtopo.has_boundary and (meshes.dtopo.petscmesh.getLabelValue('bnd', self.neighbor_edges[de - deoff,1] + deoff) == 1):
                    self.neighbor_edges[de - deoff,1] = self.neighbor_edges[de - deoff,0]
            for de in range(meshes.dtopo.kcells[meshes.dim-1][0], meshes.dtopo.kcells[meshes.dim-1][1]):
                self.edge_lengths[de - deoff] = meshes.dgeom.entitysizes[meshes.dim-1][de - deoff]

            if meshes.dtopo.has_boundary:
                self.bedges[:] = meshes.dtopo.bkcells[meshes.dim-1]
                for de in range(meshes.dtopo.kcells[meshes.dim-1][0], meshes.dtopo.kcells[meshes.dim-1][1]):
                    self.bnd[de - deoff] = (meshes.dtopo.petscmesh.getLabelValue('bnd', de) == 1)
                    self.CE[de - deoff, :] = meshes.dtopo.higher_dim_TC(de, meshes.dim) - dcoff
                    self.edge_orients[de - deoff, :] = meshes.dtopo.higher_orientation(de)
                    #print(de,self.CE[de - deoff, :],self.edge_orients[de - deoff, :])

            self.internalvars = {}
            self.internalvars['B'] = KForm(0, self.meshes.ptopo, meshes.pRBundle, 'B', create_petsc=True, ndofs = ndofs)
            self.internalvars['u'] =  KForm(0, self.meshes.ptopo, meshes.pTBundle, 'u', create_petsc=True)
            self.internalvars['h'] = KForm(meshes.dim, self.meshes.dtopo, meshes.dRBundle, 'h', create_petsc=True)

    def add_diagnostic_vars(self, dvars):
        dvars['eps'] = KForm(self.meshes.dim-1, self.meshes.dtopo, self.meshes.pRBundle, 'eps', create_petsc=True)
        dvars['Pi'] = KForm(0, self.meshes.ptopo, self.meshes.pRBundle, 'Pi', create_petsc=True)
        dvars['alpha'] = KForm(self.meshes.dim-1, self.meshes.dtopo, self.meshes.pRBundle, 'alpha', create_petsc=True)


#BROKEN, TO BE FIXED
    def compute_diagnostic_vars(self, prog_vars, diag_vars):

        epsilonarr = diag_vars['eps'].petsc_vec.getArray()
        alphaarr = diag_vars['alpha'].petsc_vec.getArray()
        densarr = prog_vars['dens'].petsc_vec.getArray()
        densarr = densarr.reshape((prog_vars['dens'].nelems, prog_vars['dens'].ndofs))
        Marr = prog_vars['M'].petsc_vec.getArray()
        Marr = Marr.reshape((prog_vars['M'].nelems, prog_vars['M'].bsize))
        Piarr = diag_vars['Pi'].petsc_vec.getArray()
        Piarr = Piarr.reshape((diag_vars['Pi'].nelems, 1))
        Barr = self.internalvars['B'].petsc_vec.getArray()
        Barr = Barr.reshape((self.internalvars['B'].nelems, self.internalvars['B'].ndofs))
        uarr = self.internalvars['u'].petsc_vec.getArray()
        uarr = uarr.reshape((self.internalvars['u'].nelems, self.internalvars['u'].bsize))
        harr = self.internalvars['h'].petsc_vec.getArray()


#compute B,u,h
        self.hamiltonian.compute_dHdx(prog_vars, self.internalvars)
        self.hamiltonian.compute_pointwise_energy(prog_vars, self.internalvars, self.internalvars['h'])

#compute epsilon

        de_off = self.meshes.dtopo.kcells_off[self.meshes.dim-1]
        for de in range(self.meshes.dtopo.kcells[self.meshes.dim-1][0] - de_off, self.meshes.dtopo.kcells[self.meshes.dim-1][1] - self.meshes.dtopo.nbkcells[self.meshes.dim-1] - de_off):
        #compute epsilon

            c0 = self.CE[de,0]
            c1 = self.CE[de,1]
            v0 = self.dn_to_p0[c0]
            v1 = self.dn_to_p0[c1]
            primal_edge_length = self.primal_edge_lengths[de]
            edgeorient = self.edge_orients[de,:]
            edge_normal = self.edge_normals[de,:]

            cellarea0 = self.cellareas[c0]
            cellarea1 = self.cellareas[c1]

            dens0 = densarr[c0,:]/cellarea0
            dens1 = densarr[c1,:]/cellarea1
            M0 = Marr[c0,:]/cellarea0
            M1 = Marr[c1,:]/cellarea1
            B0 = Barr[v0,:]
            B1 = Barr[v1,:]
            u0 = uarr[v0,:]
            u1 = uarr[v1,:]
            h0 = harr[c0]/cellarea0
            h1 = harr[c1]/cellarea1

            em = self.neighbor_edges[de,0]
            cm0 = self.CE[em,0]
            cm1 = self.CE[em,1]
            cellaream0 = self.cellareas[cm0]
            cellaream1 = self.cellareas[cm1]
            edgeorientm = self.edge_orients[em,:]
            hm0 = harr[cm0]/cellaream0
            hm1 = harr[cm1]/cellaream1
            densm0 = densarr[cm0]/cellaream0
            densm1 = densarr[cm1]/cellaream1

            ep = self.neighbor_edges[de,1]
            cp0 = self.CE[ep,0]
            cp1 = self.CE[ep,1]
            cellareap0 = self.cellareas[cp0]
            cellareap1 = self.cellareas[cp1]
            edgeorientp = self.edge_orients[ep,:]
            hp0 = harr[cp0]/cellareap0
            hp1 = harr[cp1]/cellareap1
            densp0 = densarr[cp0]/cellareap0
            densp1 = densarr[cp1]/cellareap1

            hjump_m = hm0*edgeorientm[0] + hm1*edgeorientm[1]
            hjump_p = hp0*edgeorientp[0] + hp1*edgeorientp[1]
            hjump = h0*edgeorient[0] + h1*edgeorient[1]

            densjump_m = densm0*edgeorientm[0] + densm1*edgeorientm[1]
            densjump_p = densp0*edgeorientp[0] + densp1*edgeorientp[1]
            densjump = dens0*edgeorient[0] + dens1*edgeorient[1]

            eps, alpha = _compute_epsilon(self.epsilon_type, self.epsilon_value, self.alphamin, self.alphamax, dens0, dens1, M0, M1, B0, B1, u0, u1, h0, h1, \
            hjump_m, hjump_p, hjump, densjump_m, densjump_p, densjump, primal_edge_length, edge_normal, edgeorient, self.meshes.dim, self.ndofs)
            epsilonarr[de] = eps
            alphaarr[de] = alpha

#now compute Pi using epsilon, B, u

        _compute_entropy_source(densarr, Marr, Barr, uarr, Piarr, epsilonarr, \
        self.EC, self.nEC, self.CE, self.dn_to_p0, self.edge_orients, self.edge_lengths, self.primal_edge_lengths, self.cellareas, self.bnd, self.meshes.dtopo.nkcells[self.meshes.dim], self.meshes.dim, self.ndofs, 0)


    def compute_fluxes(self, state, aux_vars):

        #extract needed arrays
        Barr = aux_vars['B'].petsc_vec.getArray()
        Barr = Barr.reshape((aux_vars['B'].nelems, aux_vars['B'].ndofs))
        densarr = state['dens'].petsc_vec.getArray()
        densarr = densarr.reshape((state['dens'].nelems, state['dens'].ndofs))
        uarr = aux_vars['u'].petsc_vec.getArray()
        uarr = uarr.reshape((aux_vars['u'].nelems, aux_vars['u'].bsize))
        Marr = state['M'].petsc_vec.getArray()
        Marr = Marr.reshape((state['M'].nelems, state['M'].bsize))
        harr = aux_vars['h'].petsc_vec.getArray()

        densfluxarr = aux_vars['dens_flux'].petsc_vec.getArray()
        densfluxarr = densfluxarr.reshape((aux_vars['dens_flux'].nelems, aux_vars['dens_flux'].ndofs))
        Mfluxarr = aux_vars['M_flux'].petsc_vec.getArray()
        Mfluxarr = Mfluxarr.reshape((aux_vars['M_flux'].nelems, aux_vars['M_flux'].bsize))

        denssourcearr = aux_vars['dens_source'].petsc_vec.getArray()
        denssourcearr = denssourcearr.reshape((aux_vars['dens_source'].nelems, aux_vars['dens_source'].ndofs))

        epsilonarr = aux_vars['epsilon'].petsc_vec.getArray()
        alphaarr = aux_vars['alpha'].petsc_vec.getArray()

        self.compute_edge_fluxes(densarr, Marr, Barr, uarr, harr, densfluxarr, Mfluxarr, epsilonarr, alphaarr)
        self.compute_entropy_production(densarr, Marr, Barr, uarr, denssourcearr, epsilonarr)

        #self._alpha_max_int = max(self._alpha_max_int, np.max(np.abs(alphaarr)))
        self.alphamax = np.max(np.abs(alphaarr))
        #print(self.alphamax)

    def pre_step(self, aux_vars):
        pass
        #self._alpha_max_int = 0.0

    def post_step(self, aux_vars):
        pass
        #alphaarr = aux_vars['alpha'].petsc_vec.getArray()
        #self.alphamax = np.max(alphaarr)
        #self.alphamax = self._alpha_max_int
        #print(self.alphamax)

    def compute_edge_fluxes(self, dens, M, B, u, h, densflux, Mflux, epsilon, alpha):

        _compute_edge_fluxes(dens, M, B, u, h, densflux, Mflux, epsilon, alpha, \
        self.CE, self.neighbor_edges, self.dn_to_p0, self.edge_orients, self.edge_normals, self.edge_lengths, self.primal_edge_lengths, self.bedges, self.cellareas, self.meshes.dtopo.nkcells[self.meshes.dim-1] - self.meshes.dtopo.nbkcells[self.meshes.dim-1], self.meshes.dim, self.ndofs, self.epsilon_type, self.epsilon_value, self.alphamin, self.alphamax)


    def compute_entropy_production(self, dens, M, B, u, denssource, epsilon):
        _compute_entropy_source(dens, M, B, u, denssource, epsilon, \
        self.EC, self.nEC, self.CE, self.dn_to_p0, self.edge_orients, self.edge_lengths, self.primal_edge_lengths, self.cellareas, self.bnd, self.meshes.dtopo.nkcells[self.meshes.dim], self.meshes.dim, self.ndofs, self.entropydof)


def getRegularization(params, hamiltonian, meshes, entropydof, ndofs, construct=True):
    if params['viscous_type'] == 'none':
        return NoRegularization()
    if params['viscous_type'] == 'thermodynamicallycompatible':
        return ThermodynamicallyCompatibleRegularization(hamiltonian, meshes, params['eps_type'], params['eps_coeff'], params['alphamin'], entropydof, ndofs, construct=construct)
