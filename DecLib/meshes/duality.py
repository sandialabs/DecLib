from DecLib import PETSc
from math import sqrt, isfinite, factorial, cos, sin, acos
import numpy as np
from DecLib.meshes.geometry import CartesianPlanarMetric

def createTwistedTopology(smesh):
    mapping = StraightTwistedMapping(smesh)
    tdim = smesh.tdim
    twisted_nIkcells, twisted_nbkcells, twisted_nBkcells, twisted_kcells_off = _compute_dual_mesh_sizes(smesh)

    twisted_nIbkcells = twisted_nIkcells + twisted_nbkcells
    twisted_nkcells = twisted_nIbkcells + twisted_nBkcells

    petscmesh = PETSc.DMPlex().create()
    petscmesh.setChart(0, np.sum(twisted_nkcells))
    petscmesh.setDimension(dim)

    #set Ib cone/support sizes
    for k in range(tdim+1):
        if k > 0:
            start, end = smesh.get_loop_indices(tdim-k, 'Ib')
            for sk in range(start, end): #loop over straight n-k cells
                ssupportsize = smesh.petscmesh.getSupportSize(sk)
                tk = mapping.sIb_to_tIb(sk)
                petscmesh.setConeSize(tk, ssupportsize)
        if k < tdim:
            start, end = smesh.get_loop_indices(tdim-k, 'Ib')
            for sk in range(start, end): #loop over straight n-k cells
                pconesize = pmesh.petscmesh.getConeSize(sk)
                tk = mapping.sIb_to_tIb(sk)
                petscmesh.setSupportSize(sk, sconesize)

# #THIS IS pb -> dB piece I think
# DOES THIS WORK IN 3D?
    # augment cone sizes for elements of dI that come from pb by 1
    # these are 1, ..., n cells
    for k in range(1,dim+1):
        start, end = smesh.get_loop_indices(tdim-k, 'b')
        for sk in range(start, end):
            tk = mapping.sIb_to_tIb(sk)
            oldsize = petscmesh.getConeSize(tk)
            petscmesh.setConeSize(tk, oldsize+1)


    # add new entities for tB
    # these are 0, ..., n-1 cells
    # cones are sized using sb
    # supports are sized using sb + 1 (ie the element of dI that comes from sb -> tIb)
# DOES THIS WORK IN 3D?
    for k in range(dim):
        start, end = smesh.get_loop_indices(tdim-k-1, 'b')
        for sb in range(start, end):
            tB = mapping.sb_to_tB(sb)
            if k >0:
                ssupportsize = smesh.petscmesh.getSupportSize(sb)
                ssupport = smesh.petscmesh.getSupport(sb)
                ssupport_adj = mapping.sb_to_tB(ssupport.copy())
                ssupport_adj = list(ssupport_adj)
                ssupportsize = ssupportsize - ssupport_adj.count(-1)
                petscmesh.setConeSize(tB, ssupportsize)
            sconesize = pmesh.petscmesh.getConeSize(sb)
            petscmesh.setSupportSize(tB, sconesize+1)

    petscmesh.setUp()

    #set interior cones/supports
    for k in range(dim+1):
        if k > 0:
            start, end = smesh.get_loop_indices(tdim-k, 'Ib')
            for sk in range(start, end): #loop over primal n-k cells
                tc = smesh.higher_dim_TC(sk, dim-k+1)
                tk = mapping.sIb_to_tIb(sk)
                tc_adj = mapping.sIb_to_tIb(tc)
                petscmesh.setCone(tk, tc_adj)

        if k < dim:
            start, end = smesh.get_loop_indices(tdim-k, 'Ib')
            for sk in range(start, end): #loop over primal n-k cells
                tc = smesh.lower_dim_TC(sk, dim-k-1)
                tc_adj = mapping.sIb_to_tIb(tc)
                tk = mapping.sIb_to_tIb(sk)
                petscmesh.setSupport(tk, tc_adj)

    #set boundary cones/supports

    # augment cones for elements of dI that come from sB by 1
    # these are 1, ..., n cells
    for k in range(1,dim+1):
        start, end = smesh.get_loop_indices(tdim-k, 'Ib')
        for sk in range(start, end):
            tk = mapping.sIb_to_tIb(sk)
            oldcone = petscmesh.getCone(tk)
            tB = mapping.sb_to_tB(sk)
            newcone = list(oldcone)
            newcone.append(tk)
            newcone = np.array(newcone, dtype=np.int32)
            petscmesh.setCone(tk, newcone)


    # add new entities for dB
    # these are 0, ..., n-1 cells
    # cones come from pB
    # supports come from pB plus "1" (ie the element of dI that comes from pB -> dI)
    for k in range(dim):
        start, end = smesh.get_loop_indices(tdim-k, 'Ib')
        for sb in range(start, end):
            tB = mapping.sb_to_tB(sb)
            if k>0:
                ssupport = smesh.petscmesh.getSupport(sb)
                ssupport_adj = mapping.sb_to_tB(ssupport.copy())
                cone = []
                for ss in ssupport_adj:
                    if ss == -1:
                        continue
                    cone.append(ss)
                petscmesh.setCone(tB, np.array(cone, dtype=np.int32))
                # MAYBE NEED TO ORIENT STUFF IN 3D, NOT CLEAR YET!

            scone = smesh.petscmesh.getCone(sb)
            scone_adj = mapping.sb_to_tB(scone.copy())
            tk = mapping.sIb_to_tIb(sb)
            support = list(scone_adj)
            support.append(tk)
            support = np.array(support, dtype=np.int32)
            petscmesh.setSupport(tB, support)

    petscmesh.stratify()

# #ORIENTATIONS!
#create empty versions?
#need to be careful with sizing- probably needs stencils to be created?

#for Ib cells can just (mostly) inherit these from straight mesh
#for B cells (and their "connected" Ib cells) can also inherit, but a little more complicated

# #celltype LABELS!
#
# #PROBABLY NEED TO REINDEX THINGS SO THAT I,b,B ordering is maintained?
# YES HAVE TO DO THIS...
# #ie Ib <-> Ib, b <-> B and B <-> b
# #but not true that I <-> I, etc.!
#     return NewTopology(petscmesh, orientations=orientations)
# THIS WOULD ALSO REQUIRE REORDERING THE PDMAPPING and ORIENTATIONS I THINK?
# YES DEFINITELY REQUIRES THAT

#Then we can create the Topology
#with given orientations (KEY- REQUIRES SLIGHT CODE CHANGES)
#it should create the stencils automatially though


def createTwistedGeometry(stopo, sgeom, pdmapping, duality_type='barycentric'):
    pass

#create geometry arrays

#fill vertexs, edgesegments, facetriangles, cell tetrahedra (maybe also centroids)
#need to be somewhat careful with periodic stuff here- will get edges/faces/cells that straddle the "boundaries"
#here we choose between barycentric and circumcentric duality

#"center" of straight cells -> vertices for dual
#if not periodic this is enough to autogenerate the rest of the geometry
#MUST BE CAREFUL WITH NORMALS AND EDGE TANGENTS, HOWEVER, AND CORRESPONDENCE WITH ORIENTATION
#ie for this we assume that we HAVE orientations, and use them to set edge tangents and facenormals
#rather than the other way around...

#create the rest of geometric quantities as needed


def _compute_dual_mesh_sizes(pmesh):
    tdim = pmesh.tdim
    nIkcells = np.zeros(tdim+1, dtype=np.int32)
    nbkcells = np.zeros(tdim+1, dtype=np.int32)
    nBkcells = np.zeros(tdim+1, dtype=np.int32)
    kcells_off = np.zeros(tdim+1, dtype=np.int32)

    for k in range(dim+1):
        nIkcells[tdim-k] = pmesh.nIkcells[k]
        nbkcells[tdim-k] = pmesh.nBkcells[k]
        nBkcells[tdim-k] = pmesh.nbkcells[k]

    off = 0
    for k in range(tdim+1):
        kcells_off[k] = off
        off = off + nIkcells[k] + nbkcells[k] + nBkcells[k]

    return nIkcells, nbkcells, nBkcells, kcells_off


#Ib <-> Ib
#b <-> B
#B <-> b (DONT HAVE TO CREATE THIS I THINK!)
#ie the existing mesh creation process should be okay
#need to test it with mixed b/B meshes
#Probably do need to reorder/permute some of the cells to get I,b,B
#
class StraightTwistedMapping():
    def __init__(self, smesh):

        tdim = smesh.tdim
        twisted_nIkcells, twisted_nbkcells, twisted_nBkcells, twisted_kcells_off = _compute_dual_mesh_sizes(smesh)

        nIbkcells = smesh.nIkcells + smesh.nbkcells
        straight_cells = np.arange(np.sum(nIbkcells), dtype=np.int32)
        twisted_cells = np.zeros(np.sum(nIbkcells), dtype=np.int32)
        for k in range(tdim+1):
            start = smesh.kcells_off[tdim-k]
            end = smesh.kcells_off[tdim-k] + nIbkcells[tdim-k]
            twisted_cells[start:end] = np.arange(twisted_kcells_off[k], twisted_kcells_off[k] + twisted_nIkcells[k] + twisted_nbkcells[k], dtype=np.int32)
        self._sIb_to_tIb = PETSc.AO().createMapping(straight_cells, twisted_cells)

        straight_cells = np.zeros(np.sum(smesh.nbkcells), dtype=np.int32)
        twisted_cells = np.zeros(np.sum(smesh.nbkcells), dtype=np.int32)
        for k in range(tdim+1):
            start = smesh.kcells_off[tdim-k] + smesh.nIkcells[tdim-k]
            end = smesh.kcells_off[tdim-k] + smesh.nIkcells[tdim-k] + smesh.nbkcells[dim-k]
            straight_cells[start:end] = np.arange(start, end, dtype=np.int32)
            tstart = twisted_kcells_off[k] + twisted_nIkcells[k] + twisted_nbkcells[k]
            tend = twisted_kcells_off[k] + twisted_nIkcells[k] + twisted_nbkcells[k] + twisted_nBkcells[k]
            twisted_cells[start:end] = np.arange(tstart, tend, dtype=np.int32)
        self._sb_to_tB = PETSc.AO().createMapping(straight_cells, twisted_cells)

        straight_cells = np.zeros(np.sum(smesh.nBkcells), dtype=np.int32)
        twisted_cells = np.zeros(np.sum(smesh.nBkcells), dtype=np.int32)
        for k in range(tdim+1):
            start = smesh.kcells_off[tdim-k] + smesh.nIkcells[tdim-k] + smesh.nbkcells[tdim-k]
            end = smesh.kcells_off[tdim-k] + smesh.nIkcells[tdim-k] + smesh.nbkcells[tdim-k] + smesh.nBkcells[tdim-k]
            straight_cells[start:end] = np.arange(start, end, dtype=np.int32)
            tstart = twisted_kcells_off[k] + twisted_nIkcells[k]
            tend = twisted_kcells_off[k] + twisted_nIkcells[k] + twisted_nbkcells[k]
            twisted_cells[start:end] = np.arange(tstart, tend, dtype=np.int32)
        self._sB_to_tb = PETSc.AO().createMapping(straight_cells, twisted_cells)

    def sIb_to_tIb(self, sIb):
        return self._sIb_to_tIb.app2petsc(sIb)

    def tIb_to_sIb(self, tIb):
        return self._sIb_to_tIb.petsc2app(tIb)

    def sb_to_tB(self, sb):
        return self._sb_to_tB.app2petsc(sb)

    def sb_to_tB(self, tB):
        return self._sb_to_tB.petsc2app(tB)

    def sB_to_tb(self, sB):
        return self._sB_to_tb.app2petsc(sB)

    def sB_to_tb(self, tb):
        return self._sB_to_tb.petsc2app(tb)

class BarycentricDual(CartesianPlanarMetric):
    pass

class CircumentricDual(CartesianPlanarMetric):
    pass

# def DualTopology(pmesh):
#

#
# #NEED TO ADD A pB to db piece also? This is actually a little unclear, possibly it already exists? Actually it should, since we have pIb -> dIb already?
#
#
#
#         #set interior cones/supports
#         for k in range(dim+1):
#             if k > 0:
#                 for pk in range(pmesh.kcells[dim-k][0], pmesh.kcells[dim-k][1]): #loop over primal n-k cells
#                     tc = pmesh.higher_dim_TC(pk, dim-k+1)
#                     dk = mapping.pk_to_dinmk(pk)
#                     tc_adj = mapping.pk_to_dinmk(tc)
#                     petscmesh.setCone(dk, tc_adj)
#
#             if k < dim:
#                 for pk in range(pmesh.kcells[dim-k][0], pmesh.kcells[dim-k][1]): #loop over primal n-k cells
#                     tc = pmesh.lower_dim_TC(pk, dim-k-1)
#                     tc_adj = mapping.pk_to_dinmk(tc)
#                     dk = mapping.pk_to_dinmk(pk)
#                     petscmesh.setSupport(dk, tc_adj)
#
#         #set boundary cones/supports
#         if pmesh.has_boundary:
#
#             # augment cones for elements of dI that come from pB by 1
#             # these are 1, ..., n cells
#             for k in range(1,dim+1):
#                 for pb in pmesh.bkcells[dim-k]:
#                     dk = mapping.pk_to_dinmk(pb)
#                     oldcone = petscmesh.getCone(dk)
#                     db = mapping.pbk_to_dbnmk(pb)
#                     newcone = list(oldcone)
#                     newcone.append(db)
#                     newcone = np.array(newcone, dtype=np.int32)
#                     petscmesh.setCone(dk, newcone)
#
#
#             # add new entities for dB
#             # these are 0, ..., n-1 cells
#             # cones come from pB
#             # supports come from pB plus "1" (ie the element of dI that comes from pB -> dI)
#             for k in range(dim):
#                 for pb in pmesh.bkcells[dim-k-1]:
#                     db = mapping.pbk_to_dbnmk(pb)
#                     if k>0:
#                         psupport = pmesh.petscmesh.getSupport(pb)
#                         psupport_adj = mapping.pbk_to_dbnmk(psupport.copy())
#                         cone = []
#                         for pp in psupport_adj:
#                             if pp == -1:
#                                 continue
#                             cone.append(pp)
#                         petscmesh.setCone(db, np.array(cone, dtype=np.int32))
#                         # MAYBE NEED TO ORIENT STUFF IN 3D, NOT CLEAR YET!
#
#                     pcone = pmesh.petscmesh.getCone(pb)
#                     pcone_adj = mapping.pbk_to_dbnmk(pcone.copy())
#                     dk = mapping.pk_to_dinmk(pb)
#                     support = list(pcone_adj)
#                     support.append(dk)
#                     support = np.array(support, dtype=np.int32)
#                     petscmesh.setSupport(db, support)
#
#         petscmesh.stratify()
#


#def transform_geometry():
#pass
#SOMETHING LIKE THIS IS QUITE USEFUL!
#SHOULD BOUNDARIES STAY FIXED DURING THIS PROCESS? PROBABLY YES, ALTHOUGH THIS IS LESS CLEAR...
#def transform_func(x):
#    return exp(x)
#sgeom.transform(transform_func)












#OverlapGeometry Class

#Create from a pair of topologies/geometries
#Bag class with a bunch of PetscSections + associated PetscVec
#overlap areas between various geometric entities








#SimplicialDualGeometry Function

#Given a simplicial topology and geometry, can generate the corresponding dual grid geometry based on circumcenters or barycenters
#Careful with b vs B here
