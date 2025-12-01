from DecLib import PETSc
import numpy as np






#All grid entities have labels- I,b,B

#ADD GHOST/BOUNDARY CELLS AS NEEDED HERE
#https://petsc.org/release/manualpages/DMPlex/DMPlexConstructGhostCells/

#HALO IS PROBABLY SET THROUGH
#https://petsc.org/release/manualpages/DMPlex/DMPlexSetOverlap/

#Parallel is through
#https://petsc.org/release/manualpages/DMPlex/DMPlexDistribute/
#https://petsc.org/release/manualpages/DM/DMSetAdjacency/
#need to be quite careful with how geometry generation interfaces with this, however!

#this orders the points in a DMPlex as I,b,B
def permute_topology(petscmesh):

    celltypes = petscmesh.getLabelIdIS('bnd').indices

    #see if there is a boundary
    if (1 in celltypes) or (2 in celltypes):

    #loop over all the points
        pStart, pEnd = petscmesh.getChart()
        indices = np.zeros(pEnd - pStart, dtype=np.int32)
        dim = petscmesh.getDimension()
        Ioff = np.zeros(dim+1, dtype=np.int32)
        boff = np.zeros(dim, dtype=np.int32)

        Icells = petscmesh.getStratumIS('bnd', 0)
        for I in Icells.indices:
            depth = petscmesh.getLabelValue('depth', I)
            Ioff[depth] = Ioff[depth] + 1
        if 1 in celltypes:
            bcells = petscmesh.getStratumIS('bnd', 1)
            for b in bcells.indices:
                depth = petscmesh.getLabelValue('depth', b)
            boff[depth] = boff[depth] + 1


        for d in range(dim+1):
            Iindex = 0
            bindex = 0
            Bindex = 0
            pdStart, pdEnd = petscmesh.getDepthStratum(d)
            for pd in range(pdStart, pdEnd):
                celltype = petscmesh.getLabelValue('bnd', pd)
                if celltype == 0:
                    indices[pd] = Iindex + pdStart
                    Iindex = Iindex + 1
                if celltype == 1:
                    indices[pd] = bindex + pdStart + Ioff[d]
                    bindex = bindex + 1
                if celltype == 2:
                    indices[pd] = Bindex + pdStart + Ioff[d] + boff[d]
                    Bindex = Bindex + 1

        newIndices = PETSc.IS().createGeneral(indices)

        return petscmesh.permute(newIndices), indices

    return petscmesh, None

def _orient_face(VEs):
    orients = np.zeros(len(VEs))
    VEind = list(range(len(VEs)))
    current_VE = VEs.pop()
    k = VEind.pop()
    first_vertex = current_VE[0]
    current_vertex = current_VE[1]
    orients[-1] = 1.
    edgeordering = []
    edgeordering.append(k)
    while not (current_vertex == first_vertex):
        for ind,VE in enumerate(VEs):
            k = VEind[ind]
            if VE[0] == current_vertex:
                current_VE = VEs.pop(ind)
                _ = VEind.pop(ind)
                orients[k] = 1.
                current_vertex = current_VE[1]
                edgeordering.append(k)
                break
            if VE[1] == current_vertex:
                current_VE = VEs.pop(ind)
                _ = VEind.pop(ind)
                orients[k] = -1.
                current_vertex = current_VE[0]
                edgeordering.append(k)
                break
    return orients, edgeordering


def _orient_1d(topo, geom):

    voff = topo.kcells_off[0]
    eoff = topo.kcells_off[1]
    for e in range(topo.kcells[1][0], topo.kcells[1][1]):
        VE = topo.VE[e-eoff, :topo.nVE[e-eoff]]
        e0 = geom.centroids[1][e - eoff, 0, :]
        for vind, v in enumerate(VE):
            fnorm = geom.face_normals[v-voff, :]
            EV = list(topo.EV[v-voff, :topo.nEV[v-voff]])
            eind = EV.index(e)
            v0 = geom.edge_segments[e- eoff, 0, vind, :]
            candidate_fnorm = e0 - v0
            orient = np.sign(np.dot(fnorm,candidate_fnorm))
            topo.incidence_numbers[0][1][v-voff, eind] = orient
            topo.incidence_numbers[1][0][e-eoff, vind] = orient




#We have the following self-consistent orientation choices:
#edge tangents point from VE[0] to VE[1]
#face normals are right-hand rule ie t x n = k, where k is upwards
#cells are oriented counter-clockwise
#inwards is positive, outwards is negative

def _orient_2d(topo, geom):
    voff = topo.kcells_off[0]
    eoff = topo.kcells_off[1]
    foff = topo.kcells_off[2]

    #These are consistent with _generate_edgetangents_facenormals
    #This is key!
    topo.incidence_numbers[1][0][:,0] = -1
    topo.incidence_numbers[1][0][:,1] = 1

    for v in range(topo.kcells[0][0], topo.kcells[0][1]):
        EV = topo.EV[v-voff, :topo.nEV[v-voff]]
        for eind,e in enumerate(EV):
            VE = list(topo.VE[e-eoff, :topo.nVE[e-eoff]])
            vind = VE.index(v)
            topo.incidence_numbers[0][1][v-voff, eind] = topo.incidence_numbers[1][0][e-eoff, vind]

    for f in range(topo.kcells[2][0], topo.kcells[2][1]):
        EF = topo.EF[f-foff, :topo.nEF[f-foff]]
        f0 = geom.centroids[2][f - foff, :]
        for eind, e in enumerate(EF):
            FE = list(topo.FE[e-eoff, :topo.nFE[e-eoff]])
            find = FE.index(f)
            e0 = np.average(geom.face_triangles[f-foff, eind, 0, :, :], axis=0)
            fnorm = geom.face_normals[e-eoff, 0, :]
            candidate_fnorm = f0 - e0
            orient = np.sign(np.dot(fnorm,candidate_fnorm))
            topo.incidence_numbers[2][0][f-foff, eind] = orient
            topo.incidence_numbers[1][1][e-eoff, find] = orient

def _orient_3d(topo, geom):
    voff = topo.kcells_off[0]
    eoff = topo.kcells_off[1]
    foff = topo.kcells_off[2]
    coff = topo.kcells_off[3]

    #These are consistent with _generate_edgetangents_facenormals
    #This is key!
    topo.incidence_numbers[1][0][:,0] = -1
    topo.incidence_numbers[1][0][:,1] = 1

    for v in range(topo.kcells[0][0], topo.kcells[0][1]):
        EV = topo.EV[v-voff, :topo.nEV[v-voff]]
        for eind,e in enumerate(EV):
            VE = list(topo.VE[e-eoff, :topo.nVE[e-eoff]])
            vind = VE.index(v)
            topo.incidence_numbers[0][1][v-voff, eind] = topo.incidence_numbers[1][0][e-eoff, vind]

    for f in range(topo.kcells[2][0], topo.kcells[2][1]):
        EF = topo.EF[f-foff, :topo.nEF[f-foff]]
        VElist = []
        for eind, e in enumerate(EF):
            VElist.append(topo.VE[e-eoff, :topo.nVE[e-eoff]])
        #This is consistent with facenormals code
        orients, _ = _orient_face(VElist)
        for eind, e in enumerate(EF):
            FE = list(topo.FE[e-eoff, :topo.nFE[e-eoff]])
            find = FE.index(f)
            topo.incidence_numbers[2][0][f-foff, eind] = orients[eind]
            topo.incidence_numbers[1][1][e-eoff, find] = orients[eind]

    for c in range(topo.kcells[3][0], topo.kcells[3][1]):
        FC = topo.FC[c-coff, :topo.nFC[c-coff]]
        c0 = geom.centroids[3][c - coff, :]
        for find, f in enumerate(FC):
            f0 = geom.cell_tetrahedra[c - coff, find, 0, 0, 2, :]
            CF = list(topo.CF[f-foff, :topo.nCF[f-foff]])
            cind = CF.index(c)
            fnorm = geom.face_normals[f-foff, 0, 0, :]
            candidate_fnorm = c0 - f0
            orient = np.sign(np.dot(fnorm,candidate_fnorm))
            topo.incidence_numbers[3][0][c-coff, find] = orient
            topo.incidence_numbers[2][1][f-foff, cind] = orient


class Topology():
    def __init__(self, petscmesh, name='', halowidth=1, is_simplicial=False, incidence_numbers=None):

        self.name = name
        self.petscmesh = petscmesh
        #self._check_mesh(petscmesh)
        self.tdim = self.petscmesh.getDimension()
        self.is_simplicial = is_simplicial

        self.kcells = []
        self.nkcells = []
        self.kcells_off = []
        for d in range(self.tdim+1):
            self.kcells.append(self.petscmesh.getDepthStratum(d))
            self.nkcells.append(self.kcells[d][1] - self.kcells[d][0])
            self.kcells_off.append(self.petscmesh.getDepthStratum(d)[0])
        self.kcells = np.array(self.kcells, dtype=np.int32)
        self.nkcells = np.array(self.nkcells, dtype=np.int32)
        self.kcells_off = np.array(self.kcells_off, dtype=np.int32)

        #cells are stored as I,b,B to make mapping trivial!

        self.nIkcells = []
        self.nbkcells = []
        self.nBkcells = []
        self.nIbkcells = []
        for k in range(self.tdim+1):
            kcells = self.petscmesh.getStratumIS('depth',k).getIndices()
            Inum = 0
            bnum = 0
            Bnum = 0
            for p in kcells:
                bnd = self.petscmesh.getLabelValue('bnd',p)
                if bnd == 0: Inum = Inum + 1
                if bnd == 1: bnum = bnum + 1
                if bnd == 2: Bnum = Bnum + 1
            self.nIkcells.append(Inum)
            self.nbkcells.append(bnum)
            self.nBkcells.append(Bnum)
            self.nIbkcells.append(Inum + bnum)

        self._create_stencils()

        if incidence_numbers is None:
            self._create_incidence_numbers()
        else:
            self.incidence_numbers = incidence_numbers

        self._distribute_mesh()


    def _build_connectivity_array(self, startcell, targetcell, size, trim=False):
        carr = np.zeros((self.nkcells[startcell],size), dtype=np.int32) - 1
        ncarr = np.zeros(self.nkcells[startcell], dtype=np.int32)
        if startcell < targetcell:
            TC = self.higher_dim_TC
        if startcell > targetcell:
            TC = self.lower_dim_TC
        for p in range(self.kcells[startcell][0], self.kcells[startcell][1]):
            pcarr = TC(p, targetcell)
            npcarr = pcarr.shape[0]
            carr[p - self.kcells_off[startcell], :npcarr] = pcarr
            ncarr[p - self.kcells_off[startcell]] = npcarr
        if trim:
            max = np.max(ncarr)
            carr = carr[:,:max]
        return carr, ncarr

    def _build_composed_connectivity_array(self, startcell, middlecell, targetcell, size, trim=False):
        carr = np.zeros((self.nkcells[startcell], size), dtype=np.int32) - 1
        ncarr = np.zeros(self.nkcells[startcell], dtype=np.int32)
        if startcell < middlecell:
            TC1 = self.higher_dim_TC
        if startcell > middlecell:
            TC1 = self.lower_dim_TC
        if middlecell < targetcell:
            TC2 = self.higher_dim_TC
        if middlecell > targetcell:
            TC2 = self.lower_dim_TC
        for p in range(self.kcells[startcell][0], self.kcells[startcell][1]):
            pc1arr = TC1(p, middlecell)
            pcarr = set()
            for m in pc1arr:
                pc2arr = TC2(m, targetcell)
                pcarr.update(list(pc2arr))
            npcarr = len(pcarr)
            carr[p - self.kcells_off[startcell], :npcarr] = np.array(list(pcarr), dtype=np.int32)
            ncarr[p - self.kcells_off[startcell]] = npcarr
        if trim:
            max = np.max(ncarr)
            carr = carr[:,:max]
        return carr, ncarr

    def get_loop_indices(self, k, celltype):
        offset = self.kcells_off[k]
        start, end = get_zerobased_loop_indices(k, celltype)
        return offset+start, offset+end

    def get_zerobased_loop_indices(self, k, celltype):

        if celltype=='all':
            start, end = 0, self.nkcells[k]

        if celltype=='I':
            start, end = 0, self.nIkcells[k]

        if celltype=='b':
            start, end = self.nIkcells[k], self.nIkcells[k] + self.nbkcells[k]

        if celltype=='B':
            start, end = self.nIkcells[k] + self.nbkcells[k], self.nkcells[k]

        if celltype=='Ib':
            start, end = 0, self.nIkcells[k] + self.nbkcells[k]

        return start,end

    def _create_stencils(self):
        if self.tdim == 1:

            self.maxEV = 2
            self.maxVE = 2

            self.VE, self.nVE = self._build_connectivity_array(1, 0, 2)
            self.EV, self.nEV = self._build_connectivity_array(0, 1, 2)

        if self.tdim == 2:

            self.maxVF, self.maxFV = self.petscmesh.getMaxSizes()
            self.maxEF, self.maxEV = self.petscmesh.getMaxSizes()

            self.maxFE = 2 #actually always true
            self.maxVE = 2 #actually always true

            self.VF, self.nVF = self._build_connectivity_array(2, 0, self.maxVF)
            self.EF, self.nEF = self._build_connectivity_array(2, 1, self.maxEF)

            self.FE, self.nFE = self._build_connectivity_array(1, 2, 2)
            self.VE, self.nVE = self._build_connectivity_array(1, 0, 2)
            self.EFP, self.nEFP = self._build_composed_connectivity_array(1, 2, 1, self.maxVF * 2 - 1)

            self.EV, self.nEV = self._build_connectivity_array(0, 1, self.maxEV)
            self.FV, self.nFV = self._build_connectivity_array(0, 2, self.maxFV)



        if self.tdim == 3:

            self.maxCF = 2 #actually always this
            self.maxVE = 2 #actually always this

            max1, max2 = self.petscmesh.getMaxSizes()
#THIS IS PROBABLY A LITTLE AGGRESSIVE BUT IT WILL WORK AT LEAST?
#IDEALLY WHAT WE DO IS PROBABLY TWO LOOPS- ONE TO COMPUTE MAX SIZE FOR CONNECTIVITY
#AND THEN ONE TO ACTUALLY CREATE THE ARRAYS
            maxsize = max(max1, max2) * 3

            self.VC, self.nVC = self._build_connectivity_array(3, 0, maxsize, trim=True)
            self.EC, self.nEC = self._build_connectivity_array(3, 1, maxsize, trim=True)
            self.FC, self.nFC = self._build_connectivity_array(3, 2, maxsize, trim=True)

            self.VF, self.nVF = self._build_connectivity_array(2, 0, maxsize, trim=True)
            self.EF, self.nEF = self._build_connectivity_array(2, 1, maxsize, trim=True)
            self.CF, self.nCF = self._build_connectivity_array(2, 3, maxsize, trim=True)

            self.VE, self.nVE = self._build_connectivity_array(1, 0, maxsize, trim=True)
            self.FE, self.nFE = self._build_connectivity_array(1, 2, maxsize, trim=True)
            self.CE, self.nCE = self._build_connectivity_array(1, 3, maxsize, trim=True)

            self.EV, self.nEV = self._build_connectivity_array(0, 1, maxsize, trim=True)
            self.FV, self.nFV = self._build_connectivity_array(0, 2, maxsize, trim=True)
            self.CV, self.nCV = self._build_connectivity_array(0, 3, maxsize, trim=True)

            self.maxVC = np.max(self.nVC)
            self.maxEC = np.max(self.nEC)
            self.maxFC = np.max(self.nFC)
            self.maxVF = np.max(self.nVF)
            self.maxEF = np.max(self.nEF)
            self.maxFE = np.max(self.nFE)
            self.maxCE = np.max(self.nCE)
            self.maxEV = np.max(self.nEV)
            self.maxFV = np.max(self.nFV)
            self.maxCV = np.max(self.nCV)

            #NOT SURE WHAT ELSE NEEDS TO BE ADDED HERE!
            #IE SOME SORT OF COMPOSED STENCILS ALSO?

#SHOULD BE ABLE TO CREATE/LOAD MESH IN SERIAL, AND THEN DISTRIBUTE INTO PARALLEL...
    def _distribute_mesh(self):
        pass

    def _create_incidence_numbers(self):

        self.incidence_numbers = []
        for d in range(self.tdim+1):
            self.incidence_numbers.append([[],[]])

        self.incidence_numbers[0][0] = None
        self.incidence_numbers[0][1] = np.zeros((self.nkcells[0], self.maxEV), dtype=np.int32)
        self.incidence_numbers[1][0] = np.zeros((self.nkcells[1], self.maxVE), dtype=np.int32)
        self.incidence_numbers[self.tdim][1] = None

        if self.tdim >= 2:
            self.incidence_numbers[1][1] = np.zeros((self.nkcells[1], self.maxFE), dtype=np.int32)
            self.incidence_numbers[2][0] = np.zeros((self.nkcells[2], self.maxEF), dtype=np.int32)

        if self.tdim == 3:
            self.incidence_numbers[2][1] = np.zeros((self.nkcells[2], self.maxCF), dtype=np.int32)
            self.incidence_numbers[3][0] = np.zeros((self.nkcells[3], self.maxFC), dtype=np.int32)

    def lower_incidence_numbers(self, p):
        depth = self.petscmesh.getLabelValue('depth',p)
        return self.incidence_numbers[depth][0][p - self.kcells_off[depth]]

    def higher_incidence_numbers(self, p):
        depth = self.petscmesh.getLabelValue('depth',p)
        return self.incidence_numbers[depth][1][p - self.kcells_off[depth]]

    def lower_dim_TC(self, p, d):
        pdepth = self.petscmesh.getLabelValue('depth',p)
        #required since sets don't preserve ordering!
        if pdepth-1 == d:
            tc = self.petscmesh.getCone(p)
        else:
            tc = set()
            dcells = set()
            dcells.add(p)
            for depth in range(pdepth-1,d-1,-1):
                new_dcells = set()
                for cell in dcells:
                    cones = self.petscmesh.getCone(cell)
                    if (depth == d): tc.update(set(cones))
                    new_dcells.update(cones)
                dcells = new_dcells.copy()
        return np.array(list(tc), dtype=np.int32)

#OLD CODE THAT NEW PETSC BROKE ON MESHES WITH ARBITRARY POLYGONAL CELLS
        # TC = self.petscmesh.getTransitiveClosure(p, useCone=True)
        # tc = []
        # for p in TC[0]:
        #     depth = self.petscmesh.getLabelValue('depth',p)
        #     if (depth == d): tc.append(p)
        # return np.array(tc, dtype=np.int32)

    def higher_dim_TC(self, p, d):
        pdepth = self.petscmesh.getLabelValue('depth',p)
        #required since sets don't preserve ordering!
        if pdepth+1 == d:
            tc = self.petscmesh.getSupport(p)
        else:
            tc = set()
            dcells = set()
            dcells.add(p)
            for depth in range(pdepth+1,d+1):
                new_dcells = set()
                for cell in dcells:
                    supports = self.petscmesh.getSupport(cell)
                    if (depth == d): tc.update(set(supports))
                    new_dcells.update(supports)
                dcells = new_dcells.copy()
        return np.array(list(tc), dtype=np.int32)

# need to output sizing + ranges, upper/lower TC + orientations, anything else?
#SHOULD HAVE EVERYTHING NEEDED TO SEAMLESSLY RECREATE MESH, AND ALSO DO VARIOUS PLOTTING/ANALYSIS STUFF
    def output_mesh(self, meshfile):
        pass
