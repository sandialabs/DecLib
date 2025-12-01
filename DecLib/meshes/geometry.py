from DecLib import PETSc
from math import sqrt, isfinite, factorial, cos, sin, acos
import numpy as np
from DecLib.meshes.topology import _orient_face


#in 1D
# vertex locations
# edge centroids
# edge segments
# face normals

# vertex quad pts + wts
# edge quad pts + wts + face normals (all per segment)

#in 2D
# vertex locations
# edge centroids
# edge segments
# face centroids
# face triangles (keyed to a given edge)
# face circumcenters (if simplicial)
# edge tangents
# face normals

# vertex quad pts + wts
# edge quad pts + wts + edge tangents + face normals (all per segment)
# face quad pts + wts (all per triangle)

#in 3D
# vertex locations
# edge centroids
# edge segments
# face centroids
# face triangles
# cell centroids
# cell tetrahedra (keyed to a given face)
# cell circumcenters (if simplicial)
# edge tangents
# face normals

# vertex quad pts + wts
# edge quad pts + wts + edge tangents (all per segment)
# face quad pts + wts + face normals (all per triangle)
# cell quad pts + wts (all per tetrahedra)


#Basic idea is that all faces are divded into triangles, and all cells are divided into tetrahedra
#This can be highly suboptimal in terms of #pts, but it works!
#Might lead to inefficiencies when fields have to be set at every time step, but this is fine
#Also might fail for certain concave polyhedra

xhat = np.array([1.0, 0.0, 0.0])
yhat = np.array([0.0, 1.0, 0.0])
zhat = np.array([0.0, 0.0, 1.0])

def _outside_bound_box(pts, lower, upper):
    outside = False
    for d in range(pts.shape[1]):
        larger = np.any(np.greater(pts[:,d], upper[d]))
        smaller = np.any(np.less(pts[:,d], lower[d]))
        outside = outside or larger or smaller
    return outside

def _signed_polygon_area2D(vertices):
    area = 0.0
    for i in range(vertices.shape[0]-1):
        area = area + (vertices[i,0]*vertices[i+1,1] - vertices[i+1,0]*vertices[i,1])
    return 0.5 * area

def _polygon_centroid2D(vertices):
    area = _signed_polygon_area2D(vertices)
    coords = np.zeros(vertices.shape[1])
    for i in range(vertices.shape[0]-1):
        coords[0] = coords[0] + (vertices[i,0]+vertices[i+1,0]) * (vertices[i,0]*vertices[i+1,1]-vertices[i+1,0]*vertices[i,1])
        coords[1] = coords[1] + (vertices[i,1]+vertices[i+1,1]) * (vertices[i,0]*vertices[i+1,1]-vertices[i+1,0]*vertices[i,1])
    return 1./6./area * coords


def _polyhedral_centroid(faces, orients):
    vol = 0.0
    coords = np.zeros(faces.shape[2])
    for i in range(faces.shape[0]):
        a = faces[i, 0, :]
        b = faces[i, 1, :]
        c = faces[i, 2, :]
        n = np.cross(b - a,c - a) * orients[i]
        vol = vol + np.dot(a,n)/6.
        coords = coords + np.multiply(n,np.square(a+b) + np.square(b+c) + np.square(c+a))
    return coords / (48.*vol)

#this assumes an orientation for faces that comes from edge ordering
#we correct it in the face normal code below, where it is weighted by the relative
#orientation of faces to edges
def _face_normal(facetriangle):
    fnorm = np.cross(facetriangle[1,:] - facetriangle[0,:], facetriangle[2,:] - facetriangle[0,:])
    return fnorm / np.linalg.norm(fnorm)


def _compute_circumcenter2D(vertices):
    ax = vertices[0,0]
    ay = vertices[0,1]
    bx = vertices[1,0]
    by = vertices[1,1]
    cx = vertices[2,0]
    cy = vertices[2,1]
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return np.array((ux, uy))

def _compute_circumcenter3D(vertices):
    pass



def _ordered_padded_face_vertices(edgesegments, nsegements, edgeordering, orients):
#loop over edges using edgeordering

    faceverts = []
    for k in edgeordering:
        segment = edgesegments[k]
        orient = orients[k]
        segrange = range(nsegements[k])
        vertexind = 0
        if orient == -1:
            segrange = segrange[::-1]
            vertexind = 1
        for p in segrange:
            faceverts.append(segment[p,vertexind,:])
    faceverts = np.array(faceverts)
    padded_faceverts = np.zeros((faceverts.shape[0]+1, faceverts.shape[1]))
    padded_faceverts[:-1,:] = faceverts[:,:]
    padded_faceverts[-1,:] = faceverts[0,:]

    return padded_faceverts


class _PlanarMetric():
#if provided, higher_dim_geom is edge_segments, num_edge_segments, face_triangles, cell_tetrahedra
#can be empty as needed ie 2D has cell_tetrahedra = None
    def __init__(self, topo, vertexcoords, higher_dim_geom=None):
        self.topo = topo
        self.vertexcoords = vertexcoords
        self.gdim = self.topo.tdim

        self.centroids = []

        self._compute_vertex_centroids()

        if higher_dim_geom is None:
            self.edge_segments, self.num_edge_segments, self.face_triangles, self.cell_tetrahedra = self._fill_higher_dim_geom()
            self._compute_edge_centroids()
            if self.topo.tdim>=2: self._compute_face_centroids()
            if self.topo.tdim==3: self._compute_cell_centroids()

        else:
            self.edge_segments, self.num_edge_segments, self.face_triangles, self.cell_tetrahedra = higher_dim_geom[0], higher_dim_geom[1], higher_dim_geom[2], higher_dim_geom[3]
            self._compute_edge_centroids()
            if self.topo.tdim>=2: self._copy_face_centroid()
            if self.topo.tdim==3: self._copy_cell_centroids()

        self._generate_edgetangents_facenormals()

#EXTEND TO 3D CASE ALSO
# See https://math.stackexchange.com/questions/2414640/circumsphere-of-a-tetrahedron
        if self.topo.is_simplicial and topo.tdim==2:
            self._fill_circumcenters2D()

        if self.topo.is_simplicial and topo.tdim==3:
            self._fill_circumcenters3D()

#REALLY WHAT WE NEED IS CODE TO EXTRACT SIMPLICIAL VERTICES
#AND THEN SOME GENERAL CODE HERE TO GET SIMPLEX CIRCUMCENTERS
    def _fill_circumcenters2D(self):

        self.circumcenters = np.zeros((self.topo.nkcells[2], self.gdim))
        foff = self.topo.kcells_off[2]
        eoff = self.topo.kcells_off[1]

        for f in range(self.topo.kcells[2][0], self.topo.kcells[2][1]):
            edgesegments = []
            VElist = []
            nsegements = []
            for eind,e in enumerate(self.topo.EF[f-foff, :self.topo.nEF[f-foff]]):
                VElist.append(self.topo.VE[e-eoff, :self.topo.nVE[e-eoff]])
                edgesegments.append(self.face_triangles[f-foff, eind, :, :2, :])
                nsegements.append(self.num_edge_segments[e-eoff])
            orients, edgeordering = _orient_face(VElist)
            facevertices = _ordered_padded_face_vertices(edgesegments, nsegements, edgeordering, orients)
            self.circumcenters[f-foff, :] = _compute_circumcenter2D(facevertices[:-1,:])

#BROKEN
#REALLY SHOULD JUST LOOP OVER CELL TETRAHEDRA AND GET UNIQUE VERTICES...
    def _fill_circumcenters3D(self):
        self.circumcenters = np.zeros((self.topo.nkcells[3], self.gdim))
        coff = self.topo.kcells_off[3]
        foff = self.topo.kcells_off[2]
        print('filling 3D circumcenters- BROKEN')
        # for c in range(self.topo.kcells[3][0], self.topo.kcells[3][1]):
        #     edgesegments = []
        #     VElist = []
        #     nsegements = []
        #     for find,f in enumerate(self.topo.FC[c-coff, :self.topo.nFC[c-coff]]):
        #         VElist.append(self.topo.VE[e-eoff, :self.topo.nVE[e-eoff]])
        #         edgesegments.append(self.face_triangles[f-foff, eind, :, :2, :])
        #         nsegements.append(self.num_edge_segments[e-eoff])
        #     orients, edgeordering = _orient_face(VElist)
        #     facevertices = _ordered_padded_face_vertices(edgesegments, nsegements, edgeordering, orients)
        #     self.circumcenters[f-foff, :] = _compute_circumcenter3D(facevertices[:-1,:])


    def _compute_vertex_centroids(self):

        self.centroids.append(np.zeros((self.topo.nkcells[0], self.gdim)))

        self.centroids[0][:, :] = self.vertexcoords[:, :]

#THERE SHOULD BE A BETTER WAY TO DO THESE- PROBABLY VIA NUMPY
#IE PYTHON LOOP FREE!
#WE SHOULD BE ABLE TO MAKE MOST/ALL OF THESE CALCULATIONS AUTOMATIC...
#JUST BE CAREFUL WITH "EMPTY"/NON-EXISTENT STUFF?

    def _compute_edge_centroids(self):

        self.centroids.append(np.zeros((self.topo.nkcells[1], self.edge_segments.shape[1], self.gdim)))

        eoff = self.topo.kcells_off[1]

        for e in range(self.topo.kcells[1][0], self.topo.kcells[1][1]):
            for p in range(self.num_edge_segments[e-eoff]):
                segment = self.edge_segments[e - eoff, p, :, :]
                segment_midpoint = (segment[0,:] + segment[1,:])/2.
                self.centroids[1][e-eoff, p, :] = segment_midpoint


    def _copy_face_centroids(self):
        self.centroids.append(np.zeros((self.topo.nkcells[2], self.gdim)))
        self.centroids[:, :] = self.face_triangles[:, 0, 2, :]

    def _copy_cell_centroids(self):
        self.centroids.append(np.zeros((self.topo.nkcells[3], self.gdim)))
        self.centroids[:, :] = self.cell_tetrahedra[:, 0, 0, 0, 3, :]




    def _generate_edgetangents_facenormals(self):

        #create edgetangents
        self.edgetangents = None
        if self.topo.tdim >=2:

            eStart, eEnd = self.topo.kcells[1][0], self.topo.kcells[1][1]
            ne = eEnd - eStart
            eoff = self.topo.kcells_off[1]

            self.edge_tangents = np.zeros((ne, self.edge_segments.shape[1], self.gdim))

            for e in range(eStart, eEnd):
                orient = self.topo.lower_incidence_numbers(e)
                for p in range(self.num_edge_segments[e - eoff]):
                    etan = self.edge_segments[e - eoff, p, 1, :] - self.edge_segments[e - eoff, p, 0, :]
                    self.edge_tangents [e-eoff, p, :] = etan/np.sqrt(np.dot(etan,etan))

#TRY TO WRITE SOME IMPLICIT NUMPY INDEXING EXPRESSIONS HERE

        if self.topo.tdim == 1:
            self.face_normals = np.zeros((self.topo.nkcells[0], self.gdim))
            self.face_normals[:,0] = 1.0

        if self.topo.tdim == 2:
            eoff = self.topo.kcells_off[1]
            foff = self.topo.kcells_off[2]
            self.face_normals = np.zeros((self.topo.nkcells[1], self.edge_segments.shape[1], self.gdim))
            for e in range(self.topo.kcells[1][0], self.topo.kcells[1][1]):
                for p in range(self.num_edge_segments[e-eoff]):
                    etan = self.edge_tangents[e-eoff, p, :]
                    fnorm1 = np.array((-etan[1], etan[0])) #This is a righthand rule
                    self.face_normals[e - eoff, p, :] = fnorm1

        if self.topo.tdim == 3:
            foff = self.topo.kcells_off[2]
            coff = self.topo.kcells_off[3]
            self.face_normals = np.zeros((self.topo.nkcells[2], self.face_triangles.shape[1], self.face_triangles.shape[2], self.gdim))
            for f in range(self.topo.kcells[2][0], self.topo.kcells[2][1]):

                edgesegments = []
                VElist = []
                nsegements = []
                for eind,e in enumerate(self.topo.EF[f-foff, :self.topo.nEF[f-foff]]):
                    VElist.append(self.topo.VE[e-eoff, :self.topo.nVE[e-eoff]])
                    edgesegments.append(self.face_triangles[f-foff, eind, :, :2, :])
                    nsegements.append(self.num_edge_segments[e-eoff])
                orients, edgeordering = _orient_face(VElist)
                for eind,e in enumerate(self.topo.EF[f-foff, :self.topo.nEF[f-foff]]):
                    for p in range(self.num_edge_segments[e-eoff]):
                        facetriangle = self.face_triangles[f- foff, eind, p, :, :]
#DO THIS BASED ON ORDERED PADDED FACE VERTICES SO THAT IT IS CONSISTENT WITH ORIENT 3D CODE
                        fnorm = _face_normal(facetriangle)
                        self.face_normals[f - foff, eind, p, :] = fnorm * orients[eind]



#THERE SHOULD BE A BETTER WAY TO DO THIS- PROBABLY VIA NUMPY
#WE SHOULD BE ABLE TO MAKE MOST/ALL OF THESE CALCULATIONS AUTOMATIC...
#JUST BE CAREFUL WITH "EMPTY"/NON-EXISTENT STUFF?

class CartesianPlanarMetric(_PlanarMetric):

    def _fill_higher_dim_geom(self):

        #edge segments
        eStart, eEnd = self.topo.kcells[1][0], self.topo.kcells[1][1]
        ne = eEnd - eStart
        eoff = self.topo.kcells_off[1]
        voff = self.topo.kcells_off[0]

        edge_segments = np.zeros((ne, 1, 2, self.gdim))
        num_edge_segments = np.zeros(ne, dtype=np.int32) + 1
        face_triangles = None
        cell_tetrahedra = None

        for e in range(eStart, eEnd):
            VE = self.topo.VE[e-eoff, :self.topo.nVE[e-eoff]]
            edge_segments[e - eoff, 0, 0, :] = self.vertexcoords[VE[0] - voff,:]
            edge_segments[e - eoff, 0, 1, :] = self.vertexcoords[VE[1] - voff,:]

        #face triangles
        if self.topo.tdim >=2:
            fStart, fEnd = self.topo.kcells[2][0], self.topo.kcells[2][1]
            nf = fEnd - fStart
            foff = self.topo.kcells_off[2]
            eoff = self.topo.kcells_off[1]

            face_triangles = np.zeros((nf, self.topo.maxEF, edge_segments.shape[1], 3, self.gdim))

            for f in range(fStart, fEnd):
                EF = self.topo.EF[f-foff, :self.topo.nEF[f-foff]]
                for eind,e in enumerate(EF):
                    for p in range(num_edge_segments[e - eoff]):
                        face_triangles[f - foff, eind, p, 0, :] = edge_segments[e - eoff, p, 0, :]
                        face_triangles[f - foff, eind, p, 1, :] = edge_segments[e - eoff, p, 1, :]

        #cell tetrahedra
        if self.topo.tdim ==3:
            cStart, cEnd = self.topo.kcells[3][0], self.topo.kcells[3][1]
            nc = cEnd - cStart
            coff = self.topo.kcells_off[3]
            foff = self.topo.kcells_off[2]
            eoff = self.topo.kcells_off[1]

            cell_tetrahedra = np.zeros((nc, self.topo.maxFC, face_triangles.shape[1], face_triangles.shape[2], 4, self.gdim))

            for c in range(cStart, cEnd):
                FC = self.topo.FC[c-coff, :self.topo.nFC[c-coff]]
                for find,f in enumerate(FC):
                    EF = self.topo.EF[f-foff, :self.topo.nEF[f-foff]]
                    for eind,e in enumerate(EF):
                        for p in range(num_edge_segments[e - eoff]):
                            cell_tetrahedra[c - coff, find, eind, p, :2, :] = face_triangles[f - foff, eind, p, :2, :]


        return edge_segments, num_edge_segments, face_triangles, cell_tetrahedra

#This assumes that faces are truly planar ie not made up of facets
#If they are faceted, should feed in complete face triangles
    def _compute_face_centroids(self):
        foff = self.topo.kcells_off[2]
        eoff = self.topo.kcells_off[1]

        self.centroids.append(np.zeros((self.topo.nkcells[2], self.gdim)))
        if self.topo.tdim == 2:
            for f in range(self.topo.kcells[2][0], self.topo.kcells[2][1]):
                edgesegments = []
                VElist = []
                nsegements = []
                for eind,e in enumerate(self.topo.EF[f-foff, :self.topo.nEF[f-foff]]):
                    VElist.append(self.topo.VE[e-eoff, :self.topo.nVE[e-eoff]])
                    edgesegments.append(self.face_triangles[f-foff, eind, :, :2, :])
                    nsegements.append(self.num_edge_segments[e-eoff])
                orients, edgeordering = _orient_face(VElist)
                facevertices = _ordered_padded_face_vertices(edgesegments, nsegements, edgeordering, orients)
                #print(f, facevertices)
                centroid = _polygon_centroid2D(facevertices)
                self.centroids[2][f-foff, :] = centroid
                self.face_triangles[f-foff, :, :, 2, :] = centroid


        if self.topo.tdim == 3:

            coff = self.topo.kcells_off[3]
            foff = self.topo.kcells_off[2]
            eoff = self.topo.kcells_off[1]

            for f in range(self.topo.kcells[2][0], self.topo.kcells[2][1]):

#THIS IS HACKY AND WRONG IN GENERAL
#IT WORKS FOR SIMPLICES AND UNIFORM QUADS
#WHICH IS ENOUGH FOR NOW?
                rough_centroid = 0.0
                total_segments = 0
                for eind,e in enumerate(self.topo.EF[f-foff, :self.topo.nEF[f-foff]]):
                    for p in range(self.num_edge_segments[e-eoff]):
                        edgecentroid = (self.face_triangles[f-foff, eind, p, 0, :] + self.face_triangles[f-foff, eind, p, 1, :])/2.
                        rough_centroid = rough_centroid + edgecentroid
                        total_segments = total_segments + 1
                rough_centroid = rough_centroid / total_segments

                self.centroids[2][f-foff, :] = rough_centroid
                self.face_triangles[f-foff, :, :, 2, :] = rough_centroid

            for c in range(self.topo.kcells[3][0], self.topo.kcells[3][1]):
                for find, f in enumerate(self.topo.FC[c-coff,:self.topo.nFC[c-coff]]):
#THIS IS HACKY AND WRONG IN GENERAL
#IT WORKS FOR SIMPLICES AND UNIFORM QUADS
#WHICH IS ENOUGH FOR NOW?
                    rough_centroid = 0.0
                    total_segments = 0
                    for eind,e in enumerate(self.topo.EF[f-foff, :self.topo.nEF[f-foff]]):
                        for p in range(self.num_edge_segments[e-eoff]):
                            edgecentroid = (self.cell_tetrahedra[c-coff, find, eind, p, 0, :] + self.cell_tetrahedra[c-coff, find, eind, p, 1, :])/2.
                            rough_centroid = rough_centroid + edgecentroid
                            total_segments = total_segments + 1
                    rough_centroid = rough_centroid / total_segments
                    self.cell_tetrahedra[c-coff, find, :, :, 2, :] = rough_centroid

#it would be useful to be able to compute the actual vertex coordinates for a k-cell
#this info is easily extracted from edge_segments
#but it is less easily computed for faces and cells
#this is important for circumcenter calculations
#and also roughly estimating a cell centroid for routine below

    def _compute_cell_centroids(self):
        self.centroids.append(np.zeros((self.topo.nkcells[3], self.gdim)))

        coff = self.topo.kcells_off[3]
        foff = self.topo.kcells_off[2]
        eoff = self.topo.kcells_off[1]
        for c in range(self.topo.kcells[3][0], self.topo.kcells[3][1]):
            facetriangles = []
            orients = []
            rough_centroid = 0.0
            for find, f in enumerate(self.topo.FC[c-coff,:self.topo.nFC[c-coff]]):
                rough_centroid = rough_centroid + self.cell_tetrahedra[c-coff, find, 0, 0, 2, :]/self.topo.nFC[c-coff]
            FC = self.topo.FC[c-coff,:self.topo.nFC[c-coff]]
            nf = self.topo.nFC[c-coff]
            for find, f in enumerate(self.topo.FC[c-coff,:self.topo.nFC[c-coff]]):
                for eind, e in enumerate(self.topo.EF[f-foff,:self.topo.nEF[f-foff]]):
                    for p in range(self.num_edge_segments[e-eoff]):
                        facetriangles.append(self.cell_tetrahedra[c-coff, find, eind, p, :3, :])
                        fnorm = _face_normal(self.cell_tetrahedra[c-coff, find, eind, p, :3, :])
                        fnorm1 = rough_centroid - self.cell_tetrahedra[c-coff, find, 0, 0, 2, :]
                        #fnorm2 = self.centroids[2][FC[(find+1)%nf]-foff, :] - self.centroids[2][f-foff, :]
                        orients.append(np.sign(np.dot(fnorm,fnorm1)))
                        #orients.append(np.sign(-np.dot(fnorm,fnorm2)))
            facetriangles = np.array(facetriangles)
            orients = np.array(orients)
            centroid =  _polyhedral_centroid(facetriangles, orients)
            # if _outside_bound_box(np.reshape(centroid, (1,3)), [0,0,0],[1,1,1]) or (c-coff == 73):
            #     print(c, centroid, rough_centroid)
            #     _plot_cell(self.topo, self, c-coff, self.cell_tetrahedra[c-coff, :, :, :, :, :], centroid, rough_centroid, 'celltet-' + str(c))

                # for find, f in enumerate(self.topo.FC[c-coff,:self.topo.nFC[c-coff]]):
                #     for eind, e in enumerate(self.topo.EF[f-foff,:self.topo.nEF[f-foff]]):
                #         for p in range(self.num_edge_segments[e-eoff]):
                #             vertices = self.cell_tetrahedra[c-coff, find, eind, p, :3, :]
                #             xy = _polygon_centroid2D(vertices[:,:-1])
                #             xz = _polygon_centroid2D(vertices[:,::2])
                #             yz = _polygon_centroid2D(vertices[:,1:])
                #             print(xy,xz,yz)
            self.cell_tetrahedra[c-coff, :, :, :, 3, :] = centroid
            self.centroids[3][c-coff, :] = centroid


# #This does 1D, 2D and 3D fully periodic
# #It is for topologies that are products of S1, or their subspaces
#ACTUALLY IT SHOULD WORK FOR ANY TYPE OF PERIODICITY
#NEED TO TEST THIS THOUGH

class PeriodicPlanarMetric(CartesianPlanarMetric):

    def __init__(self, topo, vertexcoords, cellcoords, cellcoordsection, periodic_bnds):
        self.cellcoords = cellcoords
        self.cellcoordsection = cellcoordsection
        self.periodic_maxes = periodic_bnds[1]
        self.periodic_mins = periodic_bnds[0]
        _PlanarMetric.__init__(self, topo, vertexcoords)

#THIS ASSUMES THAT PERIODIC COORDINATES OCCUR ONLY AT MAX!
#THIS IS A BAD GENERAL ASSUMPTION
#SHOULD REALLY CHECK BACKWARDS AS WELL?
#OR DO SOME SORT OF CAREFUL MODULO...
#ACTUALLY YES JUST DO A FLOATING POINT MODULO?
    def _check_periodic_same(self, v0, v1):
        same = True
        for d in range(v0.shape[0]):
            v0s = np.array([v0[d], v0[d]+self.periodic_maxes[d], v0[d]   ])
            v1s = np.array([v1[d], v1[d]                       , v1[d]+self.periodic_maxes[d]])
            same = np.any(np.isclose(v0s, v1s)) and same
        return same

        # v0ys = np.array([v0[1], v0[1]+self.periodic_maxes[1], v0[1]   ])
        # v1ys = np.array([v1[1], v1[1]                       , v1[1]+self.periodic_maxes[1]])
        # v0zs = np.array([v0[2], v0[2]+self.periodic_maxes[2], v0[2]   ])
        # v1zs = np.array([v1[2], v1[2]                       , v1[2]+self.periodic_maxes[2]])
        # ysame = np.any(np.isclose(v0ys, v1ys))
        # zsame = np.any(np.isclose(v0zs, v1zs))
        # return (xsame and ysame and zsame)

    def _fill_higher_dim_geom(self):

        edge_segments, num_edge_segments, face_triangles, cell_tetrahedra = CartesianPlanarMetric._fill_higher_dim_geom(self)

        poff =  self.topo.kcells_off[self.topo.tdim]
        voff =  self.topo.kcells_off[0]
        eoff =  self.topo.kcells_off[1]

        for p in range(self.topo.kcells[self.topo.tdim][0], self.topo.kcells[self.topo.tdim][1]):
            offset = self.cellcoordsection.getOffset(p)
            ndof = self.cellcoordsection.getDof(p)
            cellcoord = self.cellcoords[offset:offset+ndof]
            if ndof > 0:

                if self.topo.tdim == 1:

                    edge_segments[p - poff, 0, :, 0] = cellcoord

                if self.topo.tdim == 2:

                    nVF = self.topo.nVF[p - poff]
                    cellcoord = np.reshape(cellcoord, (nVF, self.topo.tdim))
                    for eind,e in enumerate(self.topo.EF[p-poff, :self.topo.nEF[p-poff]]):
                        v0 = face_triangles[p-poff, eind, 0, 0, :]
                        v1 = face_triangles[p-poff, eind, 0, 1, :]
                        #locate v0 and v1 in cellcoord away
                        adjusted_v0 = v0.copy()
                        adjusted_v1 = v1.copy()
                        changed_v0 = False
                        changed_v1 = False
                        for k in range(nVF):
                            vk = cellcoord[k, :]
                            if self._check_periodic_same(v0, vk):
                                adjusted_v0[:] = vk[:]
                                changed_v0 = np.allclose(adjusted_v0, v0)
                            if self._check_periodic_same(v1, vk):
                                adjusted_v1[:] = vk[:]
                                changed_v1 = np.allclose(adjusted_v1, v1)
                        face_triangles[p - poff, eind, 0, 0, :] = adjusted_v0
                        face_triangles[p - poff, eind, 0, 1, :] = adjusted_v1
                        if changed_v0 != changed_v1:
                            edge_segments[e-eoff, 0, 0, :] = adjusted_v0
                            edge_segments[e-eoff, 0, 1, :] = adjusted_v1

                if self.topo.tdim == 3:
                    nVC = self.topo.nVC[p - poff]
                    cellcoord = np.reshape(cellcoord, (nVC, self.topo.tdim))
                    foff =  self.topo.kcells_off[2]

                    for find,f in enumerate(self.topo.FC[p-poff, :self.topo.nFC[p-poff]]):

#THIS DOES MOVE faces from "0" to "1"
#Not sure if this is an issue, since by definition these are the same in periodic coords
#It does get correct locations for cell tetrahedra, which is what matters?
                        #get face vertices
                        edgesegments = []
                        VElist = []
                        nsegements = []
                        for eind,e in enumerate(self.topo.EF[f-foff, :self.topo.nEF[f-foff]]):
                            VElist.append(self.topo.VE[e-eoff, :self.topo.nVE[e-eoff]])
                            edgesegments.append(face_triangles[f-foff, eind, :, :2, :])
                            nsegements.append(num_edge_segments[e-eoff])
                        orients, edgeordering = _orient_face(VElist)
                        facevertices = _ordered_padded_face_vertices(edgesegments, nsegements, edgeordering, orients)
                        #determine if this face has all fake vertices
                        #all_changed is True if ALL the vertices are "fake"
                        all_changed = True
                        for v in facevertices:
                            adjusted_v = v.copy()
                            for k in range(nVC):
                                vk = cellcoord[k, :]
                                if self._check_periodic_same(v, vk):
                                    adjusted_v[:] = vk[:]
                                    all_changed = np.allclose(adjusted_v, v) and all_changed

#fix cell tetrahedra
#fix face triangles (IFF needed)
#fix edges (IFF needed)

                        for eind,e in enumerate(self.topo.EF[f-foff, :self.topo.nEF[f-foff]]):
                            v0 = cell_tetrahedra[p-poff, find, eind, 0, 0, :]
                            v1 = cell_tetrahedra[p-poff, find, eind, 0, 1, :]
                            adjusted_v0 = v0.copy()
                            adjusted_v1 = v1.copy()
                            changed_v0 = False
                            changed_v1 = False
                            for k in range(nVC):
                                vk = cellcoord[k, :]
                                if self._check_periodic_same(v0, vk):
                                    adjusted_v0[:] = vk[:]
                                    changed_v0 = np.allclose(adjusted_v0, v0)
                                if self._check_periodic_same(v1, vk):
                                    adjusted_v1[:] = vk[:]
                                    changed_v1 = np.allclose(adjusted_v1, v1)
                            cell_tetrahedra[p - poff, find, eind, 0, 0, :] = adjusted_v0
                            cell_tetrahedra[p - poff, find, eind, 0, 1, :] = adjusted_v1
                            if changed_v0 != changed_v1:
                                 edge_segments[e-eoff, 0, 0, :] = adjusted_v0
                                 edge_segments[e-eoff, 0, 1, :] = adjusted_v1
                            if not all_changed:
                                face_triangles[f-foff, eind, 0, 0, :] = adjusted_v0
                                face_triangles[f-foff, eind, 0, 1, :] = adjusted_v1

        return edge_segments, num_edge_segments, face_triangles, cell_tetrahedra
