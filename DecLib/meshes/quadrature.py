from DecLib import PETSc
from math import sqrt, isfinite, factorial, cos, sin, acos
import modepy as mp
import numpy as np









def get_reference_quadrature(order, k):
    if k==0:
        return 1.,1.
    if k==1:
        pts,wts = np.polynomial.legendre.leggauss(order) #on [-1,1]
        pts,wts = 0.5*pts + 0.5, wts*0.5 #rescales quadrature to live on [0,1]
        return pts,wts
    if k==2:
        quad = mp.GrundmannMoellerSimplexQuadrature(order, 2)
        return quad.nodes.T, quad.weights.T
    if k==3:
        quad = mp.GrundmannMoellerSimplexQuadrature(order, 3)
        return quad.nodes.T, quad.weights.T


def _interval_quad(line_pts, pts, wts):
    dim = line_pts.shape[1]
    qpts = np.zeros((pts.shape[0],dim))
    qwts = np.zeros(pts.shape[0])
    diff = line_pts[1,:] - line_pts[0,:]
    for q in range(qpts.shape[0]):
        qpts[q,:] = line_pts[0,:] + pts[q] * diff
    scale = np.sqrt(np.sum(np.square(diff), axis=-1))
    qwts = scale * wts
    return qpts, qwts





def _simplex_volume(simplex_pts):
    cayley_menger = np.zeros((simplex_pts.shape[0]+1,simplex_pts.shape[0]+1))

    #np.fill_diagonal(cayley_menger, 0.0)
    for i in range(simplex_pts.shape[0]):
        for j in range(simplex_pts.shape[0]):
            cayley_menger[i+1,j+1] = np.linalg.norm(simplex_pts[i] - simplex_pts[j])
    cayley_menger = np.square(cayley_menger)
    cayley_menger[0,:] = 1.0
    cayley_menger[:,0] = 1.0
    cayley_menger[0,0] = 0.0
    n = simplex_pts.shape[0] - 1
    scalefactor = ((-1.)**(n+1))/(factorial(n)*factorial(n)*(2**n))
    return sqrt(scalefactor * np.linalg.det(cayley_menger))


#^ s
#|
#C
#|\
#| \
#|  O
#|   \
#|    \
#A-----B--> r
#O = ( 0,  0)
#A = (-1, -1)
#B = ( 1, -1)
#C = (-1,  1)


#THERE SHOULD BE A BETTER WAY TO DO THIS- PROBABLY VIA NUMPY
#WE SHOULD BE ABLE TO MAKE MOST/ALL OF THESE CALCULATIONS AUTOMATIC...
#JUST BE CAREFUL WITH "EMPTY"/NON-EXISTENT STUFF?

ref_tri = np.array([[-1.,-1.], [1.,-1.], [-1.,1]])
ref_area = _simplex_volume(ref_tri)
Mref_tri = np.linalg.inv(np.array([[-1, 1., -1.],[-1., -1., 1.],[1., 1., 1.]]))


#            ^ s
#            |
#            C
#           /|\
#          / | \
#         /  |  \
#        /   |   \
#       /   O|    \
#      /   __A-----B---> r
#     /_--^ ___--^^
#    ,D--^^^
# t L
# O = ( 0,  0,  0)
# A = (-1, -1, -1)
# B = ( 1, -1, -1)
# C = (-1,  1, -1)
# D = (-1, -1,  1)


ref_tet = np.array([[-1.,-1.,-1.], [1.,-1.,-1.], [-1.,1,-1.],[-1.,-1.,1.]])
ref_volume = _simplex_volume(ref_tet)
#M is [x0, x1, x2],[y0, y1, y2, etc.]
#final row is 1,1,1,1,etc.
Mref_tet = np.zeros((4,4))
Mref_tet[:3,:] = ref_tet.T
Mref_tet[3,:] = 1.
Mref_tet = np.linalg.inv(Mref_tet)

#define an affine transformation
def _affine_transform(simplex_pts, Mrefsimplex):
    Mpts = np.zeros((simplex_pts.shape[0], simplex_pts.shape[0]))
    Mpts[:-1, :] = simplex_pts.T
    Mpts[-1, :] = 1.0
    return np.dot(Mpts,Mrefsimplex)

def _subsimplex_quad(subsimplex_pts, simplex_pts, pts, wts, Mref_simplex, ref_volume):
    qpts = np.zeros(pts.shape)
    qwts = np.zeros(wts.shape)
    volume = _simplex_volume(subsimplex_pts)
    M = _affine_transform(simplex_pts, Mref_simplex)
    qwts[:] = wts[:] * volume/ref_volume
    for q in range(pts.shape[0]):
        pts_aug = np.zeros(pts.shape[1]+1) + 1.
        pts_aug[:-1] = pts[q,:]
        qpts[q,:] = np.dot(M, pts_aug)[:-1]
    return qpts, qwts

def _simplex_quad(simplex_pts, pts, wts, Mref_simplex, ref_volume):
    qpts = np.zeros(pts.shape)
    qwts = np.zeros(wts.shape)
    volume = _simplex_volume(simplex_pts)
    M = _affine_transform(simplex_pts, Mref_simplex)
    qwts[:] = wts[:] * volume/ref_volume
    for q in range(pts.shape[0]):
        pts_aug = np.zeros(pts.shape[1]+1) + 1.
        pts_aug[:-1] = pts[q,:]
        qpts[q,:] = np.dot(M, pts_aug)[:-1]
    return qpts, qwts

class _Quadrature():
    def __init__(self, topo, geom, quadorder):
        self.quadorder = quadorder
        self.quadwts = []
        self.quadpts = []
        self.topo = topo
        self.geom = geom

        refpts, refwts = get_reference_quadrature(quadorder, 0)
        vertexquadpts = np.zeros(geom.vertexcoords.shape)
        vertexquadwts = np.zeros(geom.vertexcoords.shape[0])
        self.quadpts.append(vertexquadpts)
        self.quadwts.append(vertexquadwts)
        self._fill_vertex_quad(refpts, refwts)

        nsegments = geom.edge_segments.shape[1]
        refpts, refwts = get_reference_quadrature(quadorder, 1)
        nquad = refpts.shape[0]
        edgequadpts = np.zeros((topo.nkcells[1], nsegments, nquad, self.geom.gdim))
        edgequadwts = np.zeros((topo.nkcells[1], nsegments, nquad))
        self.quadpts.append(edgequadpts)
        self.quadwts.append(edgequadwts)
        self._fill_edge_quad(refpts, refwts)

        if self.topo.tdim >=2:
            refpts, refwts = get_reference_quadrature(quadorder, 2)
            if self.topo.tdim == 3:
                newrefpts = np.zeros((refpts.shape[0], 3))
                newrefpts[:, :2] = refpts[:]
                newrefpts[:, 2] = -1. #makes ref tri part of ref tet so affine transform works
                refpts = newrefpts
            nquad2 = refpts.shape[0]
            facequadpts = np.zeros((topo.nkcells[2], topo.maxEF, nsegments, nquad2, self.geom.gdim))
            facequadwts = np.zeros((topo.nkcells[2], topo.maxEF, nsegments, nquad2))
            self.quadpts.append(facequadpts)
            self.quadwts.append(facequadwts)
            self.edge_tangents = np.zeros((topo.nkcells[1], nsegments, nquad, self.geom.gdim))
            self._fill_face_quad(refpts, refwts)
            self._fill_edge_tangents()

        if self.topo.tdim >=3:
            refpts, refwts = get_reference_quadrature(quadorder, 3)
            nquad3 = refpts.shape[0]
            cellquadpts = np.zeros((topo.nkcells[3], topo.maxFC, topo.maxEF, nsegments, nquad3, self.geom.gdim))
            cellquadwts = np.zeros((topo.nkcells[3], topo.maxFC, topo.maxEF, nsegments, nquad3))
            self.quadpts.append(cellquadpts)
            self.quadwts.append(cellquadwts)
            self._fill_cell_quad(refpts, refwts)

        if self.topo.tdim == 1:
            self.face_normals = np.zeros((topo.nkcells[0], self.geom.gdim))
        if self.topo.tdim == 2:
            self.face_normals = np.zeros((topo.nkcells[1], nsegments, nquad, self.geom.gdim))
        if self.topo.tdim == 3:
            self.face_normals = np.zeros((topo.nkcells[2], topo.maxEF, nsegments, nquad2, self.geom.gdim))

        self._fill_face_normals()

#DESPERATELY NEEDS TO BE WRAPPED IN NUMBA OR SOMETHING EFFICIENT

#THIS NEEDS TO BE CLEANED UP FOR BUNDLE VALUED FORMS
#Here what we do is take dot product of basis with values? unclear...
    def set_kform(self, ndofs, nelems, bundle, k, formfunclist, var, type='I', linetype='tangent', t=None, force_scalar_avg=False):
        quadpts = self.quadpts[k]
        quadwts = self.quadwts[k]

        dim = quadpts.shape[-1]
        nelems = quadpts.shape[0]
        nquad = np.prod(quadpts.shape)//nelems//dim
        quadpts = np.reshape(quadpts, (nelems, nquad, dim))
        quadwts = np.reshape(quadwts, (nelems, nquad))

        vararr = var.getArray()
        vararr = vararr.reshape(nelems, ndofs, bundle.size())

        start, end = self.topo.get_zerobased_loop_indices(k, type)

        if (force_scalar_avg):
#BROKEN SINCE THERE IS NO CELLSIZES
            for i in range(start,end):
                qpts = quadpts[i,:,:]
                qwts = quadwts[i,:]
                for l in range(ndofs):
                    val = 0.
                    for q in range(nquad):
                        val = val + formfunclist[l](t, *qpts[q,:]) * qwts[q]
                    vararr[i,l] = val/cellsizes[i]

        elif (k==self.topo.tdim or k==0):
            bsize = bundle.size()
            bbasis = bundle.basis()
            for i in range(start,end):
                qpts = quadpts[i,:,:]
                qwts = quadwts[i,:]
                val = np.zeros((ndofs,bsize))
                for l in range(ndofs):
                    for d in range(bsize):
                        for q in range(nquad):
                            val[l,d] = val[l,d] + np.dot(formfunclist[l](t, *qpts[q,:]), bbasis[d]) * qwts[q]
                vararr[i,:,:] = val

#BUNDLE-VALUED 1 and 2 FORMS!
#For T and C these are like 2-tensors
#There is a choice about which indices in the 2-tensors
#pair with bundle basis, and which pair with form basis?

        elif ((k==1 and linetype=='tangent' and self.topo.tdim==2) or (k==1 and self.topo.tdim==3)) and bundle.name == 'real':
            bsize = bundle.size()
            bbasis = bundle.basis()
            edge_tangents = self.edge_tangents.reshape((nelems,nquad,dim))
            for i in range(start,end):
                qpts = quadpts[i,:,:]
                qwts = quadwts[i,:]
                etan = edge_tangents[i,:,:]
                val = np.zeros((ndofs,bsize))
                for l in range(ndofs):
                    for d in range(bsize):
                        for q in range(nquad):
                            val[l,d] = val[l,d] + np.dot(formfunclist[l](t, *qpts[q,:]), etan[q,:]) * qwts[q]
                vararr[i,:,:] = val

        elif ((k==1 and linetype=='normal' and self.topo.tdim==2) or (k==2 and self.topo.tdim==3)) and bundle.name == 'real':
            #DONT ACTUALLY WANT TO RESHAPE- JUST WANT A VIEW...
            bsize = bundle.size()
            bbasis = bundle.basis()
            face_normals = self.face_normals.reshape((nelems,nquad,dim))
            for i in range(start,end):
                qpts = quadpts[i,:,:]
                qwts = quadwts[i,:]
                fnorm = face_normals[i,:,:]
                val = np.zeros((ndofs,bsize))
                for l in range(ndofs):
                    for d in range(bsize):
                        for q in range(nquad):
                            val[l,d] = val[l,d] + np.dot(formfunclist[l](t, *qpts[q,:]), fnorm[q,:]) * qwts[q]
                vararr[i,:,:] = val

        else:
            exit('setting values not implemented for for k=' + str(k) + ' and dim=' + str(self.topo.tdim) + ' and bundle=' + bundle.name + ' and linetype=' + str(linetype))

        var.assemble()

        return var

class PlanarQuadrature(_Quadrature):
    def _fill_vertex_quad(self, refpts, refwts):
        self.quadpts[0][:,:] = self.geom.vertexcoords[:,:]
        self.quadwts[0] = np.zeros(self.geom.vertexcoords.shape[0]) + 1.

    def _fill_edge_quad(self, refpts, refwts):
        eoff = self.topo.kcells_off[1]
        for e in range(self.topo.kcells[1][0], self.topo.kcells[1][1]):
            for p in range(self.geom.num_edge_segments[e - eoff]):
                pts,wts = _interval_quad(self.geom.edge_segments[e - eoff, p, :, :], refpts, refwts)
                self.quadpts[1][e - eoff, p, :, :] = pts
                self.quadwts[1][e - eoff, p, :] = wts

    def _fill_face_quad(self, refpts, refwts):
        if self.topo.tdim == 2:
            foff = self.topo.kcells_off[2]
            eoff = self.topo.kcells_off[1]
            for f in range(self.topo.kcells[2][0], self.topo.kcells[2][1]):
                EF = self.topo.EF[f-foff, :self.topo.nEF[f-foff]]
                for eind,e in enumerate(EF):
                    for p in range(self.geom.num_edge_segments[e - eoff]):
                        pts,wts = _simplex_quad(self.geom.face_triangles[f - foff, eind, p, :, :], refpts, refwts, Mref_tri, ref_area)
                        self.quadpts[2][f - foff, eind, p, :, :] = pts
                        self.quadwts[2][f - foff, eind, p, :] = wts

        if self.topo.tdim == 3:

            coff = self.topo.kcells_off[3]
            foff = self.topo.kcells_off[2]
            eoff = self.topo.kcells_off[1]
            for c in range(self.topo.kcells[3][0], self.topo.kcells[3][1]):
                for find,f in enumerate(self.topo.FC[c-coff, :self.topo.nFC[c-coff]]):
                    for eind,e in enumerate(self.topo.EF[f-foff, :self.topo.nEF[f-foff]]):
                        for p in range(self.geom.num_edge_segments[e - eoff]):
                            pts,wts = _subsimplex_quad(self.geom.face_triangles[f - foff, eind, p, :, :], self.geom.cell_tetrahedra[c - coff, find, eind, p, :, :], refpts, refwts, Mref_tet, ref_area)
                            self.quadpts[2][f - foff, eind, p, :, :] = pts
                            self.quadwts[2][f - foff, eind, p, :] = wts


    def _fill_cell_quad(self, refpts, refwts):
        coff = self.topo.kcells_off[3]
        foff = self.topo.kcells_off[2]
        eoff = self.topo.kcells_off[1]
        for c in range(self.topo.kcells[3][0], self.topo.kcells[3][1]):
            for find,f in enumerate(self.topo.FC[c-coff, :self.topo.nFC[c-coff]]):
                for eind,e in enumerate(self.topo.EF[f-foff, :self.topo.nEF[f-foff]]):
                    for p in range(self.geom.num_edge_segments[e - eoff]):
                        pts,wts = _simplex_quad(self.geom.cell_tetrahedra[c - coff, find, eind, p, :, :], refpts, refwts, Mref_tet, ref_volume)
                        self.quadpts[3][c - coff, find, eind, p, :, :] = pts
                        self.quadwts[3][c - coff, find, eind, p, :] = wts

    def _fill_edge_tangents(self):

        eoff = self.topo.kcells_off[1]
        for e in range(self.topo.kcells[1][0], self.topo.kcells[1][1]):
            for p in range(self.geom.num_edge_segments[e - eoff]):
                self.edge_tangents[e - eoff, p, :, :] = self.geom.edge_tangents[e - eoff, p, :]

    def _fill_face_normals(self):

        if self.topo.tdim == 1:
            self.face_normals[:, :] = self.geom.face_normals[:, :]

        if self.topo.tdim == 2:
            eoff = self.topo.kcells_off[1]
            for e in range(self.topo.kcells[1][0], self.topo.kcells[1][1]):
                for p in range(self.geom.num_edge_segments[e - eoff]):
                    self.face_normals[e - eoff, p, :, :] = self.geom.face_normals[e - eoff, p, :]

        if self.topo.tdim == 3:
            foff = self.topo.kcells_off[2]
            eoff = self.topo.kcells_off[1]
            for f in range(self.topo.kcells[2][0], self.topo.kcells[2][1]):
                for eind,e in enumerate(self.topo.EF[f-foff, :self.topo.nEF[f-foff]]):
                    for p in range(self.geom.num_edge_segments[e - eoff]):
                        self.face_normals[f - foff, eind, p, :, :] = self.geom.face_normals[f - foff, eind, p, :]
