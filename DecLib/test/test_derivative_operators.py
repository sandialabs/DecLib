from DecLib import PETSc
from DecLib import BoxMesh
from DecLib import createTwistedTopology, createTwistedGeometry
from DecLib import ExtDeriv, ExtDeriv_Higher, RealBundle
from DecLib import stdoutindexview, stdoutinfoview
from DecLib import KForm
import numpy as np
from DecLib import gmshDisk, gmshRect, gmshDiskCircHole, gmshRectCircHole
from DecLib import gmshBackwardsStep2D, gmshEllipsoid
from DecLib import gmshCube, gmshCubeSphereHole, gmshBackwardsStep3D


def check_D0c(topo, bundle, D0):
    x0 = KForm(0, topo, bundle, 'x0', create_petsc=True)
    x0.petsc_vec.set(1.3)
    x1 = KForm(1, topo, bundle, 'x1', create_petsc=True)
    D0.petsc_mat.mult(x0.petsc_vec, x1.petsc_vec)
    max, min = x1.petsc_vec.max()[1], x1.petsc_vec.min()[1]
    assert(min == 0.0)
    assert(max == 0.0)

def check_deriv_lower_higher(D, Dhat):
    diff = D.petsc_mat.duplicate(copy=True)
    diff.axpy(-1.0, Dhat.petsc_mat)
    row, ind, vals = diff.getValuesCSR()
    max, min = np.max(vals), np.min(vals)
    assert(max == 0.0)
    assert(min == 0.0)

def check_DD(D0,D1):
    D0.petsc_mat.convert('aij')
    D1.petsc_mat.convert('aij')
    DD = D1.petsc_mat.matMult(D0.petsc_mat)
    #DD.view(viewer=stdoutindexview)
    row, ind, vals = DD.getValuesCSR()
    max, min = np.max(vals), np.min(vals)
    #D0.petsc_mat.view(viewer=stdoutindexview)
    #D1.petsc_mat.view(viewer=stdoutindexview)
    #DD.view(viewer=stdoutindexview)
    assert(min == 0.0)
    assert(max == 0.0)


#ADD IBP TESTS

                #Dtilde = ExtDeriv(0, ttopo, TRbundle)
                #Dtildehat = ExtDeriv_Higher(0, ttopo, TRbundle)

# #CAN ACTUALLY MIX BOUNDARIES AND PERIODIC, SO NEED TO TREAT THAT CASE!

def create_Ds(degrees, topo, bundle):
    Ds = []
    for deg in degrees:
        D = ExtDeriv(deg, topo, bundle)
        D.create_petsc_mat()
        Dhat = ExtDeriv_Higher(deg, topo, bundle)
        Dhat.create_petsc_mat()
        check_deriv_lower_higher(D, Dhat)
        Ds.append(D)
    return Ds

def create_Bs(degrees, stopo, ttopo, SRbundle, TRbundle):
    Bs = []
    for deg in degrees:
        Bs.append((None,None))
    return Bs

def check_IBP(D, Dtilde, B, Btilde):
    return

def check_1d(stopo, ttopo):
    SRbundle = RealBundle(stopo)
    TRbundle = RealBundle(ttopo)

    D0 = create_Ds([0,], stopo, SRbundle)[0]
    Dtilde0 = create_Ds([0,], ttopo, TRbundle)[0]
    B, Btilde = create_Bs([0,], stopo, ttopo, SRbundle, TRbundle)[0]

    check_D0c(stopo, SRbundle, D0)
    check_D0c(ttopo, TRbundle, Dtilde0)

#THIS IS TESTING IBP
#REALLY NEEDS BOUNDARY OPERATORS TO BE DONE CORRECTLY...
    check_IBP(D0, Dtilde0, B, Btilde)
    #x0.petsc_vec.setRandom()
    #D0.petsc_mat.mult(x0.petsc_vec, x1.petsc_vec)
    #x1sum = x1.petsc_vec.sum()
    #print(x1sum)

    #x0.petsc_vec.view(stdoutindexview)
    #x1.petsc_vec.view(stdoutindexview)
    #D.petsc_mat.view(stdoutindexview)



def check_2d(stopo, ttopo):
    SRbundle = RealBundle(stopo)
    TRbundle = RealBundle(ttopo)

    D0, D1 = create_Ds([0,1], stopo, SRbundle)
    Dtilde0, Dtilde1 = create_Ds([0,1], ttopo, TRbundle)
    Bs = create_Bs([0,1], stopo, ttopo, SRbundle, TRbundle)
    B0, Btilde0 = Bs[0]
    B1, Btilde1 = Bs[1]



    check_DD(D0, D1)
    check_DD(Dtilde0, Dtilde1)
    check_D0c(stopo, SRbundle, D0)
    check_D0c(ttopo, TRbundle, Dtilde0)

#THIS IS TESTING IBP
#REALLY NEEDS BOUNDARY OPERATORS TO BE DONE CORRECTLY...
    check_IBP(D0, Dtilde1, B0, Btilde1)
    check_IBP(D1, Dtilde0, B1, Btilde0)

#THERE IS PROBABLY AN IBP FORMULA FOR x0/D0 ALSO?

    #D0tilde = ExtDeriv(0, ttopo, TRbundle)
    #D0tildehat = ExtDeriv_Higher(0, ttopo, TRbundle)
    #D1tilde = ExtDeriv(1, ttopo, TRbundle)
    #D1tildehat = ExtDeriv_Higher(1, ttopo, TRbundle)
    #D0tilde.create_petsc_mat()
    #D0tildehat.create_petsc_mat()
    #D1tilde.create_petsc_mat()
    #D1tildehat.create_petsc_mat()

#ADD IBP TESTS



def check_3d(stopo, ttopo):
    SRbundle = RealBundle(stopo)
    TRbundle = RealBundle(ttopo)

    D0, D1, D2 = create_Ds([0,1,2], stopo, SRbundle)
    Dtilde0, Dtilde1, Dtilde2 = create_Ds([0,1,2], ttopo, TRbundle)
    Bs = create_Bs([0,1,2], stopo, ttopo, SRbundle, TRbundle)
    B0, Btilde0 = Bs[0]
    B1, Btilde1 = Bs[1]
    B2, Btilde2 = Bs[2]

    check_DD(D0, D1)
    check_DD(D1, D2)
    check_DD(Dtilde0, Dtilde1)
    check_DD(Dtilde1, Dtilde2)
    check_D0c(stopo, SRbundle, D0)
    check_D0c(ttopo, TRbundle, Dtilde0)
#THIS IS TESTING IBP
#REALLY NEEDS BOUNDARY OPERATORS TO BE DONE CORRECTLY...
    check_IBP(D0, Dtilde2, B0, Btilde2)
    check_IBP(D2, Dtilde0, B2, Btilde0)
    check_IBP(D1, Dtilde1, B1, Btilde1)
#ARE THERE MORE TO TEST HERE? I DONT THINK SO...

def test_interval():
    for lbnd in ['b', 'B', 'periodic']:  #'periodic'
        for rbnd in ['b', 'B', 'periodic']: #'periodic'
            if lbnd == 'periodic' and (rbnd == 'b' or rbnd == 'B'):
                continue
            if rbnd == 'periodic' and (lbnd == 'b' or lbnd == 'B'):
                continue
            for nx in [4,5]: #5,6?
                print(nx, lbnd, rbnd)
                stopo, _, _ = BoxMesh([nx,], [lbnd, rbnd])
                #ttopo, stmapping = createTwistedTopology(stopo)
                #check_1d(stopo, ttopo)
                check_1d(stopo, stopo)

def test_rect():
    for bnd in ['b', 'B', 'periodic']:  #periodic
        for nx in [4,5]: #5,6
            for ny in [4,5]: #5,6
                print(nx,ny,bnd)
                stopo, _, _ = BoxMesh([nx, ny], bnd)
                #ttopo, stmapping = createTwistedTopology(stopo)
                #check_2d(stopo, ttopo)
                check_2d(stopo, stopo)

def test_tri():
    for meshsize in [0.6, 0.8, 1.2]:
        print('gmshDisk', meshsize)
        stopo, _, _  = gmshDisk([0.0, 0.0], 1.0, meshsize)
        #ttopo, stmapping = createTwistedTopology(stopo)
        #check_2d(stopo, ttopo)
        check_2d(stopo, stopo)

    for meshsize in [0.3, 0.4, 0.5]:
        print('gmshRect', meshsize)
        stopo, _, _  = gmshRect([0.0, 0.0], 1.0, 1.0, meshsize)
        #ttopo, stmapping = createTwistedTopology(stopo)
        #check_2d(stopo, ttopo)
        check_2d(stopo, stopo)

    for meshsize in [0.5, 0.6, 0.7]:
        print('gmshDiskCircHole', meshsize)
        stopo, _, _  = gmshDiskCircHole([0.0, 0.0], 1.0, meshsize)
        #ttopo, stmapping = createTwistedTopology(stopo)
        #check_2d(stopo, ttopo)
        check_2d(stopo, stopo)


    for meshsize in [0.2, 0.3, 0.4]:
        print('gmshRectCircHole', meshsize)
        stopo, _, _  = gmshRectCircHole([0.0, 0.0], 1.0, 1.0, meshsize)
        #ttopo, stmapping = createTwistedTopology(stopo)
        #check_2d(stopo, ttopo)
        check_2d(stopo, stopo)

    for meshsize in [0.3, 0.4, 0.5]:
        print('gmshBackwardsStep2D', meshsize)
        stopo, _, _  = gmshBackwardsStep2D([0.0, 0.0], 1.0, 1.0, 0.4, 0.3, meshsize)
        #ttopo, stmapping = createTwistedTopology(stopo)
        #check_2d(stopo, ttopo)
        check_2d(stopo, stopo)


def test_quad():
    for meshsize in [0.4, 0.6, 0.8]:
        print('gmshDiskQuads', meshsize)
        stopo, _, _  = gmshDisk([0.0, 0.0], 1.0, meshsize, quads=True)
        #ttopo, stmapping = createTwistedTopology(stopo)
        #check_2d(stopo, ttopo)
        check_2d(stopo, stopo)

    for meshsize in [0.2, 0.3, 0.4]:
        print('gmshRectQuads', meshsize)
        stopo, _, _  = gmshRect([0.0, 0.0], 1.0, 1.0, meshsize, quads=True)
        #ttopo, stmapping = createTwistedTopology(stopo)
        #check_2d(stopo, ttopo)
        check_2d(stopo, stopo)

    for meshsize in [0.15,]:
        print('gmshRectCircHoleQuads', meshsize)
        stopo, _, _  = gmshRectCircHole([0.0, 0.0], 1.0, 1.0, meshsize, quads=True)
        #ttopo, stmapping = createTwistedTopology(stopo)
        #check_2d(stopo, ttopo)
        check_2d(stopo, stopo)

    for meshsize in [0.5, 0.6]:
        print('gmshDiskCircHoleQuads', meshsize)
        stopo, _, _  = gmshDiskCircHole([0.0, 0.0], 1.0, meshsize, quads=True)
        #ttopo, stmapping = createTwistedTopology(stopo)
        #check_2d(stopo, ttopo)
        check_2d(stopo, stopo)

    for meshsize in [0.2, 0.3]:
        print('gmshBackwardsStepQuads2D', meshsize)
        stopo, _, _  = gmshBackwardsStep2D([0.0, 0.0], 1.0, 1.0, 0.4, 0.3, meshsize, quads=True)
        #ttopo, stmapping = createTwistedTopology(stopo)
        #check_2d(stopo, ttopo)
        check_2d(stopo, stopo)


# #CAN ACTUALLY MIX BOUNDARIES AND PERIODIC, SO NEED TO TREAT THAT CASE!
def test_box():
    for bnd in ['b', 'B', 'periodic']:  #b B periodic
        for nx in [4,5]:
            for ny in [4,5]:
                for nz in [4,5]:
                    print(nx,ny,nz,bnd)
                    stopo, _, _ = BoxMesh([nx, ny, nz], bnd)
                    #ttopo, stmapping = createTwistedTopology(stopo)
                    #check_3d(stopo, ttopo)
                    check_3d(stopo, stopo)

def test_tet():
    for meshsize in [0.3, 0.4, 0.5]: #,
        print('gmshCube', meshsize)
        stopo, _, _  = gmshCube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], meshsize)
        #ttopo, stmapping = createTwistedTopology(stopo)
        #check_3d(stopo, ttopo)
        check_3d(stopo, stopo)

    # for meshsize in [1.0, 1.5, 2.0]: #0.3, 0.4, 0.5,
    #     print('gmshEllipsoid', meshsize)
    #     stopo, _, _  = gmshEllipsoid([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], meshsize)
    #     check_3d(stopo)

    # for meshsize in [0.3, 0.4, 0.5]: #0.3, 0.4, 0.5,
    #     print('gmshCubeSphereHole', meshsize)
    #     stopo, _, _  = gmshCubeSphereHole([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], meshsize)
    #     check_3d(stopo)

#ADD HOLE AND BACKWARDS STEP 3D

# def test_hex():
#     for meshsize in [0.2, 0.3, 0.4]:
#         print('gmshCubeHexes', meshsize)
#         stopo, _, _  = gmshCube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], meshsize, hexes=True)
#         check_3d(stopo)
# #ADD HOLE AND BACKWARDS STEP 3D

test_interval()
test_rect()
test_tri()
test_quad()
test_tet()
test_box()
#test_hex()
