from DecLib import PETSc
from DecLib import plot1Dmesh, plot2Dmesh, plot3Dmesh
from DecLib import plot1Dmeshpair, plot2Dmeshpair, plot3Dmeshpair
from DecLib import BoxMesh
from DecLib import createTwistedTopology, createTwistedGeometry
from DecLib import gmshDisk, gmshRect, gmshDiskCircHole, gmshRectCircHole
from DecLib import gmshBackwardsStep2D, gmshEllipsoid
from DecLib import gmshCube, gmshCubeSphereHole, gmshBackwardsStep3D
from DecLib import RealBundle, TangentBundle, CotangentBundle, KForm

def func1D(t, x):
    return x*x + 2.3

def func2D(t, x, y):
    return x*x + 2.3 + y*y*y

def func3D(t, x, y, z):
    return x*x + 2.3 + y*y*y + z

def vecfunc2D(t, x, y):
    return [x*x + 2.3 + y*y*y, x + y*y + 2.1]

def vecfunc3D(t, x, y, z):
    return [x*x + 2.3 + y*y*y + z, x*z + y*y + 2.1, z*y + x*y]

def test_1d(topo, geom, quad):
    Rbundle = RealBundle(topo)
    Tbundle = TangentBundle(topo)
    CTbundle = CotangentBundle(topo)
    x0 = KForm(0, topo, Rbundle, 'x0', create_petsc=True)
    xT0 = KForm(0, topo, Tbundle, 'xT0', create_petsc=True)
    xCT0 = KForm(0, topo, CTbundle, 'xCT0', create_petsc=True)
    x1 = KForm(1, topo, Rbundle, 'x1', create_petsc=True)
    xT1 = KForm(1, topo, Tbundle, 'xT1', create_petsc=True)
    xCT1 = KForm(1, topo, CTbundle, 'xCT1', create_petsc=True)

    x0.set_petsc_vec(quad, [func1D,], formtype='all')
    x0.set_petsc_vec(quad, [func1D,], formtype='I')
    x0.set_petsc_vec(quad, [func1D,], formtype='b')
    x0.set_petsc_vec(quad, [func1D,], formtype='B')
    xT0.set_petsc_vec(quad, [func1D,], formtype='all')
    xT0.set_petsc_vec(quad, [func1D,], formtype='I')
    xT0.set_petsc_vec(quad, [func1D,], formtype='b')
    xT0.set_petsc_vec(quad, [func1D,], formtype='B')
    xCT0.set_petsc_vec(quad, [func1D,], formtype='all')
    xCT0.set_petsc_vec(quad, [func1D,], formtype='I')
    xCT0.set_petsc_vec(quad, [func1D,], formtype='b')
    xCT0.set_petsc_vec(quad, [func1D,], formtype='B')

    x1.set_petsc_vec(quad, [func1D,], formtype='all')
    x1.set_petsc_vec(quad, [func1D,], formtype='I')
    x1.set_petsc_vec(quad, [func1D,], formtype='b')
    x1.set_petsc_vec(quad, [func1D,], formtype='B')
    xT1.set_petsc_vec(quad, [func1D,], formtype='all')
    xT1.set_petsc_vec(quad, [func1D,], formtype='I')
    xT1.set_petsc_vec(quad, [func1D,], formtype='b')
    xT1.set_petsc_vec(quad, [func1D,], formtype='B')
    xCT1.set_petsc_vec(quad, [func1D,], formtype='all')
    xCT1.set_petsc_vec(quad, [func1D,], formtype='I')
    xCT1.set_petsc_vec(quad, [func1D,], formtype='b')
    xCT1.set_petsc_vec(quad, [func1D,], formtype='B')

def test_2d(topo, geom, quad):
    Rbundle = RealBundle(topo)
    Tbundle = TangentBundle(topo)
    CTbundle = CotangentBundle(topo)
    x0 = KForm(0, topo, Rbundle, 'x0', create_petsc=True)
    xT0 = KForm(0, topo, Tbundle, 'xT0', create_petsc=True)
    xCT0 = KForm(0, topo, CTbundle, 'xCT0', create_petsc=True)
    x1 = KForm(1, topo, Rbundle, 'x1', create_petsc=True, one_form_type='tangent')
    xT1 = KForm(1, topo, Tbundle, 'xT1', create_petsc=True, one_form_type='tangent')
    xCT1 = KForm(1, topo, CTbundle, 'xCT1', create_petsc=True, one_form_type='tangent')
    xnm1 = KForm(1, topo, Rbundle, 'xnm1', create_petsc=True, one_form_type='normal')
    xTnm1 = KForm(1, topo, Tbundle, 'xTnm1', create_petsc=True, one_form_type='normal')
    xCTnm1 = KForm(1, topo, CTbundle, 'xCTnm1', create_petsc=True, one_form_type='normal')
    x2 = KForm(2, topo, Rbundle, 'x2', create_petsc=True)
    xT2 = KForm(2, topo, Tbundle, 'xT2', create_petsc=True)
    xCT2 = KForm(2, topo, CTbundle, 'xCT2', create_petsc=True)

    x0.set_petsc_vec(quad, [func2D,], formtype='all')
    x0.set_petsc_vec(quad, [func2D,], formtype='I')
    x0.set_petsc_vec(quad, [func2D,], formtype='b')
    x0.set_petsc_vec(quad, [func2D,], formtype='B')
    xT0.set_petsc_vec(quad, [vecfunc2D,], formtype='all')
    xT0.set_petsc_vec(quad, [vecfunc2D,], formtype='I')
    xT0.set_petsc_vec(quad, [vecfunc2D,], formtype='b')
    xT0.set_petsc_vec(quad, [vecfunc2D,], formtype='B')
    xCT0.set_petsc_vec(quad, [vecfunc2D,], formtype='all')
    xCT0.set_petsc_vec(quad, [vecfunc2D,], formtype='I')
    xCT0.set_petsc_vec(quad, [vecfunc2D,], formtype='b')
    xCT0.set_petsc_vec(quad, [vecfunc2D,], formtype='B')

    x1.set_petsc_vec(quad, [vecfunc2D,], formtype='all')
    x1.set_petsc_vec(quad, [vecfunc2D,], formtype='I')
    x1.set_petsc_vec(quad, [vecfunc2D,], formtype='b')
    x1.set_petsc_vec(quad, [vecfunc2D,], formtype='B')
    xnm1.set_petsc_vec(quad, [vecfunc2D,], formtype='all')
    xnm1.set_petsc_vec(quad, [vecfunc2D,], formtype='I')
    xnm1.set_petsc_vec(quad, [vecfunc2D,], formtype='b')
    xnm1.set_petsc_vec(quad, [vecfunc2D,], formtype='B')
#ADD X1CT, etc.!!!

    x2.set_petsc_vec(quad, [func2D,], formtype='all')
    x2.set_petsc_vec(quad, [func2D,], formtype='I')
    x2.set_petsc_vec(quad, [func2D,], formtype='b')
    x2.set_petsc_vec(quad, [func2D,], formtype='B')
    xT2.set_petsc_vec(quad, [vecfunc2D,], formtype='all')
    xT2.set_petsc_vec(quad, [vecfunc2D,], formtype='I')
    xT2.set_petsc_vec(quad, [vecfunc2D,], formtype='b')
    xT2.set_petsc_vec(quad, [vecfunc2D,], formtype='B')
    xCT2.set_petsc_vec(quad, [vecfunc2D,], formtype='all')
    xCT2.set_petsc_vec(quad, [vecfunc2D,], formtype='I')
    xCT2.set_petsc_vec(quad, [vecfunc2D,], formtype='b')
    xCT2.set_petsc_vec(quad, [vecfunc2D,], formtype='B')

def test_3d(topo, geom, quad):
    Rbundle = RealBundle(topo)
    Tbundle = TangentBundle(topo)
    CTbundle = CotangentBundle(topo)
    x0 = KForm(0, topo, Rbundle, 'x0', create_petsc=True)
    xT0 = KForm(0, topo, Tbundle, 'xT0', create_petsc=True)
    xCT0 = KForm(0, topo, CTbundle, 'xCT0', create_petsc=True)
    x1 = KForm(1, topo, Rbundle, 'x1', create_petsc=True)
    xT1 = KForm(1, topo, Tbundle, 'xT1', create_petsc=True)
    xCT1 = KForm(1, topo, CTbundle, 'xCT1', create_petsc=True)
    x2 = KForm(2, topo, Rbundle, 'x2', create_petsc=True)
    xT2 = KForm(2, topo, Tbundle, 'xT2', create_petsc=True)
    xCT2 = KForm(2, topo, CTbundle, 'xCT2', create_petsc=True)
    x3 = KForm(3, topo, Rbundle, 'x3', create_petsc=True)
    xT3 = KForm(3, topo, Tbundle, 'xT3', create_petsc=True)
    xCT3 = KForm(3, topo, CTbundle, 'xCT3', create_petsc=True)

    x0.set_petsc_vec(quad, [func3D,], formtype='all')
    x0.set_petsc_vec(quad, [func3D,], formtype='I')
    x0.set_petsc_vec(quad, [func3D,], formtype='b')
    x0.set_petsc_vec(quad, [func3D,], formtype='B')
    xT0.set_petsc_vec(quad, [vecfunc3D,], formtype='all')
    xT0.set_petsc_vec(quad, [vecfunc3D,], formtype='I')
    xT0.set_petsc_vec(quad, [vecfunc3D,], formtype='b')
    xT0.set_petsc_vec(quad, [vecfunc3D,], formtype='B')
    xCT0.set_petsc_vec(quad, [vecfunc3D,], formtype='all')
    xCT0.set_petsc_vec(quad, [vecfunc3D,], formtype='I')
    xCT0.set_petsc_vec(quad, [vecfunc3D,], formtype='b')
    xCT0.set_petsc_vec(quad, [vecfunc3D,], formtype='B')

    x1.set_petsc_vec(quad, [vecfunc3D,], formtype='all')
    x1.set_petsc_vec(quad, [vecfunc3D,], formtype='I')
    x1.set_petsc_vec(quad, [vecfunc3D,], formtype='b')
    x1.set_petsc_vec(quad, [vecfunc3D,], formtype='B')
    x2.set_petsc_vec(quad, [vecfunc3D,], formtype='all')
    x2.set_petsc_vec(quad, [vecfunc3D,], formtype='I')
    x2.set_petsc_vec(quad, [vecfunc3D,], formtype='b')
    x2.set_petsc_vec(quad, [vecfunc3D,], formtype='B')

    x3.set_petsc_vec(quad, [func3D,], formtype='all')
    x3.set_petsc_vec(quad, [func3D,], formtype='I')
    x3.set_petsc_vec(quad, [func3D,], formtype='b')
    x3.set_petsc_vec(quad, [func3D,], formtype='B')
    xT3.set_petsc_vec(quad, [vecfunc3D,], formtype='all')
    xT3.set_petsc_vec(quad, [vecfunc3D,], formtype='I')
    xT3.set_petsc_vec(quad, [vecfunc3D,], formtype='b')
    xT3.set_petsc_vec(quad, [vecfunc3D,], formtype='B')
    xCT3.set_petsc_vec(quad, [vecfunc3D,], formtype='all')
    xCT3.set_petsc_vec(quad, [vecfunc3D,], formtype='I')
    xCT3.set_petsc_vec(quad, [vecfunc3D,], formtype='b')
    xCT3.set_petsc_vec(quad, [vecfunc3D,], formtype='B')

def test_interval():
    for lbnd in ['b', 'B', 'periodic']:  #'periodic'
        for rbnd in ['b', 'B', 'periodic']: #'periodic'
            if lbnd == 'periodic' and (rbnd == 'b' or rbnd == 'B'):
                continue
            if rbnd == 'periodic' and (lbnd == 'b' or lbnd == 'B'):
                continue
            for nx in [4,5]: #5,6?
                print(nx, lbnd, rbnd)
                stopo, sgeom, squad = BoxMesh([nx,], [lbnd, rbnd], lowers=[-0.5,0.,0.], uppers=[2.3,0.0,0.0], quadorder=4)
                #ttopo, stmapping = createDualTopology(stopo)
                #tgeom, tquad = createDualGeometry(stopo, ttopo, sgeom, type='centroid')
#NEED A WAY TO MAKE EITHER PRIMAL OR DUAL GEOMETRY UNIFORM HERE..
                test_1d(stopo, sgeom, squad)

# #CAN ACTUALLY MIX BOUNDARIES AND PERIODIC, SO NEED TO TREAT THAT CASE!
def test_rect():
    for bnd in ['periodic', 'b', 'B', ]:  #
        for nx in [4,5]: #5,6
            for ny in [4,5]: #5,6
                print(nx, ny, bnd)
                stopo, sgeom, squad = BoxMesh([nx, ny], bnd, lowers=[0.,0.,0.], uppers=[1.0,1.0,1.0], quadorder=4)
                #ttopo, stmapping = createDualTopology(stopo)
                #tgeom = createDualGeometry(stopo, ttopo, sgeom, type='centroid')
                #NEED A WAY TO MAKE EITHER PRIMAL OR DUAL GEOMETRY UNIFORM HERE..
                test_2d(stopo, sgeom, squad)


def test_tri():
    for meshsize in [0.6, 0.8, 1.2]:
        print('gmshDisk', meshsize)
        stopo, sgeom, squad  = gmshDisk([0.0, 0.0], 1.0, meshsize)
        test_2d(stopo, sgeom, squad)

    for meshsize in [0.3, 0.4, 0.5]:
        print('gmshRect', meshsize)
        stopo, sgeom, squad  = gmshRect([0.0, 0.0], 1.0, 1.0, meshsize)
        test_2d(stopo, sgeom, squad)


    for meshsize in [0.2, 0.3, 0.4]:
        print('gmshRectCircHole', meshsize)
        stopo, sgeom, squad  = gmshRectCircHole([0.0, 0.0], 1.0, 1.0, meshsize)
        test_2d(stopo, sgeom, squad)


    for meshsize in [0.5, 0.6, 0.7]:
        print('gmshDiskCircHole', meshsize)
        stopo, sgeom, squad  = gmshDiskCircHole([0.0, 0.0], 1.0, meshsize)
        test_2d(stopo, sgeom, squad)


    for meshsize in [0.3, 0.4, 0.5]:
        print('gmshBackwardsStep2D', meshsize)
        stopo, sgeom, squad  = gmshBackwardsStep2D([0.0, 0.0], 1.0, 1.0, 0.4, 0.3, meshsize)
        test_2d(stopo, sgeom, squad)



def test_quad():
    for meshsize in [0.4, 0.6, 0.8]:
        print('gmshDiskQuads', meshsize)
        stopo, sgeom, squad  = gmshDisk([0.0, 0.0], 1.0, meshsize, quads=True)
        test_2d(stopo, sgeom, squad)

    for meshsize in [0.2, 0.3, 0.4]:
        print('gmshRectQuads', meshsize)
        stopo, sgeom, squad  = gmshRect([0.0, 0.0], 1.0, 1.0, meshsize, quads=True)
        test_2d(stopo, sgeom, squad)

    for meshsize in [0.15,]:
        print('gmshRectCircHoleQuads', meshsize)
        stopo, sgeom, squad  = gmshRectCircHole([0.0, 0.0], 1.0, 1.0, meshsize, quads=True)
        test_2d(stopo, sgeom, squad)

    for meshsize in [0.5, 0.6]:
        print('gmshDiskCircHoleQuads', meshsize)
        stopo, sgeom, squad  = gmshDiskCircHole([0.0, 0.0], 1.0, meshsize, quads=True)
        test_2d(stopo, sgeom, squad)

    for meshsize in [0.2, 0.3]:
        print('gmshBackwardsStepQuads2D', meshsize)
        stopo, sgeom, squad  = gmshBackwardsStep2D([0.0, 0.0], 1.0, 1.0, 0.4, 0.3, meshsize, quads=True)
        test_2d(stopo, sgeom, squad)


# #CAN ACTUALLY MIX BOUNDARIES AND PERIODIC, SO NEED TO TREAT THAT CASE!
def test_box():
    for bnd in ['b', 'B', 'periodic']:  #b B periodic
        for nx in [4,]:
            for ny in [4,]:
                for nz in [4,]:
                    print(nx, ny, nz, bnd)
                    stopo, sgeom, squad = BoxMesh([nx, ny, nz], bnd, lowers=[0.,0.,0.], uppers=[1.0,1.0,1.0])
                    #ttopo, stmapping = createDualTopology(stopo)
                    #tgeom = createDualGeometry(stopo, ttopo, sgeom, type='centroid')
                    #NEED A WAY TO MAKE EITHER PRIMAL OR DUAL GEOMETRY UNIFORM HERE..
                    test_3d(stopo, sgeom, squad)

def test_tet():
    for meshsize in [0.3, 0.4, 0.5]: #0.3, 0.4, 0.5,
        print('gmshCube', meshsize)
        stopo, sgeom, squad  = gmshCube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], meshsize)
        test_3d(stopo, sgeom, squad)

    # for meshsize in [1.0, 1.5, 2.0]: #0.3, 0.4, 0.5,
    #     print('gmshEllipsoid', meshsize)
    #     stopo, sgeom, squad  = gmshEllipsoid([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], meshsize)
    #     print(stopo.nEV)
    #     print(stopo.nFV)
    #     print(stopo.nCV)
    #     plot3Dmesh(stopo, sgeom, squad, 'gmshEllipsoid-' + 'h=' + str(meshsize))
    #     plot3Dmesh(stopo, sgeom, squad, 'gmshEllipsoid-I-' + 'h=' + str(meshsize), celltype='I')
    #     plot3Dmesh(stopo, sgeom, squad, 'gmshEllipsoid-b-' + 'h=' + str(meshsize), celltype='b')
    #     plot3Dmesh(stopo, sgeom, squad, 'gmshEllipsoid-B-' + 'h=' + str(meshsize), celltype='B')
    #
    # for meshsize in [0.3, 0.4, 0.5]: #0.3, 0.4, 0.5,
    #     print('gmshCubeSphereHole', meshsize)
    #     stopo, sgeom, squad  = gmshCubeSphereHole([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], meshsize)
    #     plot3Dmesh(stopo, sgeom, squad, 'gmshCubeSphereHole-' + 'h=' + str(meshsize))
    #     plot3Dmesh(stopo, sgeom, squad, 'gmshCubeSphereHole-I-' + 'h=' + str(meshsize), celltype='I')
    #     plot3Dmesh(stopo, sgeom, squad, 'gmshCubeSphereHole-b-' + 'h=' + str(meshsize), celltype='b')
    #     plot3Dmesh(stopo, sgeom, squad, 'gmshCubeSphereHole-B-' + 'h=' + str(meshsize), celltype='B')

#ADD HOLE AND BACKWARDS STEP 3D

# def test_hex():
#     for meshsize in [0.2, 0.3, 0.4]:
#         print('gmshCubeHexes', meshsize)
#         stopo, sgeom, squad  = gmshCube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], meshsize, hexes=True)
#         plot3Dmesh(stopo, sgeom, squad, 'gmshCubeHexes-' + 'h=' + str(meshsize))
#         plot3Dmesh(stopo, sgeom, squad, 'gmshCubeHexes-I-' + 'h=' + str(meshsize), celltype='I')
#         plot3Dmesh(stopo, sgeom, squad, 'gmshCubeHexes-b-' + 'h=' + str(meshsize), celltype='b')
#         plot3Dmesh(stopo, sgeom, squad, 'gmshCubeHexes-B-' + 'h=' + str(meshsize), celltype='B')
# #ADD HOLE AND BACKWARDS STEP 3D

# def test_meshio():
#   pass

#ADD PERIODIC GMSH STUFF

test_interval()
test_rect()
test_tri()
test_quad()
test_box()
test_tet()
#test_hex()
#test_meshio()
