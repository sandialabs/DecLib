from DecLib import PETSc
from DecLib import plot1Dmesh, plot2Dmesh, plot3Dmesh
from DecLib import plot1Dmeshpair, plot2Dmeshpair, plot3Dmeshpair
from DecLib import BoxMesh
from DecLib import createTwistedTopology, createTwistedGeometry
from DecLib import gmshDisk, gmshRect, gmshDiskCircHole, gmshRectCircHole
from DecLib import gmshBackwardsStep2D, gmshEllipsoid
from DecLib import gmshCube, gmshCubeSphereHole, gmshBackwardsStep3D

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
                plot1Dmesh(stopo, sgeom, squad, 'uniform1D-straight-' + str(nx) + '-' + lbnd + '-' + rbnd)
                #plot1Dmesh(ttopo, tgeom, 'uniform1D-twisted-' + str(nx) + '-' + lbnd + '-' + rbnd)
                #plot1Dmeshpair(stopo, ttopo, sgeom, tgeom, stmapping, 'uniform1D-pair-' + str(nx) + '-' + lbnd + '-' + rbnd)


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
                plot2Dmesh(stopo, sgeom, squad, 'uniform2D-straight-' + str(nx) + '-' + str(ny) + '-' + bnd)
                plot2Dmesh(stopo, sgeom, squad, 'uniform2D-straight_I-' + str(nx) + '-' + str(ny) + '-' + bnd, celltype='I')
                plot2Dmesh(stopo, sgeom, squad, 'uniform2D-straight_b-' + str(nx) + '-' + str(ny) + '-' + bnd, celltype='b')
                plot2Dmesh(stopo, sgeom, squad, 'uniform2D-straight_B-' + str(nx) + '-' + str(ny) + '-' + bnd, celltype='B')
                #plot2Dmesh(ttopo, tgeom, 'uniform2D-twisted-' + str(nx) + '-' + str(ny) + '-' + bnd)
                #plot2Dmeshpair(stopo, ttopo, sgeom, tgeom, stmapping, 'uniform2D-pair-' + str(nx) + '-' + str(ny) + '-' + bnd)


def test_tri():
    for meshsize in [0.6, 0.8, 1.2]:
        print('gmshDisk', meshsize)
        stopo, sgeom, squad  = gmshDisk([0.0, 0.0], 1.0, meshsize)
        plot2Dmesh(stopo, sgeom, squad, 'gmshDisk-' + 'h=' + str(meshsize))
        plot2Dmesh(stopo, sgeom, squad, 'gmshDisk-I-' + 'h=' + str(meshsize), celltype='I')
        plot2Dmesh(stopo, sgeom, squad, 'gmshDisk-b-' + 'h=' + str(meshsize), celltype='b')
        plot2Dmesh(stopo, sgeom, squad, 'gmshDisk-B-' + 'h=' + str(meshsize), celltype='B')

    for meshsize in [0.3, 0.4, 0.5]:
        print('gmshRect', meshsize)
        stopo, sgeom, squad  = gmshRect([0.0, 0.0], 1.0, 1.0, meshsize)
        plot2Dmesh(stopo, sgeom, squad, 'gmshRect-' + 'h=' + str(meshsize))
        plot2Dmesh(stopo, sgeom, squad, 'gmshRect-I-' + 'h=' + str(meshsize), celltype='I')
        plot2Dmesh(stopo, sgeom, squad, 'gmshRect-b-' + 'h=' + str(meshsize), celltype='b')
        plot2Dmesh(stopo, sgeom, squad, 'gmshRect-B-' + 'h=' + str(meshsize), celltype='B')

    for meshsize in [0.2, 0.3, 0.4]:
        print('gmshRectCircHole', meshsize)
        stopo, sgeom, squad  = gmshRectCircHole([0.0, 0.0], 1.0, 1.0, meshsize)
        plot2Dmesh(stopo, sgeom, squad, 'gmshRectCircHole-' + 'h=' + str(meshsize))
        plot2Dmesh(stopo, sgeom, squad, 'gmshRectCircHole-I-' + 'h=' + str(meshsize), celltype='I')
        plot2Dmesh(stopo, sgeom, squad, 'gmshRectCircHole-b-' + 'h=' + str(meshsize), celltype='b')
        plot2Dmesh(stopo, sgeom, squad, 'gmshRectCircHole-B-' + 'h=' + str(meshsize), celltype='B')

    for meshsize in [0.5, 0.6, 0.7]:
        print('gmshDiskCircHole', meshsize)
        stopo, sgeom, squad  = gmshDiskCircHole([0.0, 0.0], 1.0, meshsize)
        plot2Dmesh(stopo, sgeom, squad, 'gmshDiskCircHole-' + 'h=' + str(meshsize))
        plot2Dmesh(stopo, sgeom, squad, 'gmshDiskCircHole-I-' + 'h=' + str(meshsize), celltype='I')
        plot2Dmesh(stopo, sgeom, squad, 'gmshDiskCircHole-b-' + 'h=' + str(meshsize), celltype='b')
        plot2Dmesh(stopo, sgeom, squad, 'gmshDiskCircHole-B-' + 'h=' + str(meshsize), celltype='B')

    for meshsize in [0.3, 0.4, 0.5]:
        print('gmshBackwardsStep2D', meshsize)
        stopo, sgeom, squad  = gmshBackwardsStep2D([0.0, 0.0], 1.0, 1.0, 0.4, 0.3, meshsize)
        plot2Dmesh(stopo, sgeom, squad, 'gmshBackwardsStep2D-' + 'h=' + str(meshsize))
        plot2Dmesh(stopo, sgeom, squad, 'gmshBackwardsStep2D-I-' + 'h=' + str(meshsize), celltype='I')
        plot2Dmesh(stopo, sgeom, squad, 'gmshBackwardsStep2D-b-' + 'h=' + str(meshsize), celltype='b')
        plot2Dmesh(stopo, sgeom, squad, 'gmshBackwardsStep2D-B-' + 'h=' + str(meshsize), celltype='B')


def test_quad():
    for meshsize in [0.4, 0.6, 0.8]:
        print('gmshDiskQuads', meshsize)
        stopo, sgeom, squad  = gmshDisk([0.0, 0.0], 1.0, meshsize, quads=True)
        plot2Dmesh(stopo, sgeom, squad, 'gmshDiskQuads-' + 'h=' + str(meshsize))
        plot2Dmesh(stopo, sgeom, squad, 'gmshDiskQuads-I-' + 'h=' + str(meshsize), celltype='I')
        plot2Dmesh(stopo, sgeom, squad, 'gmshDiskQuads-b-' + 'h=' + str(meshsize), celltype='b')
        plot2Dmesh(stopo, sgeom, squad, 'gmshDiskQuads-B-' + 'h=' + str(meshsize), celltype='B')

    for meshsize in [0.2, 0.3, 0.4]:
        print('gmshRectQuads', meshsize)
        stopo, sgeom, squad  = gmshRect([0.0, 0.0], 1.0, 1.0, meshsize, quads=True)
        plot2Dmesh(stopo, sgeom, squad, 'gmshRectQuads-' + 'h=' + str(meshsize))
        plot2Dmesh(stopo, sgeom, squad, 'gmshRectQuads-I-' + 'h=' + str(meshsize), celltype='I')
        plot2Dmesh(stopo, sgeom, squad, 'gmshRectQuads-b-' + 'h=' + str(meshsize), celltype='b')
        plot2Dmesh(stopo, sgeom, squad, 'gmshRectQuads-B-' + 'h=' + str(meshsize), celltype='B')

    for meshsize in [0.15,]:
        print('gmshRectCircHoleQuads', meshsize)
        stopo, sgeom, squad  = gmshRectCircHole([0.0, 0.0], 1.0, 1.0, meshsize, quads=True)
        plot2Dmesh(stopo, sgeom, squad, 'gmshRectCircHoleQuads-' + 'h=' + str(meshsize))
        plot2Dmesh(stopo, sgeom, squad, 'gmshRectCircHoleQuads-I-' + 'h=' + str(meshsize), celltype='I')
        plot2Dmesh(stopo, sgeom, squad, 'gmshRectCircHoleQuads-b-' + 'h=' + str(meshsize), celltype='b')
        plot2Dmesh(stopo, sgeom, squad, 'gmshRectCircHoleQuads-B-' + 'h=' + str(meshsize), celltype='B')

    for meshsize in [0.5, 0.6]:
        print('gmshDiskCircHoleQuads', meshsize)
        stopo, sgeom, squad  = gmshDiskCircHole([0.0, 0.0], 1.0, meshsize, quads=True)
        plot2Dmesh(stopo, sgeom, squad, 'gmshDiskCircHoleQuads-' + 'h=' + str(meshsize))
        plot2Dmesh(stopo, sgeom, squad, 'gmshDiskCircHoleQuads-I-' + 'h=' + str(meshsize), celltype='I')
        plot2Dmesh(stopo, sgeom, squad, 'gmshDiskCircHoleQuads-b-' + 'h=' + str(meshsize), celltype='b')
        plot2Dmesh(stopo, sgeom, squad, 'gmshDiskCircHoleQuads-B-' + 'h=' + str(meshsize), celltype='B')

    for meshsize in [0.2, 0.3]:
        print('gmshBackwardsStepQuads2D', meshsize)
        stopo, sgeom, squad  = gmshBackwardsStep2D([0.0, 0.0], 1.0, 1.0, 0.4, 0.3, meshsize, quads=True)
        plot2Dmesh(stopo, sgeom, squad, 'gmshBackwardsStepQuads2D-' + 'h=' + str(meshsize))
        plot2Dmesh(stopo, sgeom, squad, 'gmshBackwardsStepQuads2D-I-' + 'h=' + str(meshsize), celltype='I')
        plot2Dmesh(stopo, sgeom, squad, 'gmshBackwardsStepQuads2D-b-' + 'h=' + str(meshsize), celltype='b')
        plot2Dmesh(stopo, sgeom, squad, 'gmshBackwardsStepQuads2D-B-' + 'h=' + str(meshsize), celltype='B')


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
                    plot3Dmesh(stopo, sgeom, squad, 'uniform3D-straight-' + str(nx) + '-' + str(ny) + '-' + str(nz) + '-' + bnd)
                    plot3Dmesh(stopo, sgeom, squad, 'uniform3D-straight-I-' + str(nx) + '-' + str(ny) + '-' + str(nz) + '-' + bnd, celltype='I')
                    plot3Dmesh(stopo, sgeom, squad, 'uniform3D-straight-b-' + str(nx) + '-' + str(ny) + '-' + str(nz) + '-' + bnd, celltype='b')
                    plot3Dmesh(stopo, sgeom, squad, 'uniform3D-straight-B-' + str(nx) + '-' + str(ny) + '-' + str(nz) + '-' + bnd, celltype='B')

                    #plot3Dmesh(ttopo, tgeom, 'uniform3D-twisted-' + str(nx) + '-' + str(ny) + '-' + str(nz) + '-' + bnd)
                    #plot3Dmeshpair(stopo, ttopo, sgeom, tgeom, stmapping, 'uniform3D-pair-' + str(nx) + '-' + str(ny) + '-' + str(nz) + '-' + bnd)

def test_tet():
    for meshsize in [0.3, 0.4, 0.5]: #0.3, 0.4, 0.5,
        print('gmshCube', meshsize)
        stopo, sgeom, squad  = gmshCube([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], meshsize)
        plot3Dmesh(stopo, sgeom, squad, 'gmshCube-' + 'h=' + str(meshsize))
        plot3Dmesh(stopo, sgeom, squad, 'gmshCube-I-' + 'h=' + str(meshsize), celltype='I')
        plot3Dmesh(stopo, sgeom, squad, 'gmshCube-b-' + 'h=' + str(meshsize), celltype='b')
        plot3Dmesh(stopo, sgeom, squad, 'gmshCube-B-' + 'h=' + str(meshsize), celltype='B')

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

# test_interval()
test_rect()
# test_tri()
# test_quad()
test_box()
test_tet()
#test_hex()
#test_meshio()
