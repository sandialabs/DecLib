from DecLib import PETSc
from DecLib import plot1Dmesh, plot2Dmesh, plot3Dmesh
from DecLib import TriangleMesh, meshioMesh
from DecLib import IntervalMesh, UniformIntervalGeometry
from DecLib import QuadMesh, UniformQuadGeometry
from DecLib import HexMesh, UniformHexGeometry
from DecLib import StructuredIntervalMesh, StructuredQuadMesh, StructuredHexMesh

#ADD PARALLEL STUFF AS WELL!!
#follow the model of master process creating the mesh, then distributing it
#I think this should work...

#ADD PERIODIC EVENTUALLY
def test_interval():
    for lbnd in ['b', 'B']:  #b B periodic
        for rbnd in ['b', 'B']:
            for nx in [4,5,6]:
                stopo = IntervalMesh(nx, lbnd, rbnd)
                sgeom = UniformIntervalGeometry(stopo, Lx=1.0, xc=0.5)
                #NEED A WAY TO MAKE EITHER PRIMAL OR DUAL GEOMETRY UNIFORM HERE..
                plot1Dmesh(stopo, sgeom, 'uniform1D-' + str(nx) + '-' + lbnd + '-' + rbnd)
                #WHAT ASSERTS DO WE NEED HERE?
                #mesh sizes, orientation stuff
                #Ib <-> Ib, b <-> B, B <-> b for dual topologies
                #ANYTHING ELSE?
                if lbd == 'b' and rbd == 'b':
                    assert()
                if lbd == 'b' and rbd == 'B':
                    assert()
                if lbd == 'B' and rbd == 'b':
                    assert()
                if lbd == 'B' and rbd == 'B':
                    assert()

#ADD PERIODIC EVENTUALLY
#CAN ACTUALLY MIX BOUNDARIES AND PERIODIC, SO NEED TO TREAT THAT CASE!
def test_quad():
    for bnd in ['b', 'B']:  #b B periodic
        for nx in [4,5,6]:
            for ny in [4,5,6]:
                stopo = QuadMesh(nx, ny, bnd)
                sgeom = UniformQuadGeometry(stopo, Lx=1.0, Ly=1.0, xc=0.5, yc=0.5)
                #NEED A WAY TO MAKE EITHER PRIMAL OR DUAL GEOMETRY UNIFORM HERE..
                plot2Dmesh(stopo, sgeom, 'uniform2D-' + str(nx) + '-' + str(ny) + '-' + bnd)
                if bnd == 'b':
                    assert()
                if bnd == 'B':
                    assert()

#ADD PERIODIC EVENTUALLY
#CAN ACTUALLY MIX BOUNDARIES AND PERIODIC, SO NEED TO TREAT THAT CASE!
def test_hex():
    for bnd in ['b', 'B']:  #b B periodic
        for nx in [4,5,6]:
            for ny in [4,5,6]:
                for nz in [4,5,6]:
                    stopo = HexMesh(nx, ny, nz, bnd)
                    sgeom = UniformHexGeometry(stopo, Lx=1.0, Ly=1.0, Lz=1.0, xc=0.5, yc=0.5, zc=0.5)
                    #NEED A WAY TO MAKE EITHER PRIMAL OR DUAL GEOMETRY UNIFORM HERE..
                    plot3Dmesh(stopo, sgeom, 'uniform3D-' + str(nx) + '-' + str(ny) + '-' + str(nz) + '-' + bnd)
                    if bnd == 'b':
                        assert()
                    if bnd == 'B':
                        assert()


#ADD PERIODIC EVENTUALLY
#CAN ACTUALLY MIX BOUNDARIES AND PERIODIC, SO NEED TO TREAT THAT CASE!
def test_structured():
    for bnd in ['b', 'B']: #periodic
        for nx in [4,5,6]:
            stopo, sgeom = StructuredIntervalMesh(nx, bnd)
            #NEED A WAY TO MAKE EITHER PRIMAL OR DUAL GEOMETRY UNIFORM HERE..
            plot1Dmesh(stopo, sgeom, 'uniform1D-' + str(nx) + '-' + bnd)
            if bnd == 'b':
                assert()
            if bnd == 'B':
                assert()
            for ny in [4,5,6]:
                stopo, sgeom = StructuredQuadMesh(nx, ny, bnd)
            #NEED A WAY TO MAKE EITHER PRIMAL OR DUAL GEOMETRY UNIFORM HERE..
                plot2Dmesh(stopo, sgeom, 'uniform2D-' + str(nx) + '-' + str(ny) + '-' + bnd)
                if bnd == 'b':
                    assert()
                if bnd == 'B':
                    assert()
                for nz in [4,5,6]:
                    stopo, sgeom = StructuredHexMesh(nx, ny, nz, bnd)
                    #NEED A WAY TO MAKE EITHER PRIMAL OR DUAL GEOMETRY UNIFORM HERE..
                    plot3Dmesh(stopo, sgeom, 'uniform3D-' + str(nx) + '-' + str(ny) + '-' + str(nz) + '-' + bnd)
                    if bnd == 'b':
                        assert()
                    if bnd == 'B':
                        assert()

def test_Triangle():
    for meshfilename in ['Triangle1', 'Triangle3', 'Triangle4', 'Triangle4']:
        stopo, sgeom = TriangleMesh(meshfilename)
        plot2Dmesh(stopo, sgeom, meshfilename)
        assert()

def test_gmsh2D():
    for meshfilename in ['Gmsh2D-1', 'Gmsh2D-2', 'Gmsh2D-3', 'Gmsh2D-4']:
        stopo, sgeom = meshioMesh(meshfilename)
        plot2Dmesh(stopo, sgeom, meshfilename)
        assert()

def test_gmsh3D():
    for meshfilename in ['Gmsh3D-1', 'Gmsh3D-2', 'Gmsh3D-3', 'Gmsh3D-4']:
        stopo, sgeom = meshioMesh(meshfilename)
        plot3Dmesh(stopo, sgeom, meshfilename)
        assert()
