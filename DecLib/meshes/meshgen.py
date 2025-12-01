import pygmsh
import numpy as np
from DecLib.forms.forms import RealBundle, TangentBundle, CotangentBundle
from DecLib import PETSc
from DecLib.meshes.topology import permute_topology, Topology, _orient_1d, _orient_2d, _orient_3d
from DecLib.meshes.geometry import CartesianPlanarMetric, PeriodicPlanarMetric
from DecLib.meshes.quadrature import PlanarQuadrature

class Meshes():
    def __init__(self, stopo, ttopo, sgeom, tgeom, stmapping):
        self.tdim = stopo.tdim
        self.gdim = sgeom.gdim
        self.stopo = stopo
        self.ttopo = ttopo
        self.sgeom = sgeom
        self.tgeom = tgeom
        self.stmapping = stmapping
        self.sRBundle = RealBundle(stopo)
        self.tRBundle = RealBundle(ttopo)
        self.sTBundle = TangentBundle(stopo)
        self.tTBundle = TangentBundle(ttopo)
        self.sCTBundle = CotangentBundle(stopo)
        self.tCTBundle = CotangentBundle(ttopo)


def _GeneralMesh(petscmesh, name, quadorder=3, halowidth=1, labelbnd=True, bndtype='b', periodic= False, periodic_bnds=None, is_simplicial=False):

    faceStart, faceEnd = petscmesh.getDepthStratum(petscmesh.getDimension()-1)

    if labelbnd:
        petscmesh.createLabel('bnd')
        for p in range(*petscmesh.getChart()):
            petscmesh.setLabelValue('bnd', p, 0)

        if bndtype == 'b':
            bndval = 1
        if bndtype == 'B':
            bndval = 2
        for face in range(faceStart, faceEnd):
             ss = petscmesh.getSupportSize(face)
             if ss == 1:
                 petscmesh.clearLabelValue('bnd', face, 0)
                 petscmesh.setLabelValue('bnd', face, bndval)
                 TC = petscmesh.getTransitiveClosure(face, useCone=True)[0]
                 for p in TC[1:]: #first entry of TC is always self
                     petscmesh.clearLabelValue('bnd', p, 0) #transitive closure includes self as first entry
                     petscmesh.setLabelValue('bnd', p, bndval) #transitive closure includes self as first entry

    petscmesh, _ = permute_topology(petscmesh)

    topo = Topology(petscmesh, name = name, halowidth=halowidth, is_simplicial=is_simplicial)

    coords = petscmesh.getCoordinates().getArray()
    coords = np.reshape(coords, (topo.nkcells[0], topo.tdim))

    if periodic:
        cellcoords = petscmesh.getCellCoordinates().getArray()
        cellcoordsection = petscmesh.getCellCoordinateSection()
        geom = PeriodicPlanarMetric(topo, coords, cellcoords, cellcoordsection, periodic_bnds)
    else:
        geom = CartesianPlanarMetric(topo, coords)

    quad = PlanarQuadrature(topo, geom, quadorder)

    if topo.tdim == 1:
        _orient_1d(topo, geom)
    if topo.tdim == 2:
        _orient_2d(topo, geom)
    if topo.tdim == 3:
        _orient_3d(topo, geom)

    return topo, geom, quad



def _CellCoordMesh(cells, coords, name, quadorder=3, halowidth=1, labelbnd=True, bndtype='b', bndlabels=None, periodic= False, periodic_bnds=None, is_simplicial=False):

    cells = np.asarray(cells, dtype=np.int32)
    coords = np.asarray(coords, dtype=np.double)

    petscmesh = PETSc.DMPlex().createFromCellList(coords.shape[1], cells, coords)
    petscmesh.stratify()
#ADD CORRECT INGESTION OF BNDLABELS, IF THEY EXIST!!!!

    return _GeneralMesh(petscmesh, name, quadorder=quadorder, halowidth=halowidth, labelbnd=labelbnd, bndtype=bndtype, periodic=periodic, periodic_bnds=periodic_bnds, is_simplicial=is_simplicial)

def _get_stencil(asdf):
#THIS STUFF IS BROKEN WITH PERMUTED TOPOLOGY
#WHAT WE REALLY NEED TO DO IS CAREFUL GENERATION THROUGH EDGES..
    eStart, eEnd = petscmesh.getDepthStratum(1)
    ne = eEnd-eStart
    topo.xstencil_cells = np.zeros((ne, stencil_width))
    off = (stencil_width-1)//2
    for e in range(eStart,eEnd):
        for j in range(stencil_width):
            topo.xstencil_cells[e-eStart, j] = (e-off+j)%ne #THIS IS PERIODIC
            topo.xstencil_cells[e-eStart, j] = (e-off+j)%ne #BROKEN, BUT SHOULD BE MIRRORING
    #print(topo.xstencil_cells)
#THIS FAILS FOR CERTAIN BCS
#REALLY WE SHOULD BE DOING MIRRORING HERE...
#BUT ALSO THERE IS PERIODIC
#Alternatively, we could do a choice here (default to mirroring, ghost cells is another, periodic for periodic meshes)

#ADD PARTIALLY PERIODIC SUPPORT


    # markervals = petscmesh.getLabelIdIS('marker').indices
    # if 1 in markervals:
    #     for p in petscmesh.getStratumIS('marker',1).indices:
    #         petscmesh.clearLabelValue('bnd', p, 0)
    #         if bnd == 'b':
    #             petscmesh.setLabelValue('bnd', p, 1)
    #         if bnd == 'B':
    #             petscmesh.setLabelValue('bnd', p, 2)

#EVENTUALLY ADD SUPPORT FOR MIXED PERIODICITY HERE!!!

def BoxMesh(nxs, bnd, lowers=[0.0,0.0,0.0], uppers=[1.0, 1.0, 1.0], stencil_width = 3, halowidth=1, quadorder=3):

    periodic = False
    periodic_bnds = None
    labelbnd=True
    if len(nxs) == 1:
        lbnd = bnd[0]
        rbnd = bnd[1]
        bnd = lbnd
        labelbnd=False


    if bnd == 'periodic':
        periodic = True
        periodic_bnds = [lowers,uppers]

    petscmesh = PETSc.DMPlex().createBoxMesh(nxs, lower=lowers, upper=uppers, simplex=False, periodic=periodic)

    if not labelbnd:
        petscmesh.createLabel('bnd')
        for p in range(*petscmesh.getChart()):
            petscmesh.setLabelValue('bnd', p, 0)

        markervals = petscmesh.getLabelIdIS('marker').indices
        if 1 in markervals:
            endpts = petscmesh.getStratumIS('marker',1).indices
            petscmesh.clearLabelValue('bnd', endpts[0], 0)
            petscmesh.clearLabelValue('bnd', endpts[1], 0)
            if lbnd == 'b':
                petscmesh.setLabelValue('bnd', endpts[0], 1)
            if lbnd == 'B':
                petscmesh.setLabelValue('bnd', endpts[0], 2)
            if rbnd == 'b':
                petscmesh.setLabelValue('bnd', endpts[1], 1)
            if rbnd == 'B':
                petscmesh.setLabelValue('bnd', endpts[1], 2)

    #DO SOMETHING TO GENERATE STENCILS HERE!
    #ALSO NEEDS TO RESPECT PERMUTATION OF TOPOLOGY...

    if len(nxs) == 1: name = 'interval-' + str(nxs[0]) + '-' + bnd #+ '-' + rbnd
    if len(nxs) == 2: name = 'quad-' + str(nxs[0]) + '-' + str(nxs[1]) + '-' + bnd
    if len(nxs) == 3: name = 'hex-' + str(nxs[0]) + '-' + str(nxs[1]) + '-' + str(nxs[2]) + '-' + bnd

    return _GeneralMesh(petscmesh, name , halowidth=halowidth, quadorder=quadorder, labelbnd=labelbnd, bndtype=bnd, periodic=periodic, periodic_bnds=periodic_bnds)



def meshioMesh():
    pass


def gmshDisk(center, radius, mesh_size, quadorder=3, halowidth=1, quads=False):

    with pygmsh.geo.Geometry() as geom:
        circ = geom.add_circle(center, radius, mesh_size=mesh_size)
        if quads:
            geom.set_recombined_surfaces([circ.plane_surface])
        mesh = geom.generate_mesh()
#For some reason the circle center is added as a vertex that doesn't connect with anything
        if quads:
            coords, cells = mesh.points[1:,:2], mesh.cells_dict['quad']-1
        else:
            coords, cells = mesh.points[1:,:2], mesh.cells_dict['triangle']-1
        return _CellCoordMesh(cells, coords, 'gmshDisk-h=' + str(mesh_size), quadorder=quadorder, halowidth=halowidth, is_simplicial=(not quads))

def gmshRect(ll, lx, ly, mesh_size, quadorder=3, halowidth=1, quads=False):

    with pygmsh.geo.Geometry() as geom:
        p = geom.add_polygon([ll, [ll[0], ll[1] + ly], [ll[0] + lx, ll[1] + ly], [ll[0] + lx, ll[1]]], mesh_size=mesh_size)
        if quads:
            geom.set_recombined_surfaces([p.surface])
        mesh = geom.generate_mesh()
        if quads:
            coords, cells = mesh.points[:,:2], mesh.cells_dict['quad']
        else:
            coords, cells = mesh.points[:,:2], mesh.cells_dict['triangle']

        return _CellCoordMesh(cells, coords, 'gmshRect-h=' + str(mesh_size), quadorder=quadorder, halowidth=halowidth, is_simplicial=(not quads))


def gmshBackwardsStep2D(ll, lx, ly, step_x, stepheight, mesh_size, quadorder=3, halowidth=1, quads=False):

    with pygmsh.geo.Geometry() as geom:
        vertices = []
        vertices.append([ll[0], ll[1] + ly])
        vertices.append([ll[0] + lx, ll[1] + ly])
        vertices.append([ll[0] + lx, ll[1]])
        vertices.append([ll[0] + step_x, ll[1]])
        vertices.append([ll[0] + step_x, ll[1] + stepheight])
        vertices.append([ll[0], ll[1] + stepheight])
        poly = geom.add_polygon(vertices, mesh_size=mesh_size)
        if quads:
            geom.set_recombined_surfaces([poly.surface])
        mesh = geom.generate_mesh()
        if quads:
            coords, cells = mesh.points[:,:2], mesh.cells_dict['quad']
        else:
            coords, cells = mesh.points[:,:2], mesh.cells_dict['triangle']
        return _CellCoordMesh(cells, coords, 'gmshBackwardsStep2D-h=' + str(mesh_size), quadorder=quadorder, halowidth=halowidth, is_simplicial=(not quads))

def gmshDiskCircHole(center, radius, mesh_size, nholesections=6, quadorder=3, halowidth=1, quads=False):

    with pygmsh.geo.Geometry() as geom:
        hole = geom.add_circle(center, radius/3., mesh_size=mesh_size, num_sections=nholesections, make_surface=False)
        circ = geom.add_circle(center, radius, mesh_size=mesh_size, holes=[hole.curve_loop])
        if quads:
            geom.set_recombined_surfaces([circ.plane_surface])
        mesh = geom.generate_mesh()

#For some reason the circle center is added as a vertex that doesn't connect with anything
#Actually needs to happen twice- once to remove hole vertex, and once to remove circle vertex
#Vertices are ordered as centervertex,hole,center vertex,rest of disk
        if quads:
            coords, cells = mesh.points[1:,:2], mesh.cells_dict['quad']-1
        else:
            coords, cells = mesh.points[1:,:2], mesh.cells_dict['triangle']-1
        coords[nholesections:-1, :] = coords[nholesections+1:, :]
        coords = coords[:-1, :]
        badcoords = np.where(np.greater(cells, nholesections))
        cells[badcoords] = cells[badcoords] - 1
        return _CellCoordMesh(cells, coords, 'gmshDiskCircHole-h=' + str(mesh_size), quadorder=quadorder, halowidth=halowidth, is_simplicial=(not quads))


def gmshRectCircHole(ll, lx, ly, mesh_size, nholesections=6, quadorder=3, halowidth=1, quads=False):

    with pygmsh.geo.Geometry() as geom:
        hole = geom.add_circle([ll[0] + lx/2., ll[1] + ly/2.], min(lx/3.0, ly/3.0), mesh_size=mesh_size, num_sections=nholesections, make_surface=False)
        poly = geom.add_polygon([ll, [ll[0], ll[1] + ly], [ll[0] + lx, ll[1] + ly], [ll[0] + lx, ll[1]]], mesh_size=mesh_size, holes=[hole.curve_loop])
        if quads:
            geom.set_recombined_surfaces([poly.surface])
        mesh = geom.generate_mesh()
#For some reason the circle center is added as a vertex that doesn't connect with anything

        if quads:
            coords, cells = mesh.points[1:,:2], mesh.cells_dict['quad']-1
        else:
            coords, cells = mesh.points[1:,:2], mesh.cells_dict['triangle']-1
        return _CellCoordMesh(cells, coords, 'gmshRectCircHole-h=' + str(mesh_size), quadorder=quadorder, halowidth=halowidth, is_simplicial=(not quads))

def gmshCube(ll, ur, mesh_size, quadorder=3, halowidth=1):

    with pygmsh.geo.Geometry() as geom:
        box = geom.add_box(ll[0], ur[0], ll[1], ur[1], ll[2], ur[2], mesh_size=mesh_size)
        mesh = geom.generate_mesh()

        coords, cells = mesh.points, mesh.cells_dict['tetra']

        return _CellCoordMesh(cells, coords, 'gmshCube-h=' + str(mesh_size), quadorder=quadorder, halowidth=halowidth, is_simplicial=True)

#THIS HAS STRANGE VERTICES, especially at large mesh size
def gmshEllipsoid(center, radii, mesh_size, quadorder=3, halowidth=1):

    with pygmsh.geo.Geometry() as geom:
        hole= geom.add_ellipsoid(center, radii, mesh_size)

        mesh = geom.generate_mesh()
        #strip out the center point
        coords, cells = mesh.points[1:,:], mesh.cells_dict['tetra']-1
        print(mesh.cells_dict)
        return _CellCoordMesh(cells, coords, 'gmshEllipsoid-h=' + str(mesh_size), quadorder=quadorder, halowidth=halowidth, is_simplicial=True)


def gmshCubeSphereHole(ll, ur, mesh_size, quadorder=3, halowidth=1):

    lxs = np.array(ur) - np.array(ll)
    radius = min(lxs[0]/3.0, lxs[1]/3.0, lxs[2]/3/0)
#HOLE STUFF IS A LITTLE BROKEN!
    with pygmsh.geo.Geometry() as geom:
        hole= geom.add_ellipsoid([ll[0] + lxs[0]/2., ll[1] + lxs[1]/2., ll[2] + lxs[2]/2.], [lxs[0]/3.0, lxs[1]/3.0, lxs[2]/3.0], mesh_size)
        box = geom.add_box(ll[0], ur[0], ll[1], ur[1], ll[2], ur[2], mesh_size=mesh_size, holes=[hole.surface])

        mesh = geom.generate_mesh()
        coords, cells = mesh.points, mesh.cells_dict['tetra']
        return _CellCoordMesh(cells, coords, 'gmshCubeCircHole-h=' + str(mesh_size), quadorder=quadorder, halowidth=halowidth, is_simplicial=True)


#THIS IS UNTESTED AND BROKEN
def gmshBackwardsStep3D(ll, lx, ly, lz, step_x, stepheight, mesh_size, quadorder=3, halowidth=1, hexes=False):

    with pygmsh.geo.Geometry() as geom:
        vertices = []
        vertices.append([ll[0], ll[1] + ly])
        vertices.append([ll[0] + lx, ll[1] + ly])
        vertices.append([ll[0] + lx, ll[1]])
        vertices.append([ll[0] + step_x, ll[1]])
        vertices.append([ll[0] + step_x, ll[1] + stepheight])
        vertices.append([ll[0], ll[1] + stepheight])
        poly = geom.add_polygon(vertices, mesh_size=mesh_size)
        mesh = geom.generate_mesh()
        coords, cells = mesh.points, mesh.cells_dict['tetra']
        return _CellCoordMesh(cells, coords, 'gmshBackwardsStep3D-h=' + str(mesh_size), quadorder=quadorder, halowidth=halowidth, is_simplicial=(not hexes))
