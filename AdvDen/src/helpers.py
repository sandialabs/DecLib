from DecLib import PrimalIntervalMesh, IntervalGeometry
from DecLib import PrimalQuadMesh, DualQuadGeometry, PrimalDualQuadGeometry
from DecLib import TriangleMesh, meshzooMesh, dmshMesh
from DecLib import createDualTopology, PrimalDualGeometry, DualGeometry
from DecLib import Meshes
from DecLib import plot_mesh1D, pd_mesh_plot_1D, plot_mesh2D, pd_mesh_plot_2D

def createMeshes(params):
    # #create mesh
    # #THIS SHOULD EVENTUALLY BE GENERALIZED TO YAML-DRIVEN CHOICES
    # # EVENTUALLY WANT TO HAVE NON-UNIFORM MESHES HERE
    #create mesh
    #EVENTUALLY ADD A LOT MORE READERS HERE!
    if params['meshtype'] == 'line':
        ptopo = PrimalIntervalMesh(params['nx'], params['xbc'])
    elif params['meshtype'] == 'square':
        ptopo, pgeom = PrimalQuadMesh(params['nx'], params['ny'], params['xbc'], params['ybc'], Lx=params['Lx'], Ly=params['Ly'], xc=params['xc'], yc=params['yc'], quadorder=params['quadorder'])
    elif params['meshtype'] == 'Triangle':
        ptopo, pgeom = TriangleMesh(params['meshfile'], optimize=params['optimize'], method=params['method'], Lx=params['Lx'], Ly=params['Ly'], xc=params['xc'], yc=params['yc'], quadorder=params['quadorder'])
    elif params['meshtype'] == 'meshzoo':
        ptopo, pgeom = meshzooMesh(params['zootype'], params['zoosizes'], optimize=params['optimize'], method=params['method'], Lx=params['Lx'], Ly=params['Ly'], xc=params['xc'], yc=params['yc'], quadorder=params['quadorder'])
    elif params['meshtype'] == 'dmsh':
        ptopo, pgeom =  dmshMesh(params['dmshtype'], params['dmshsize'], optimize=params['optimize'], method=params['method'], Lx=params['Lx'], Ly=params['Ly'], xc=params['xc'], yc=params['yc'], quadorder=params['quadorder'])

    #ADD GMSH- INLCUDING PERIODIC MESHES?

    #ptopo.view(detailed=True)

    dtopo, pdmapping = createDualTopology(ptopo)

    #dtopo.view(detailed=True)

    if params['meshtype'] == 'line':
        pgeom, dgeom, pdgeom = IntervalGeometry(ptopo, dtopo, pdmapping, Lx=params['Lx'], xc=params['xc'], quadorder=params['quadorder'])
    elif params['meshtype'] == 'square' and (params['xbc'] == 'periodic' or params['ybc'] == 'periodic'):
        dgeom = DualQuadGeometry(ptopo, pgeom, dtopo, pdmapping, params['nx'], params['ny'], params['xbc'], params['ybc'], Lx=params['Lx'], Ly=params['Ly'], xc=params['xc'], yc=params['yc'], quadorder=params['quadorder'], uniform_dual=True)
    #ADD SUPPORT FOR PERIODIC MESHES- SHOULD ALLOW ELIMINATION OF SPECIALIZED DUAL GEOM CODE FOR INTERVALS AND PERIODIC SQUARES...
    #THIS WILL REQUIRE SOME SORT OF PERIODIC COORDINATE SYSYEM
    elif params['meshtype'] == 'Triangle' or params['meshtype'] == 'meshzoo' or params['meshtype'] == 'square' or params['meshtype'] == 'dmsh':
        dgeom = DualGeometry(ptopo, pgeom, dtopo, pdmapping, type=params['dualtype'], quadorder=params['quadorder'])


    if params['plotmeshes']:
        if params['meshtype'] == 'line':
            plot_mesh1D(ptopo, pgeom, 'primal')
            plot_mesh1D(dtopo, dgeom, 'dual')
            pd_mesh_plot_1D(ptopo, dtopo, pdmapping, pgeom, dgeom, 'primal-dual')
        else:
            plot_mesh2D(ptopo, pgeom, 'primal')
            plot_mesh2D(dtopo, dgeom, 'dual')
            pd_mesh_plot_2D(ptopo, dtopo, pdmapping, pgeom, dgeom, 'primal-dual')

    if params['meshtype'] == 'line':
        pass
    elif params['meshtype'] == 'square':
        pdgeom = PrimalDualQuadGeometry(ptopo, dtopo, pgeom, dgeom, pdmapping)
    #ADD SUPPORT FOR PERIODIC MESHES- SHOULD ALLOW ELIMINATION OF SPECIALIZED DUAL GEOM CODE FOR INTERVALS AND SQUARES...
    #THIS WILL REQUIRE SOME SORT OF PERIODIC COORDINATE SYSYEM
    else:
    #THIS FAILS FOR DEGENERATE EDGES- REALLY NOT CLEAR WHAT TO DO FOR PDGEOM IN THIS CASE ANYWAYS...
    #CAN LARGELY PUNT ON IT AND JUST USE TOPOLOGICAL WEDGE PRODUCTS...
        pdgeom = PrimalDualGeometry(ptopo, dtopo, pgeom, dgeom, pdmapping)

    #ADD
    #pquad = PrimalQuadrature(params['quadorder'])
    #dquad = DualIntervalQuadrature(params['quadorder'])

    meshes = Meshes(ptopo, dtopo, pgeom, dgeom, pdmapping, pdgeom)
    
    return meshes
