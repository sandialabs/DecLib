from DecLib import PrimalIntervalMesh, PrimalIntervalQuadrature, DualIntervalGeometry, DualIntervalQuadrature, PrimalDualIntervalGeometry
from DecLib import PrimalQuadMesh, DualQuadGeometry, PrimalDualQuadGeometry
from DecLib import TriangleMesh, meshzooMesh, dmshMesh
from DecLib import createDualTopology, PrimalDualGeometry, DualGeometry
from DecLib import Meshes
from DecLib import plot_mesh1D, pd_mesh_plot_1D, plot_mesh2D, pd_mesh_plot_2D
from DecLib import PETSc
from DecLib import ExtDeriv, VoronoiStar, VoronoiStarDual, Codiff, HodgeLaplacian, TopoPairing, InnerProduct

import yaml
import sys

#cfgfilename = sys.argv[1]
#cfgfile =  open(cfgfilename + '.yaml", mode="r")
#params = yaml.safe_load(cfgfile)

# #READ ALL OF THIS FROM A YAML FILE
params = {}

params['meshtype'] = 'dmsh' #'Triangle' 'square' 'meshzoo' 'line' 'dmsh'

params['xbc'] = 'periodic' #'none' 'periodic'
params['ybc'] = 'periodic' #'none' 'periodic'
params['nx'] = 3 #100
params['ny'] = 3 #100

params['meshfile'] = 'meshes/box-triangles/high4/square.1'

params['plotmeshes'] = True

params['dmshtype'] = 'square' #square circ
params['dmshsize'] = 0.12

params['optimize'] = True
params['method'] = 'odt-fixed-point'

#THIS GIVES A MESH WITH NON WELL CENTERED CELLS! Good for testing various Hodge stars...
params['zootype'] = 'rectangle-tri' #disk rectangle-tri
params['zoosizes'] = [5,5]

    # "lloyd": cvt.lloyd,
    # "cvt-diaognal": cvt.lloyd,
    # "cvt-block-diagonal": cvt.block_diagonal,
    # "cvt-full": cvt.full,
    # "cpt-linear-solve": cpt.linear_solve,
    # "cpt-fixed-point": cpt.fixed_point,
    # "cpt-quasi-newton": cpt.quasi_newton,
    # "laplace": laplace,
    # "odt-fixed-point": odt.fixed_point,

params['dualtype'] = 'centroid' #'circumcenter' 'centroid'

params['hodgetype'] = 'voronoi' #'voronoi' 'signedvoronoi' 'barycentric'
#WHAT OTHERS?

params['quadorder'] = 3


# #create mesh
# #THIS SHOULD EVENTUALLY BE GENERALIZED TO YAML-DRIVEN CHOICES
# # EVENTUALLY WANT TO HAVE NON-UNIFORM MESHES HERE
#create mesh
#EVENTUALLY ADD A LOT MORE READERS HERE!
if params['meshtype'] == 'line':
    ptopo, pgeom = PrimalIntervalMesh(params['nx'], params['xbc'])
elif params['meshtype'] == 'square':
    ptopo, pgeom = PrimalQuadMesh(params['nx'], params['ny'], params['xbc'], params['ybc'], quadorder=params['quadorder'])
elif params['meshtype'] == 'Triangle':
    ptopo, pgeom = TriangleMesh(params['meshfile'], optimize=params['optimize'], method=params['method'], quadorder=params['quadorder'])
elif params['meshtype'] == 'meshzoo':
    ptopo, pgeom = meshzooMesh(params['zootype'], params['zoosizes'], optimize=params['optimize'], method=params['method'], quadorder=params['quadorder'])
elif params['meshtype'] == 'dmsh':
    ptopo, pgeom =  dmshMesh(params['dmshtype'], params['dmshsize'], optimize=params['optimize'], method=params['method'], quadorder=params['quadorder'])

#ADD GMSH- INLCUDING PERIODIC MESHES?

#ptopo.view(detailed=True)

dtopo, pdmapping = createDualTopology(ptopo)

#dtopo.view(detailed=True)

if params['meshtype'] == 'line':
    dgeom = DualIntervalGeometry(ptopo, pgeom, dtopo, pdmapping, params['nx'], params['xbc'], Lx=params['Lx'], xc=params['xc'], quadorder=params['quadorder'])
elif params['meshtype'] == 'square' and (params['xbc'] == 'periodic' or params['ybc'] == 'periodic'):
    dgeom = DualQuadGeometry(ptopo, pgeom, dtopo, pdmapping, params['nx'], params['ny'], params['xbc'], params['ybc'], Lx=params['Lx'], Ly=params['Ly'], xc=params['xc'], yc=params['yc'], quadorder=params['quadorder'])
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
    pdgeom = PrimalDualIntervalGeometry(ptopo, dtopo, pgeom, dgeom, pdmapping)
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

derivatives
wedge products
