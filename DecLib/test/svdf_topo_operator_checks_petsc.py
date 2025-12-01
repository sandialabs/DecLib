#from DecLib import PrimalIntervalMesh, PrimalIntervalQuadrature, DualIntervalGeometry, DualIntervalQuadrature, PrimalDualIntervalGeometry
#from DecLib import PrimalQuadMesh, DualQuadGeometry, PrimalDualQuadGeometry
#from DecLib import TriangleMesh, meshzooMesh, dmshMesh
#from DecLib import createDualTopology, PrimalDualGeometry, DualGeometry
#from DecLib import Meshes
#from DecLib import plot_mesh1D, pd_mesh_plot_1D, plot_mesh2D, pd_mesh_plot_2D

from DecLib import create_meshes
from DecLib import PETSc
from DecLib import ExtDeriv, TopoPairing, RealBundle, PrimalBoundaryOp, DualBoundaryOp

import yaml
import sys
import numpy as np
import sympy

def check_topo_operator_properties_1D(meshes, check_petsc=True, check_sympy=True):
    Rbund = RealBundle(meshes.ptopo, 'real')
    PSbund = RealBundle(meshes.dtopo, 'ps')

    D = ExtDeriv(1, meshes.ptopo, Rbund)
    Dbar = ExtDeriv(1, meshes.dtopo, PSbund)

    D.create_petsc_mat()
    Dbar.create_petsc_mat()

    Dmat = D.petsc_mat.convert('aij')
    Dbarmat = Dbar.petsc_mat.convert('aij')


def check_topo_operator_properties_2D(meshes, check_petsc=True, check_sympy=True):
    Rbund = RealBundle(meshes.ptopo, 'real')
    PSbund = RealBundle(meshes.dtopo, 'ps')

    D2 = ExtDeriv(2, meshes.ptopo, Rbund)
    D1 = ExtDeriv(1, meshes.ptopo, Rbund)
    D2bar = ExtDeriv(2, meshes.dtopo, PSbund)
    D1bar = ExtDeriv(1, meshes.dtopo, PSbund)

    B = PrimalBoundaryOp(meshes.ptopo, meshes.dtopo, meshes.pdmapping, PSbund)
    Bbar = DualBoundaryOp(meshes.ptopo, meshes.dtopo, meshes.pdmapping, Rbund)

    D2.create_petsc_mat()
    D1.create_petsc_mat()
    D2bar.create_petsc_mat()
    D1bar.create_petsc_mat()
    B.create_petsc_mat()
    Bbar.create_petsc_mat()

    D2mat = D2.petsc_mat.convert('aij')
    D1mat = D1.petsc_mat.convert('aij')
    D2barmat = D2bar.petsc_mat.convert('aij')
    D1barmat = D1bar.petsc_mat.convert('aij')
    Bmat = B.petsc_mat.convert('aij')
    Bbarmat = Bbar.petsc_mat.convert('aij')

    print('checking D2 D1 = 0 and D2bar D1bar = 0')
    D2D1 = D2mat.matMult(D1mat)
    D2barD1bar = D2barmat.matMult(D1barmat)
    _,_,D2D1vals = D2D1.getValuesCSR()
    _,_,D2barD1barvals = D2barD1bar.getValuesCSR()
    print('D2D1',np.min(D2D1vals),np.max(D2D1vals))
    print('D2barD1bar',np.min(D2barD1barvals),np.max(D2barD1barvals))

#         print('checking D1bar/D2 and D1/D2bar adjoints')
# #FIX THIS SO IT IS ACTUALLY CORRECT!
#         D1T = D1mat.transpose()
#         D1barT = D1mat.transpose()
#         _,_,D2vals = D2mat.getValuesCSR()
#         _,_,D2barvals = D2barmat.getValuesCSR()
#         _,_,D1Tvals = D1T.getValuesCSR()
#         _,_,D1barTvals = D1barT.getValuesCSR()
#         print(D2vals + D1barTvals)
#         print(D2barvals + D1Tvals)

# print('checking D2 D1 = 0 and D2bar D1bar = 0')
# sympy.pprint(D2 * D1 == sympy.zeros(*(D2 * D1).shape))
# sympy.pprint(D2bar * D1bar == sympy.zeros(*(D2bar * D1bar).shape))
#
# print('checking D1 I0 = 0 and D1bar Itilde0 = 0')
# sympy.pprint(sympy.simplify(D1 * I0) == sympy.zeros(*(D1 * I0).shape))
# sympy.pprint(sympy.simplify(D1bar * Itilde0) == sympy.zeros(*(D1bar * Itilde0).shape))
#
# print('checking D1bar.T + D2 = B and D1.T + D2bar = Bbar ie IBP')
# sympy.pprint(sympy.simplify(topological_inner_product(x0,D2bar * xtilde1,0,dtopo) + topological_inner_product(D1 * x0,xtilde1,1,dtopo) + x0.T * Bbar * xtilde1))
# sympy.pprint(sympy.simplify(topological_inner_product(x1,D1bar * xtilde0,1,dtopo) + topological_inner_product(D2 * x1,xtilde0,2,dtopo) + xtilde0.T * B * x1))


    #DD = 0
    #IBP properties
    #wedge properties


#meshtype = 'dmsh' #'Triangle' 'square' 'meshzoo' 'line' 'dmsh'
#dualtype = 'centroid' #'circumcenter' 'centroid'

meshparams = {}
meshparams['Lx'] = 1.0
meshparams['Ly'] = 1.0
meshparams['xc'] = 0.5
meshparams['yc'] = 0.5

for xbc in ['none', 'periodic']:
    meshparams['xbc'] = xbc
    for nx in range(3,5):
        meshparams['nx'] = nx
        name = xbc + '-' + str(nx) + '-'
        print('*****   ' + name + '    *****')
        meshes = create_meshes('line', meshparams, None, False, None, 3, plotmeshes=False, name=name, creategeom=False, createpdgeom=False)
        check_topo_operator_properties_1D(meshes)

#FIX PARTIALLY PERIODIC STUFF
#IS IT JUST PLOTTING OR ALSO GEOMETRY STUFF?

#['none','none']
for xbc, ybc in [['periodic','periodic'], ['periodic','none'] , ['none','periodic']]: #,
    meshparams['xbc'] = xbc
    meshparams['ybc'] = ybc
    for nx in range(3,5):
        meshparams['nx'] = nx
        for ny in range(3,5):
            meshparams['ny'] = nx
            name = xbc + '-' + ybc + '-' + str(nx) + '-' + str(ny) + '-'
            print('*****   ' + name + '    *****')
            meshes = create_meshes('square', meshparams, 'centroid', False, None, 3, plotmeshes=False, name=name, creategeom=False, createpdgeom=False)
            check_topo_operator_properties_2D(meshes)

meshparams['zootype'] = 'disk'
for zoosizes in [[1,6], [2,6], [3,6], [4,6], [5,6]]: #
    meshparams['zoosizes'] = zoosizes
    name = 'meshzoo-disk-' + str(zoosizes[0]) + '-' + str(zoosizes[1]) + '-'
    print('*****   ' + name + '    *****')
    meshes = create_meshes('meshzoo', meshparams, None, False, None, 3, plotmeshes=False, name=name, creategeom=False, createpdgeom=False)
    check_topo_operator_properties_2D(meshes)

meshparams['zootype'] = 'rectangle-tri'
for zoosizes in [[4,4], [4,5], [5,4], [5,5]]: #
    meshparams['zoosizes'] = zoosizes
    name = 'meshzoo-rectangle-tri-' + str(zoosizes[0]) + '-' + str(zoosizes[1]) + '-'
    print('*****   ' + name + '    *****')
    meshes = create_meshes('meshzoo', meshparams, None, False, None, 3, plotmeshes=False, name=name, createpdgeom=False)
    check_topo_operator_properties_2D(meshes)

for dmshtype in ['square', 'circ']:
    meshparams['dmshtype'] = dmshtype
    for dmshsize in [0.33, 0.22, 0.11]: #
        meshparams['dmshsize'] = dmshsize
        name = 'dmsh-' + dmshtype + '-' + str(dmshsize) + '-'
        print('*****   ' + name + '    *****')
        meshes = create_meshes('dmsh', meshparams, None, False, None, 3, plotmeshes=False, name=name, createpdgeom=False)
        check_topo_operator_properties_2D(meshes)

#ADD GMSH STUFF

#ADD 3D MESH STUFF- THIS SHOULDN'T BE TOO HARD...
#BOX QUAD MESHES- PERIODIC AND NONE
#SOME SORT OF BOX TET MESH- WITH GMSH, PERIODIC AND NONE
