from DecLib import PETSc
from DecLib import createDualTopology
from DecLib import PrimalIntervalMesh, IntervalGeometry
from DecLib import plot_mesh1D, pd_mesh_plot_1D
from DecLib import ExtDeriv, RealBundle, KForm, TopoPairing
import sympy

nx = 3
bnd = 'periodic'
#GENERALIZE TO MIXED BCS HERE EVENTUALLY!!! ie B for primal on left and B for dual on right...
#right now code just does B for dual everywhere

ptopo = PrimalIntervalMesh(nx, bnd)
dtopo, pdmapping = createDualTopology(ptopo)

pgeom, dgeom, _ = IntervalGeometry(ptopo, dtopo, pdmapping, Lx=1.0, xc=0.5, quadorder=3)

#print(dgeom.edgevertexlocs.getArray())

#print(dgeom.cellvertexlocs.getArray())

plot_mesh1D(ptopo, pgeom, 'primal')
plot_mesh1D(dtopo, dgeom, 'dual')
pd_mesh_plot_1D(ptopo, dtopo, pdmapping, pgeom, dgeom, 'primal-dual')

PRbund = RealBundle(ptopo)
DRbund = RealBundle(dtopo)

D = ExtDeriv(1, ptopo, PRbund)
D.create_sympy_mat()

Dbar = ExtDeriv(1, dtopo, DRbund)
Dbar.create_sympy_mat()

topopair = TopoPairing()

#print("D")
#sympy.pprint(D.sympy_mat)
#print("Dbar")
#sympy.pprint(Dbar.sympy_mat)

x0 =  KForm(0, ptopo, PRbund, 'x0', create_sympy=True)
x0.set_sympy_symb('x')

xtilde0 = KForm(0, ptopo, PRbund, 'xtilde0', create_sympy=True)
xtilde0.set_sympy_symb('xtilde')

x1 = KForm(1, ptopo, PRbund, 'x1', create_sympy=True)
x1.set_sympy_symb('x')

xtilde1 = KForm(1, ptopo, PRbund, 'xtilde1', create_sympy=True)
xtilde1.set_sympy_symb('xtilde')

qtilde0 = KForm(0, ptopo, PRbund, 'q0', create_sympy=True)
qtilde0.set_sympy_symb('qtilde')

qtilde0sq = KForm(0, ptopo, PRbund, 'q2', create_sympy=True)
qtilde0sq.set_sympy_symb('qtilde')
qtilde0sq.sympy_vec = qtilde0sq.sympy_vec.multiply_elementwise(qtilde0.sympy_vec)/2

I0 = KForm(1, ptopo, PRbund, 'I0', create_sympy=True)
I0.set_sympy_val(1)

Itilde0 = KForm(1, ptopo, PRbund, 'Itilde0', create_sympy=True)
Itilde0.set_sympy_val(1)

#sympy.pprint(x0.sympy_vec, use_unicode=True)
#sympy.pprint(xtilde0.sympy_vec, use_unicode=True)
#sympy.pprint(x1.sympy_vec)
#sympy.pprint(qtilde0.sympy_vec)
#sympy.pprint(qtilde0sq.sympy_vec)
#sympy.pprint(I0.sympy_vec)
#sympy.pprint(Itilde0.sympy_vec)

#R =
#Wv, Wt =
#Qv, Qt =

#print('checking pv compat/steady geostrophic for Wv/Wvt')



print('checking D I0 = 0 and Dbar Itilde0 = 0')
sympy.pprint(sympy.simplify(D.sympy_mat * I0.sympy_vec) == sympy.zeros(*(D.sympy_mat * I0.sympy_vec).shape))
sympy.pprint(sympy.simplify(Dbar.sympy_mat * Itilde0.sympy_vec) == sympy.zeros(*(Dbar.sympy_mat * Itilde0.sympy_vec).shape))

if bnd == 'periodic':
    print('checking D/Dbar adjoints ie IBP')
    sympy.pprint((Dbar.sympy_mat.T + D.sympy_mat) == sympy.zeros(*D.sympy_mat.shape))
    sympy.pprint(sympy.simplify(x0.sympy_vec.T * Dbar.sympy_mat * xtilde0.sympy_vec + xtilde0.sympy_vec.T * D.sympy_mat * x0.sympy_vec))
#COMPOSITION OF TOPO PAIRING AND EXT DERIV IS PRETTY BROKEN, UGGH
    #sympy.pprint(sympy.simplify(topopair.apply_sympy(x0.sympy_vec, Dbar.sympy_mat * xtilde0.sympy_vec) + topopair.apply_sympy(D.sympy_mat * x0.sympy_vec, xtilde0.sympy_vec)))


    print('checking mass cons')
    sympy.pprint(sympy.simplify(I0.sympy_vec.T * Dbar.sympy_mat * xtilde0.sympy_vec))
    #sympy.pprint(sympy.simplify(topopair.apply_sympy(I0.sympy_vec, Dbar.sympy_mat * xtilde0.sympy_vec)))

#GENERALIZE TO MIXED BCS HERE EVENTUALLY!!! ie B for primal on left and B for dual on right...
else:
    pass

    #print('checking mass cons')
    #sympy.pprint(sympy.simplify(topological_inner_product(I0,Dbar * xtilde0,0,dtopo)))
    #sympy.pprint(sympy.simplify(topological_inner_product(I0,Dbar * Ftilde0,0,dtopo)))
    #sympy.pprint(sympy.simplify(I0.T * Bbar * xtilde0))
    #sympy.pprint(sympy.simplify(I0.T * Bbar * Ftilde0))


    #print('checking D.T + Dbar = B and Dbar.T + D = Bbar and B = Bbar.T ie IBP')
    #sympy.pprint(sympy.simplify(topological_inner_product(x0,Dbar * xtilde0,0,dtopo) + topological_inner_product(D * x0,xtilde0,1,dtopo) + xtilde0.T * B * x0))
    #sympy.pprint(sympy.simplify(topological_inner_product(x0,Dbar * xtilde0,0,dtopo) + topological_inner_product(D * x0,xtilde0,1,dtopo) - x0.T * Bbar * xtilde0))
