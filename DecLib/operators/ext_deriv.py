
from DecLib.operators.operators import UnaryOperator
import numpy as np

# def Bops(stopo, ttopo, stmapping, debug=False):
#
#     nnzBbaR = np.ones(stopo.nkcells[0],dtype=np.int32)
#     nnzB = np.ones(ttopo.nkcells[0],dtype=np.int32)
#
#     Bbar =  PETSc.Mat().createAIJ((stopo.nkcells[0],ttopo.nkcells[1]),nnz=nnzBbaR)
#     B = PETSc.Mat().createAIJ((ttopo.nkcells[0],stopo.nkcells[1]),nnz=nnzB)
#
#     for bnm1 in ttopo.bkcells[1]:
#         orient = ttopo.higher_orientation(bnm1)
#         Bbar.setValues(rows=stmapping.dbnmk_to_pbk(bnm1)- stopo.kcells_off[0], cols=bnm1 - ttopo.kcells_off[1], values=orient[0])
#
#     for bnm1 in stopo.bkcells[1]:
#         orient = stopo.higher_orientation(bnm1)
#         B.setValues(rows=stmapping.pbk_to_dbnmk(bnm1)- ttopo.kcells_off[0], cols=bnm1 - stopo.kcells_off[1], values=orient[0])
#
#     Bbar.assemble()
#     B.assemble()
#
#     if debug:
#         Bbar.view(viewer=stdoutinfoview)
#         Bbar.view(viewer=stdoutindexview)
#         B.view(viewer=stdoutinfoview)
#         B.view(viewer=stdoutindexview)
#
#     return B, Bbar





class PrimalBoundaryOp(UnaryOperator):
    def __init__(self, degree, stopo, ttopo, stmapping, bundle):
        self.stopo = stopo
        self.ttopo = ttopo
        self.stmapping = stmapping
        self.narg = self.stopo.nkcells[degree]
        self.degree = degree
        self.nresult =  self.ttopo.nkcells[self.stopo.tdim - degree - 1]
        self.bsize_result = bundle.size()
        self.bsize_arg = bundle.size()
        self.name = 'B-'+str(bundle.name)

#SOMETHING IS BASICALLY THE INDEX OF THE ASSOCIATED HIGHER DIMENSIONAL CELL I THINK?
#NEED TO THINK ABOUT THIS WORKS IN GENERAL...
    def _assemble_petsc_mat(self, mat):
        for bk in range(self.stopo.nIbkcells[self.degree], self.stopo.nIbkcells[self.degree] + self.stopo.nBkcells[self.degree]):
            incidence = self.stopo.higher_incidence_numbers(bk)
            mat.setValues(rows=self.stmapping.sB_to_tb(bk)- self.ttopo.kcells_off[self.stopo.tdim - degree - 1], cols=bk - self.stopo.kcells_off[self.degree], values=incidence[SOMETHING])

    def _assemble_sympy_mat(self, mat):
        for bk in range(self.stopo.nIbkcells[self.degree], self.stopo.nIbkcells[self.degree] + self.stopo.nBkcells[self.degree]):
            incidence = self.stopo.higher_incidence_numbers(bk)
            mat[self.stmapping.sB_to_tb(bk)- self.ttopo.kcells_off[self.stopo.tdim - degree - 1], bk - self.stopo.kcells_off[self.degree]] = incidence[SOMETHING]


class DualBoundaryOp(UnaryOperator):
    def __init__(self, degree, stopo, ttopo, stmapping, bundle):
        self.stopo = stopo
        self.ttopo = ttopo
        self.stmapping = stmapping
        self.narg = self.ttopo.nkcells[degree]
        self.degree = degree
        self.nresult =  self.stopo.nkcells[self.stopo.tdim - degree - 1]
        self.bsize_result = bundle.size()
        self.bsize_arg = bundle.size()
        self.name = 'Bbar-'+str(bundle.name)

#SOMETHING IS BASICALLY THE INDEX OF THE ASSOCIATED HIGHER DIMENSIONAL CELL I THINK?
#NEED TO THINK ABOUT THIS WORKS IN GENERAL...
    def _assemble_petsc_mat(self, mat):
        for bk in range(self.ttopo.nIbkcells[self.degree], self.ttopo.nIbkcells[self.degree] + self.ttopo.nBkcells[self.degree]):
            incidence = self.ttopo.higher_incidence_numbers(bk)
            mat.setValues(rows=self.stmapping.tB_to_Sb(bk)- self.stopo.kcells_off[self.stopo.tdim - degree - 1], cols=bk - self.ttopo.kcells_off[self.degree], values=incidence[SOMETHING])

    def _assemble_sympy_mat(self, mat):
        for bk in range(self.ttopo.nIbkcells[self.degree], self.ttopo.nIbkcells[self.degree] + self.ttopo.nBkcells[self.degree]):
            incidence = self.ttopo.higher_incidence_numbers(bk)
            mat[self.stmapping.tB_to_Sb(bk)- self.stopo.kcells_off[self.stopo.tdim - degree - 1], bk - self.ttopo.kcells_off[self.degree]] = incidence[SOMETHING]


#THESE ARE 2D SPECIFIC FOR NOW....
#SHOULD BE EXPRESSIBLE IN A DIMENSION INDEP WAY HOWEVER...
#ARE THERE MULTIPLE BOUNDARY OPERATORS IN 3D?


# dim = 2
# class PrimalBoundaryOp(UnaryOperator):
#     def __init__(self, stopo, ttopo, stmapping, bundle):
#         self.stopo = stopo
#         self.ttopo = ttopo
#         self.stmapping = stmapping
#         self.narg = self.stopo.nkcells[dim-1]
#         self.nresult =  self.ttopo.nkcells[0]
#         self.bsize_result = bundle.size()
#         self.bsize_arg = bundle.size()
#         self.name = 'B-'+str(bundle.name)
#
#     def _assemble_petsc_mat(self, mat):
#         for bnm1 in self.stopo.bkcells[1]:
#             orient = self.stopo.higher_incidence_numbers(bnm1)
#             mat.setValues(rows=self.stmapping.pbk_to_dbnmk(bnm1)- self.ttopo.kcells_off[0], cols=bnm1 - self.stopo.kcells_off[1], values=orient[0])
#
#     def _assemble_sympy_mat(self, mat):
#         for bnm1 in self.stopo.bkcells[dim-1]:
#             orient = self.stopo.higher_incidence_numbers(bnm1)
#             mat[self.stmapping.pbk_to_dbnmk(bnm1)- self.ttopo.kcells_off[0], bnm1 - self.stopo.kcells_off[dim-1]] = orient[0]
#
# class DualBoundaryOp(UnaryOperator):
#     def __init__(self, stopo, ttopo, stmapping, bundle):
#         self.stopo = stopo
#         self.ttopo = ttopo
#         self.stmapping = stmapping
#         self.narg = self.ttopo.nkcells[dim-1]
#         self.nresult =  self.stopo.nkcells[0]
#         self.bsize_result = bundle.size()
#         self.bsize_arg = bundle.size()
#         self.name = 'Bbar-'+str(bundle.name)
#
#     def _assemble_petsc_mat(self, mat):
#          for bnm1 in self.ttopo.bkcells[1]:
#              orient = self.ttopo.higher_incidence_numbers(bnm1)
#              mat.setValues(rows=self.stmapping.dbnmk_to_pbk(bnm1)- self.stopo.kcells_off[0], cols=bnm1 - self.ttopo.kcells_off[1], values=orient[0])
#
#     def _assemble_sympy_mat(self, mat):
#         for bnm1 in self.ttopo.bkcells[dim-1]:
#             orient = self.ttopo.higher_incidence_numbers(bnm1)
#             mat[self.stmapping.dbnmk_to_pbk(bnm1)- self.stopo.kcells_off[0], bnm1 - self.ttopo.kcells_off[dim-1]] = orient[0]

#maybe eventually add C code to set values, and also do various apply operations?
#Here we are assuming a flat bundle!
class ExtDeriv(UnaryOperator):
    def __init__(self, degree, topo, bundle):
        self.degree = degree
        self.topo = topo
        self.narg = self.topo.nkcells[self.degree]
        self.nresult =  self.topo.nkcells[self.degree+1]
        self.bsize_result = bundle.size()
        self.bsize_arg = bundle.size()
        self.name = 'D'+str(degree)+'-'+str(bundle.name)+'-'+str(topo.name)

    def _assemble_petsc_mat(self, mat):
        for pk in range(self.topo.kcells[self.degree+1][0], self.topo.kcells[self.degree+1][1]):
            km1cells = self.topo.lower_dim_TC(pk,self.degree)
            orients = self.topo.lower_incidence_numbers(pk)
            vals = np.zeros(self.bsize_result * self.bsize_arg * km1cells.shape[0])
            #print(pk,km1cells, orients, vals.shape)
#IS THIS CORRECT FOR BVDFS? NO!
            #for i in range(self.bsize_result):
#            vals = orients[:len(km1cells)]
#            for i in range(self.bsize_result):

            #BASICALLY NEED TO CONCATENATE A BUNCH OF THINGS TOGETHER HERE...
            #print(km1cells,orients,vals.shape)
            j = 0
            for pkm1,orient in zip(km1cells,orients):
                vals[j] = orient
                j = j+1

            mat.setValuesBlocked(
            rows=pk-self.topo.kcells_off[self.degree+1],
            cols=km1cells-self.topo.kcells_off[self.degree],
            values=vals)

#IS THIS CORRECT FOR BVDFS? NO!
    def _assemble_sympy_mat(self, mat):
        for pk in range(self.topo.kcells[self.degree+1][0], self.topo.kcells[self.degree+1][1]):
            km1cells = self.topo.lower_dim_TC(pk,self.degree)
            orients = self.topo.lower_incidence_numbers(pk)
            for pkm1,orient in zip(km1cells,orients):
                mat[pk - self.topo.kcells_off[self.degree+1], pkm1 - self.topo.kcells_off[self.degree]] = orient


#REVERSED VERSION!
class ExtDeriv_Higher(ExtDeriv):
    def _assemble_petsc_mat(self, mat):
        for pkm1 in range(self.topo.kcells[self.degree][0], self.topo.kcells[self.degree][1]):
            pks = self.topo.higher_dim_TC(pkm1,self.degree+1)
            orients = self.topo.higher_incidence_numbers(pkm1)
            vals = np.zeros(self.bsize_result * self.bsize_arg * pks.shape[0])
            #print(pk,km1cells, orients, vals.shape)
#IS THIS CORRECT FOR BVDFS? NO!
            #for i in range(self.bsize_result):
#            vals = orients[:len(km1cells)]
#            for i in range(self.bsize_result):

            #BASICALLY NEED TO CONCATENATE A BUNCH OF THINGS TOGETHER HERE...
            #print(km1cells,orients,vals.shape)
            j = 0
            for pk,orient in zip(pks,orients):
                vals[j] = orient
                j = j+1

            mat.setValuesBlocked(
            rows=pks-self.topo.kcells_off[self.degree+1],
            cols=pkm1-self.topo.kcells_off[self.degree],
            values=vals)

    def _assemble_sympy_mat(self, mat):
        for pkm1 in range(self.topo.kcells[self.degree][0], self.topo.kcells[self.degree][1]):
            pks = self.topo.higher_dim_TC(pkm1,self.degree+1)
            orients = self.topo.higher_incidence_numbers(pkm1)
            for pk,orient in zip(pks,orients):
                mat[pk - self.topo.kcells_off[self.degree+1], pkm1 - self.topo.kcells_off[self.degree]] = orient
