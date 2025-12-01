from DecLib.operators.operators import UnaryOperator

class Codiff(UnaryOperator):
    def __init__(self, hodgeStarLeft, hodgeStarRight):
        # codifferential is defined as delta = (-1)^k (hodgeStarLeft) d (hodgeStarRight), where d is the exterior derivative
        self.hodgeStarLeft  = hodgeStarLeft
        self.hodgeStarRight = hodgeStarRight
        self.k = hodgeStarRight.k # we operate on the same k-forms as hodgeStarRight; we produce k-1 forms as output.
        self.meshes = hodgeStarRight.meshes
        self.ptopo  = self.meshes.ptopo
        n = self.ptopo.dim
        self.narg = hodgeStarRight.narg # we act on the input of the rightmost Hodge star
        self.nresult = hodgeStarLeft.nresult # we produce the same number of outputs as the left Hodge star
        self.extDeriv = ExtDeriv(n-k, self.ptopo, hodgeStarRight.dbundle) # exterior derivative operates on the output of rightmost Hodge star; I think this means we should use that Hodge star's dual bundle.
        self.name = '(-1)^'+str(k)+'('+hodgeStarLeft.name+').'+ self.extDeriv.name +'.('+hodgeStarRight.name+')'

    def _assemble_petsc_mat(self, mat):
        # want to call hodgeStar and extDerivative's _assemble_petsc_mat, but we need to allocate appropriately
        # sized matrices first; each is op.nresult by op.narg, where ops are (from left to right):
        # hodgeStarLeft, extDerivative, hodgeStarRight
        # but superclass (UnaryOperator) already has a create_petsc_mat; we should be able to just call that
        hodgeStarLeft.create_petsc_mat()  # allocates into hodgeStarLeft.petsc_mat, and assembles
        extDerivative.create_petsc_mat()  # allocates into extDerivative.petsc_mat, and assembles
        hodgeStarRight.create_petsc_mat() # allocates into hodgeStarRight.petsc_mat, and assembles

        hLeftMat  =  hodgeStarLeft.petsc_mat
        extDerMat =  extDerivative.petsc_mat
        hRightMat = hodgeStarRight.petsc_mat

        minus_one_factor = 1 if (self.k % 2 == 0) else -1
        # fill mat with hLeftMat * extDerMat * hRightMat
        hLeftMat.matMatMult(extDerMat,hRightMat,mat)
        mat.scale(minus_one_factor)

    def _assemble_sympy_mat(self, mat):
        hodgeStarLeft.create_sympy_mat()  # allocates into  hodgeStarLeft.sympy_mat, and assembles
        extDerivative.create_sympy_mat()  # allocates into  extDerivative.sympy_mat, and assembles
        hodgeStarRight.create_sympy_mat() # allocates into hodgeStarRight.sympy_mat, and assembles

        hLeftMat  =  hodgeStarLeft.sympy_mat
        extDerMat =  extDerivative.sympy_mat
        hRightMat = hodgeStarRight.sympy_mat

        minus_one_factor = 1 if (self.k % 2 == 0) else -1
        mat[:,:] = minus_one_factor * hLeftMat * extDerMat * hRightMat
