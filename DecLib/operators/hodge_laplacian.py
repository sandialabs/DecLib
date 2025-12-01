from DecLib.operators.operators import UnaryOperator

class HodgeLaplacian(UnaryOperator):
    def __init__(self, codiff_k, codiff_kp1):
        # Hodge Laplacian is defined as d delta + delta d, where d is the exterior derivative, and delta is the codifferential
        # The exterior derivative is a topological operator, so we don't need the constructor to specify that.
        # The codifferential is a metric operator, so there are choices, and we do need the constructor to specify that.
        # Because our operators embed the k for the k-form, and the two deltas operate on different forms, we need two deltas;
        # delta_k, and delta_{k+1}, respectively.  The exterior derivative maps from k-forms to (k+1)-forms; the codifferential
        # maps from k-forms to (k-1)-forms.  So the Hodge Laplacian maps from k-forms to k-forms.
        self.codiff_k   = codiff_k
        self.codiff_kp1 = codiff_kp1
        self.k = codiff_k.k # we operate on the same k-forms as codiff_k; we produce k-forms as output.
        self.meshes = codiff_k.meshes
        self.ptopo  = self.meshes.ptopo
        n = self.ptopo.dim
        self.narg = codiff_k.narg # we act on the input of codiff_k
        self.nresult = codiff_kp1.nresult # we produce the same number of outputs as codiff_kp1 (which should also match our number of inputs)
        self.extDeriv_k   = ExtDeriv(k,   self.ptopo, codiff_k.bundle) # extDeriv_k operates on the same bundle as codiff_k
        self.extDeriv_km1 = ExtDeriv(k-1, self.ptopo, codiff_k.bundle) # extDeriv_km1 operates on the same bundle as codiff_k
        self.name = extDeriv_km1.name + ' ' + codiff_k.name + ' ' + self.codiff_kp1.name +' ' + extDeriv_k.name

    def _assemble_petsc_mat(self, mat):
        # our operator is codiff_kp1 * extDeriv_k + extDeriv_km1 * codiff_k
        codiff_kp1.create_petsc_mat()   # allocates into .petsc_mat, and assembles
        extDeriv_k.create_petsc_mat()   # allocates into .petsc_mat, and assembles
        extDeriv_km1.create_petsc_mat() # allocates into .petsc_mat, and assembles
        codiff_k.create_petsc_mat()     # allocates into .petsc_mat, and assembles

        mat_codiff_kp1   =   codiff_kp1.petsc_mat
        mat_extDeriv_k   =   extDeriv_k.petsc_mat
        mat_extDeriv_km1 = extDeriv_km1.petsc_mat
        mat_codiff_k     =     codiff_k.petsc_mat

        # fill mat with mat_codiff_kp1 * mat_extDeriv_k
        mat_codiff_kp1.matMult(mat_extDeriv_k, mat)
        # add in extDeriv_km1 * codiff_k
        mat.axpy(1.0,mat_extDeriv_km1.matMult(mat_codiff_k))

    def _assemble_sympy_mat(self, mat):
        # our operator is codiff_kp1 * extDeriv_k + extDeriv_km1 * codiff_k
        codiff_kp1.create_sympy_mat()   # allocates into .sympy_mat, and assembles
        extDeriv_k.create_sympy_mat()   # allocates into .sympy_mat, and assembles
        extDeriv_km1.create_sympy_mat() # allocates into .sympy_mat, and assembles
        codiff_k.create_sympy_mat()     # allocates into .sympy_mat, and assembles

        mat_codiff_kp1   =   codiff_kp1.sympy_mat
        mat_extDeriv_k   =   extDeriv_k.sympy_mat
        mat_extDeriv_km1 = extDeriv_km1.sympy_mat
        mat_codiff_k     =     codiff_k.sympy_mat

        # fill mat with mat_codiff_kp1 * mat_extDeriv_k + extDeriv_km1 * codiff_k
        mat[:,:] = mat_codiff_kp1 * mat_extDeriv_k + extDeriv_km1 * codiff_k

    def _apply(self, arg, res, mode=ADD_MODE):
        # TODO: figure out what we should do to implement this
        # something like: self.petsc_mat.mult(arg,res) -- but need to account for mode argument, too...
