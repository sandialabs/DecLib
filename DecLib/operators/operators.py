from DecLib.common import ADD_MODE
from DecLib import PETSc
import sympy

# GPOD ROM operator will be dense, in general, but "relatively" small
# Compression ratios around 100? 1000? More?
# Would be useful to have a way to apply it in parallel for larger problems
# This is MPI or MPI-GPU dense linear algebra...
# There are plenty of packages for this!
# Can we store the needed matrices? Or should they just be generated at simulation time?
#
# For DDEC things are better: reduced topology is already parallelizable, just need parallel versions of GNN Hodge star and intp
#
# -Galerkin POD
# -needs U and U^T for each variable
# -needs DEIM or NN for bracket variables
# -needs DEIM or NN for Hamiltonian
#
# -DDEC
# -needs reduced topology
# -needs new Hodge stars/functionals
# -needs NN stuff for bracket variables
#
#     generate FOM data (offline)
#     generate U and DEIM/NN stuff OR reduced topology/Hodge stars/etc. (offline)
#     run the SP-ROM using generated offline data
#
#
# Should be able to write a general Galerkin POD model based on the EulerianHamiltonianModel
# ie each operator and Hamiltonian/functional knows how to apply/generate a reduced version of itself
# This should permit a lot of code reuse + commonalities!



class GalerkinPODBasis():
    def __init__(self):
        self.mat = None
        self.fom_size = fom_size
        self.reduced_size = reduced_size
        #LOAD FROM A FILE, PROBABLY? OR COMPUTE GIVEN SNAPSHOTS?
    def apply(self):
        pass
    def applyT(self):
        pass

#these are exterior derivative, boundary operators and Hodge stars
#kform1 -> kform2, possibly on different topologies and with different bundles
class UnaryOperator():

    def create_petsc_mat(self, debug=False):
        preallocator = PETSc.Mat().create()
        preallocator.setSizes((self.nresult,self.narg))
        preallocator.setType(PETSc.Mat.Type.PREALLOCATOR)
        preallocator.setBlockSizes(self.bsize_result, self.bsize_arg)
        preallocator.setUp()
        self._assemble_petsc_mat(preallocator)
        preallocator.assemble()

        self.petsc_mat = PETSc.Mat().create()
        self.petsc_mat.setSizes((self.nresult, self.narg))
        self.petsc_mat.setType('baij')
        self.petsc_mat.setBlockSizes(self.bsize_result, self.bsize_arg)
        preallocator.preallocatorPreallocate(self.petsc_mat, fill=True)
        self.petsc_mat.setUp()
        self._assemble_petsc_mat(self.petsc_mat)
        self.petsc_mat.assemble()

        preallocator.destroy()

        if debug:
            self.petsc_mat.view(viewer=stdoutinfoview)
            self.petsc_mat.view(viewer=stdoutindexview)

        self.petsc_mat.setName(self.name)

    def create_GalerkinPOD_petsc_mat(self, L, R, debug=False):
        if self.petsc_mat is None:
            self.create_petsc_mat(debug=debug)

        self.galerkin_pod_petsc_mat = L.mat.matMatMult(self.petsc_mat,R.mat)

        if debug:
            self.galerkin_pod_petsc_mat.view(viewer=stdoutinfoview)
            self.galerkin_pod_petsc_mat.view(viewer=stdoutindexview)

    def create_sympy_mat(self, debug=False):
        self.sympy_mat = sympy.zeros(int(self.nresult * self.bsize_result), int(self.narg * self.bsize_arg))
        self._assemble_sympy_mat(self.sympy_mat)

    def apply(self, arg, res, mode=ADD_MODE):
        self._apply(arg, res, mode=mode)

    def applyT(self, arg, res, mode=ADD_MODE):
        self._applyT(arg, res, mode=mode)

    def output_petsc_mat(self, viewer):
        self.petsc_mat.view(viewer)

#these are wedge products, and also interior products/triangle operators
#kform1, kform2 -> kform3, possibly on different topologies and with different bundles
class BinaryOperator():

    def create_petsc_mat_wrt_arg1(self, arg2, debug=False):
        preallocator = PETSc.Mat().create()
        preallocator.setSizes((self.nresult, self.narg1))
        preallocator.setType(PETSc.Mat.Type.PREALLOCATOR)
        preallocator.setBlockSizes(self.bsize_result, self.bsize_arg1)
        preallocator.setUp()
        self._assemble_petsc_mat_wrt_arg1(preallocator, arg2)
        preallocator.assemble()

        mat = PETSc.Mat().create()
        mat.setSizes((self.nresult, self.narg1))
        mat.setType('baij')
        mat.setBlockSizes(self.bsize_result, self.bsize_arg1)
        preallocator.preallocatorPreallocate(mat, fill=True)
        mat.setUp()
        self._assemble_petsc_mat_wrt_arg1(mat, arg2)
        mat.assemble()

        preallocator.destroy()

        if debug:
            mat.view(viewer=stdoutinfoview)
            mat.view(viewer=stdoutindexview)

        return mat

    def create_petsc_mat_wrt_arg2(self, arg1, debug=False):
        preallocator = PETSc.Mat().create()
        preallocator.setSizes((self.nresult, self.narg2))
        preallocator.setType(PETSc.Mat.Type.PREALLOCATOR)
        preallocator.setBlockSizes((self.bsize_result, self.bsize_arg2))
        preallocator.setUp()
        self._assemble_petsc_mat_wrt_arg2(preallocator, arg1)
        preallocator.assemble()

        mat = PETSc.Mat().create()
        mat.setSizes((self.nresult, self.narg2))
        mat.setType('baij')
        mat.setBlockSizes((self.bsize_result, self.bsize_arg2))
        preallocator.preallocatorPreallocate(mat, fill=True)
        mat.setUp()
        self._assemble_petsc_mat_wrt_arg2(mat, arg1)
        mat.assemble()

        preallocator.destroy()

        if debug:
            mat.view(viewer=stdoutinfoview)
            mat.view(viewer=stdoutindexview)

        return mat

    def create_petsc_mat_wrt_arg1(self, arg2, L, R, debug=False):
        petsc_mat = create_petsc_mat_wrt_arg1(arg2, debug=debug)

        galerkin_pod_petsc_mat = L.mat.matMatMult(petsc_mat,R.mat)

        if debug:
            galerkin_pod_petsc_mat.view(viewer=stdoutinfoview)
            galerkin_pod_petsc_mat.view(viewer=stdoutindexview)

        return galerkin_pod_petsc_mat

    def create_petsc_mat_wrt_arg2(self, arg1, L, R, debug=False):
        petsc_mat = create_petsc_mat_wrt_arg2(arg1, debug=debug)

        galerkin_pod_petsc_mat = L.mat.matMatMult(petsc_mat,R.mat)

        if debug:
            galerkin_pod_petsc_mat.view(viewer=stdoutinfoview)
            galerkin_pod_petsc_mat.view(viewer=stdoutindexview)

        return galerkin_pod_petsc_mat

    def create_sympy_mat_wrt_arg1(self, arg2, debug=False):
        sympy_mat = sympy.zeros(int(self.nresult * self.bsize_result), int(self.narg1 * self.bsize_arg1))
        self._assemble_sympy_mat_wrt_arg1(sympy_mat, arg2)
        return sympy_mat

    def create_sympy_mat_wrt_arg2(self, arg1, debug=False):
        sympy_mat = sympy.zeros(int(self.nresult * self.bsize_result), int(self.narg2 * self.bsize_arg2))
        self._assemble_sympy_mat_wrt_arg2(sympy_mat, arg1)
        return sympy_mat

    #given a and b, compute c
    def apply(self, arg1, arg2, res, mode=ADD_MODE):
        self._apply(arg1, arg2, res, mode=mode)

    #given b and c, compute a for binary adjoint1
    def applyT1(self, arg1, arg2, res, mode=ADD_MODE):
        self._applyT1(arg1, arg2, res, mode=mode)

    #given a and c, compute b for binary adjoint2
    def applyT2(self, arg1, arg2, res, mode=ADD_MODE):
        self._applyT2(arg1, arg2, res, mode=mode)

    #given b and c, compute a
    def solve1(self, arg1, arg2, res):
        self._solve1(arg1, arg2, res)

    #given a and c, compute b
    def solve2(self, arg1, arg2, res):
        self._solve2(arg1, arg2, res)
