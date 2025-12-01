import numpy as np
import sympy
from DecLib import PETSc

#real and tangent/cotangent
class VectorBundle():
    def __init__(self, topo):
        self.topo = topo

class RealBundle(VectorBundle):
    def __init__(self, topo):
        VectorBundle.__init__(self, topo)
        self.name = 'real'
    def size(self):
        return 1
    def basis(self):
        return (np.array((1.,)),)

class TangentBundle(VectorBundle):
    def __init__(self, topo):
        VectorBundle.__init__(self, topo)
        self.name = 'tangent'
    def size(self):
        return self.topo.tdim
    def basis(self):
        if self.topo.tdim ==1:
            return (np.array((1.,)),)
        if self.topo.tdim ==2:
            return np.array((1.,0.)),np.array((0.,1.))
        if self.topo.tdim ==3:
            return np.array((1.,0.,0.)),np.array((0.,1.,0.)),np.array((0.,0.,1.0))
#FIX THIS!
    def sympy_basis(self):
        if self.topo.tdim ==1:
            return np.array((1,))
        if self.topo.tdim ==2:
            return np.array((1.,0.),(0.,1.))
        if self.topo.tdim ==3:
            return np.array((1.,0.,0.),(0.,1.,0.),(0.,0.,1.0))

class CotangentBundle(VectorBundle):
    def __init__(self, topo):
        VectorBundle.__init__(self, topo)
        self.name = 'cotangent'
    def size(self):
        return self.topo.tdim
    def basis(self):
        if self.topo.tdim ==1:
            return (np.array((1.,)),)
        if self.topo.tdim ==2:
            return np.array((1.,0.)),np.array((0.,1.))
        if self.topo.tdim ==3:
            return np.array((1.,0.,0.)),np.array((0.,1.,0.)),np.array((0.,0.,1.0))
#FIX THIS!
    def sympy_basis(self):
        if self.topo.tdim ==1:
            return np.array((1,))
        if self.topo.tdim ==2:
            return np.array((1,0.),(0.,1.))
        if self.topo.tdim ==3:
            return np.array((1,0.,0.),(0.,1.,0.),(0.,0.,1.0))

#ALL OF THIS NEEDS TO BE EXTENDED TO EXTRUDED TOPOLOGIES/GEOMETRIES AT SOME POINT
#MAYBE WITH SEPARATE EXTRUDED FORMS?
#yes, I think so, degrees become 2-tuples, etc.

#built on a topo, with a degree and a bundle
class KForm():
    def __init__(self, degree, topo, bundle, name, create_petsc=False, create_sympy=False, ndofs=1, one_form_type=None):
        self.degree = degree
        self.bundle = bundle
        self.topo = topo
        self.petsc_vec = None
        self.sympy_vec = None
        self.name = name
        self.ndofs = ndofs
        self.bsize = bundle.size()
        self.nelems = topo.nkcells[degree]
        self.one_form_type = one_form_type

        if create_petsc:
            self.create_petsc_vec()
        if create_sympy:
            self.create_sympy_vec()

#ADD THIS
    def vector_proxy(self, geom):
#0-form
#volume form
#1-form
#n-1 form
        pass

    def duplicate(self, name):
        create_petsc = not (self.petsc_vec is None)
        create_sympy = not (self.sympy_vec is None)
        return KForm(self.degree, self.topo, self.bundle, name, create_petsc = create_petsc, create_sympy = create_sympy, ndofs=self.ndofs)

    def create_GalerkinPOD_petsc_vec(self, U):
        self.petsc_vec = PETSc.Vec().createMPI(U.reduced_size)
        self.petsc_vec.assemble()

    def create_petsc_vec(self):
        self.petsc_vec = PETSc.Vec().createMPI(self.nelems * self.bsize * self.ndofs, bsize=self.bsize * self.ndofs)
        self.petsc_vec.assemble()
        self.petsc_vec.setName(self.name)

    def set_petsc_vec(self, quadrature, funclist, formtype='all', t=None, force_scalar_avg=False):
        quadrature.set_kform(self.ndofs, self.nelems, self.bundle, self.degree, funclist, self.petsc_vec, type=formtype, linetype=self.one_form_type, t=t, force_scalar_avg=force_scalar_avg)

    def create_sympy_vec(self):
        self.sympy_vec = sympy.zeros(int(self.nelems * self.bsize * self.ndofs),1)

    #BROKEN FOR NDOFS != 1 and for BVDFs
    def set_sympy_symb(self, symb):
        for i in range(self.topo.nkcells[self.degree]):
            self.sympy_vec[i] = sympy.symbols(symb+ '^' + str(self.degree)+'_'+str(i))

    #BROKEN FOR NDOFS != 1 and for BVDFs
    def set_sympy_val(self, val):
        self.sympy_vec = sympy.ones(int(self.topo.nkcells[self.degree]),1) * val

    def extract_Ib_petsc(self):
        return self.petsc_vec.getArray()[:(self.topo.nIkcells[self.degree] + self.topo.nbkcells[self.degree])]
    def extract_I_petsc(self):
        return self.petsc_vec.getArray()[:self.topo.nIkcells[self.degree]]
    def extract_b_petsc(self):
        return self.petsc_vec.getArray()[self.topo.nIkcells[self.degree]:(self.topo.nIkcells[self.degree]+self.topo.nbkcells[self.degree])]
    def extract_B_petsc(self):
        return self.petsc_vec.getArray()[(self.topo.nIkcells[self.degree]+self.topo.nbkcells[self.degree]):]

#THIS STUFF CAN BE UPDATED FOR NEW DATA STRUCTURE IE I,b,B!

#right now these assume that only the dual grid has B components, and that dofs are numbered as i,B
#THESE ONLY WORK/MAKE SENSE ON THE DUAL GRID
#eventually, should be generalized based on petsc labels...
#ie all dofs get either an i, b or B; some sort of type label
    # def extract_ib(self):
    #     if self.degree < self.topo.tdim:
    #         nk = self.topo.nkcells[self.degree]
    #         nb = self.topo.nbkcells[self.degree]
    #         return self.sympy_vec[:nk-nb,:]
    #     else:
    #         return self.sympy_vec[:,:]
    # def extract_B(self):
    #     if self.degree < self.topo.tdim:
    #         nk = self.topo.nkcells[self.degree]
    #         nb = self.topo.nbkcells[self.degree]
    #         return self.sympy_vec[nk-nb:,:]
    #     else:
    #         return self.sympy_vec[:,:]
    # def zero_B(self):
    #     if self.degree < self.topo.tdim:
    #         for bk in self.topo.bkcells[self.degree]:
    #             self.sympy_vec[bk - self.topo.kcells_off[self.degree],0] = 0
    #     return self.sympy_vec[:,:]
