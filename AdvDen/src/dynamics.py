
from DecLib import PETSc


#Should compose (at runtime) Hamiltonians, Poisson Brackets and a varset/thermo
#To generate dynamics
#ie compute Jvars, dHdx in separate routines
#and then rhs as J(Jvars) * dHdx
#This will support mixing various types of time steppers ie explicit vs EC2SI very easily...

#generate a linear system as well, via J(J0) * dHlindx
#Hamiltonians and Poisson brackets know how to linearize themselves

#only slightly tricky bit is going to be possibly how to handle collapsing down to a simple Helmholtz system ie elimination?

#For Galerkin POD models- basically each bracket and varset takes the appropriate basis, and forms the relevant ROM operators!
#This should be very doable in a general way, that easily allows switching between FOM and GPOD-ROM

#For DDEC ROM, we basically learn new functionals and new intp operators
#This can maybe be a new type of Hodge star and intp operators, etc.?

class Dynamics():
    def __init__(self, meshes, params, ic):
        self.meshes = meshes
        self.IC = ic
        self.params = params
        self.dim = meshes.dim

        self.const_vars = {}
        self.prog_vars = {}
        self.aux_vars = {}

    def set_IC(self):
        self.IC.set(self)
        self.IC.set_IC()

class Statistics():
    def __init__(self, dyn):
        pass
    def compute(step, statnum):
        pass

class Statistic():
    def __init__(self, name, Nsteps, nstat, ndofs=1):
        self.name = name
        self.statsize = Nsteps//nstat + 1
        self.ndofs = ndofs
        self.petsc_vec =  PETSc.Vec().create()
        self.petsc_vec.setSizes(self.statsize * ndofs, bsize=ndofs)
    #SHOULD PROBABLY BE SEQUENTIAL, ACTUALLY...
        self.petsc_vec.setType(PETSc.Vec.Type.MPI)
        self.petsc_vec.assemble()
        self.petsc_vec.setName(name)
        
class Diagnostics():
    def __init__(self, dyn):
        pass

    def compute(step, diagnum):
        pass

class PoissonBracket():
	def __init__(self,):
		pass
	def compute_Jvars(self,):
		pass
