
#these are topological pairing and inner product
#kform1, kform2 -> real number, possibly on different topologies
class Pairing():

    def apply(self, arg1, arg2):
        return self._apply(arg1,arg2)

    def apply_sympy(self, arg1, arg2):
        return self._apply_sympy(arg1,arg2)




#FINISH THESE
#NEED BOTH PETSC AND SYMPY BITS
class TopoPairing(Pairing):

#BROKEN FOR FOR CERTAIN FORMS ON GRIDS WITH BOUNDARIES- NEEDS TO PAIR USING ONLY SOME OF THE DOFS!
#BASICALLY NEED AN EXTRACT_IB METHOD...
    def _apply(self, arg1, arg2):
        return arg1.petsc_vec.dot(arg2.petsc_vec)

#GENERALIZE TO ARBITRARY ORDERINGS OF STRAIGHT + TWISTED
#assumes that we have x^k, ytilde^{n-k}
    def _apply_sympy(self, arg1, arg2):
        return arg1.sympy_vec.T * arg2.extract_ib()




#WRONG- IT IS MISSING THE HODGE STAR!

class InnerProduct(Pairing):

    def _apply(self, arg1, arg2):
        return arg1.petsc_vec.dot(arg2.petsc_vec)
