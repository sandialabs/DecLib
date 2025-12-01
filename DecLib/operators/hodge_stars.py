from DecLib.operators.operators import UnaryOperator
import numpy as np

#maybe eventually add C code to set values, and also do various apply operations?
#Here we assume bundle and it's dual have the same "basis", and thus the block of values is actually diagonal!
class VoronoiStarStoT(UnaryOperator):
    def __init__(self, k, meshes, sbundle, tbundle):
        dim = stopo.tdim
        self.meshes = meshes
        self.narg = meshes.stopo.nkcells[k]
        self.nresult = meshes.ttopo.nkcells[dim-k]
        self.bsize_result = tbundle.size()
        self.bsize_arg = sbundle.size()
        self.k = k
        self.name = 'H'+str(k)+'-'+str(sbundle.name)+'-'+str(stopo.name)

    def _assemble_petsc_mat(self, mat):
        stopo, ttopo, sgeom, tgeom, stmapping = self.meshes.stopo, self.meshes.ttopo, self.meshes.sgeom, self.meshes.tgeom, self.meshes.stmapping
        dim = stopo.tdim
        k = self.k

        vals = np.zeros(self.bsize_arg,self.bsize_result)
        for i,rc in enumerate(range(ttopo.kcells[dim-k][0], ttopo.kcells[dim-k][1] - ttopo.nbkcells[dim-k])):
            cc = stmapping.dinmk_to_pk(rc)
            for i in range(self.bsize_result):
                vals[i,i] = tgeom.get_entity_size(dim-k, rc - ttopo.kcells_off[dim-k]) / sgeom.get_entity_size(k, cc - stopo.kcells_off[k])
            mat.setValuesBlocked(
            rows=rc - ttopo.kcells_off[dim-k],
            cols=cc - stopo.kcells_off[k],
            values=vals)

#maybe eventually add C code to set values, and also do various apply operations?
#Here we assume bundle and it's dual have the same "basis", and thus the block of values is actually diagonal!
class VoronoiStarTtoS(UnaryOperator):
    def __init__(self, k, meshes, sbundle, tbundle):
        dim = stopo.tdim
        self.meshes = meshes
        self.narg = meshes.ttopo.nkcells[k]
        self.nresult = meshes.stopo.nkcells[dim-k]
        self.bsize_result = sbundle.size()
        self.bsize_arg = tbundle.size()
        self.k = k
        self.name = 'H'+str(k)+'-'+str(tbundle.name)+'-'+str(ttopo.name)

    def _assemble_petsc_mat(self, mat):
        stopo, ttopo, sgeom, tgeom, stmapping = self.meshes.stopo, self.meshes.ttopo, self.meshes.sgeom, self.meshes.tgeom, self.meshes.stmapping
        dim = stopo.tdim

        vals = np.zeros(self.bsize_arg,self.bsize_result)
        for i,rc in enumerate(range(stopo.kcells[dim-k][0], stopo.kcells[dim-k][1])):
            cc = stmapping.pk_to_dinmk(rc)
            for i in range(self.bsize_result):
                vals[i,i] = sgeom.get_entity_size(dim-k, rc - stopo.kcells_off[dim-k]) / tgeom.get_entity_size(k, cc - ttopo.kcells_off[k])
            mat.setValuesBlocked(
            rows=rc - stopo.kcells_off[dim-k],
            cols=cc - ttopo.kcells_off[k],
            values=vals)
