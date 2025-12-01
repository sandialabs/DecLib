from DecLib import PETSc


#output the mesh in hdf5 format, also all the variables, etc.
class Output():
    def __init__(self, fname, dyn, diag, stats):
        self.dyn = dyn
        self.diag = diag
        self.stats = stats
        self.vwr = PETSc.ViewerHDF5().create(fname+'.h5', mode=PETSc.Viewer.Mode.WRITE)
        self.output_mesh()
        self.vwr.pushTimestepping()

    def output(self, ind):
        self.vwr.setTimestep(ind)

        self.vwr.pushGroup('prog')
        for name,v in self.dyn.prog_vars.items():
            v.petsc_vec.view(self.vwr)
        self.vwr.popGroup()

        self.vwr.pushGroup('diag')
        for name,v in self.diag.vars.items():
            v.petsc_vec.view(self.vwr)
        self.vwr.popGroup()

        self.vwr.setTimestep(0)
        self.vwr.pushGroup('stats')
        for name,v in self.stats.stats.items():
            v.petsc_vec.view(self.vwr)
        self.vwr.popGroup()

    def output_const(self):
        self.vwr.setTimestep(0)
        self.vwr.pushGroup('const')
        for name,v in self.dyn.const_vars.items():
            v.petsc_vec.view(self.vwr)
        self.vwr.popGroup()

#DMPLEX OUTPUT IS BROKEN- WHY?
    def output_mesh(self):

        self.vwr.pushGroup('mesh/primal')
        self.dyn.meshes.ptopo.petscmesh.setName('topology')
        #dyn.ptopo.petscmesh.view(self.vwr)
        for k in range(self.dyn.meshes.ptopo.dim+1):
            self.dyn.meshes.pgeom.entitysizes[k].setName('entity_size_'+str(k))
            self.dyn.meshes.pgeom.entitysizes[k].view(self.vwr)
        self.dyn.meshes.pgeom.vertexlocs.setName('vertex_locs')
        self.dyn.meshes.pgeom.vertexlocs.view(self.vwr)
        self.vwr.popGroup()

        self.vwr.pushGroup('mesh/dual')
        self.dyn.meshes.dtopo.petscmesh.setName('topology')
        #dyn.dtopo.petscmesh.view(self.vwr)
        for k in range(self.dyn.meshes.dtopo.dim+1):
            self.dyn.meshes.dgeom.entitysizes[k].setName('entity_size_'+str(k))
            self.dyn.meshes.dgeom.entitysizes[k].view(self.vwr)
        self.dyn.meshes.dgeom.vertexlocs.setName('vertex_locs')
        self.dyn.meshes.dgeom.vertexlocs.view(self.vwr)
        self.vwr.popGroup()

#ADD ORIENTATION + GEOMETRY OUTPUT...
