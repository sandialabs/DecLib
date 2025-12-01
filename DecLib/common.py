from DecLib.petsc_shim import PETSc

stdoutinfoview = PETSc.Viewer().STDOUT()
stdoutinfoview.pushFormat(PETSc.Viewer.Format.ASCII_INFO_DETAIL)

stdoutindexview = PETSc.Viewer().STDERR()
stdoutindexview.pushFormat(PETSc.Viewer.Format.ASCII_COMMON)

ADD_MODE = 0
INSERT_MODE = 1
