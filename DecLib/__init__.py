
#Ensure petsc is initialised by us before anything else gets in there.

import DecLib.petsc_shim as petsc
del petsc
del petsc_shim

from DecLib.common import *  # noqa: F401
del common

from DecLib.forms import *  # noqa: F401
del forms

from DecLib.meshes import *  # noqa: F401
del meshes

from DecLib.operators import *  # noqa: F401
del operators
