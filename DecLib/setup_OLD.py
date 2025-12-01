from __future__ import absolute_import, print_function, division

from distutils.core import setup
import sys

if sys.version_info < (3, 5):
    print("Python 3.5 or higher required, please upgrade.")
    sys.exit(1)


setup(name="DecLib",
      version="0.1",
      description="Discrete Exterior Calculus library using PETSc",
      author="Christopher Eldred, Nathan Roberts, Werner Bauer",
      author_email="celdred@sandia.gov",
      url="",
      license="",
      packages=["declib"])
