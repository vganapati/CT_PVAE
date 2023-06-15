from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy


ext_modules = [
    Extension("_warps_cy", ["_warps_cy.pyx"], include_dirs=["path/to/fused_numerics.pxd", 
                                                            "path/to/interpolation.pxd"])
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)