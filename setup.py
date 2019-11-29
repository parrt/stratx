from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from glob import glob
import numpy as np

extensions = [
    Extension(
        'stratx.cy_partdep',
        glob('stratx/*.pyx'),
        extra_compile_args=['-O3'])
]

setup(
    name='stratx',
    version='0.2',
    url='https://github.com/parrt/stratx',
    license='MIT',
    py_modules=['stratx'],
    packages=find_packages(),
    python_requires='>=3.6',
    author='Terence Parr',
    author_email='parrt@antlr.org',
    install_requires=['sklearn','pandas','numpy','matplotlib','scipy','dtreeviz'],
    description='Model-independent partial dependence plots in Python 3 that works even for codependent variables',
    keywords='model-independent net-effect plots, visualization, partial dependence plots, partial derivative plots, ICE plots, feature importance',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Intended Audience :: Developers'],

    # ext_modules=cythonize("stratx/cy_partdep.pyx", annotate=True),

    # ext_modules=cythonize(extensions, annotate=True),
    # cmdclass={'build_ext': build_ext},
    # zip_safe=False,
    # include_dirs=[np.get_include()]
)
