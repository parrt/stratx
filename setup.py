"""
MIT License

Copyright (c) 2019 Terence Parr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from setuptools import setup, find_packages, Extension
# from Cython.Build import cythonize
# from Cython.Distutils import build_ext
from glob import glob
# import numpy as np

# Run with "export CC=gcc-8; python setup.py install" if using cython

# extensions = [
#     Extension(
#         'stratx.cy_partdep',
#         glob('stratx/*.pyx'),
#         extra_compile_args=['-O3','-fopenmp'], # brew install libomp
#         extra_link_args=['-fopenmp','-Wl,-rpath,/usr/local/Cellar/gcc/8.2.0/lib/gcc/8'])
# ]

setup(
    name='stratx',
    version='0.5',
    url='https://github.com/parrt/stratx',
    license='MIT',
    py_modules=['stratx', 'stratx.partdep', 'stratx.featimp', 'stratx.plot'],
    packages=find_packages(),
    python_requires='>=3.6',
    author='Terence Parr',
    author_email='parrt@antlr.org',
    install_requires=['scikit-learn','pandas','numpy','matplotlib','scipy','numba','colour'],
    description='Model-independent partial dependence plots in Python 3 that works even for codependent variables',
    keywords='model-independent net-effect plots, visualization, partial dependence plots, partial derivative plots, ICE plots, feature importance',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Intended Audience :: Developers'],

    # ext_modules=cythonize(extensions, annotate=True),
    # cmdclass={'build_ext': build_ext},
    # zip_safe=False,
    # include_dirs=[np.get_include()]
)
