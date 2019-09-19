from setuptools import setup, find_packages

setup(
    name='stratx',
    version='0.1.3',
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
                 'Intended Audience :: Developers']
)
