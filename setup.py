from setuptools import setup, find_packages

setup(
    name='stratpd',
    version='0.1',
    url='https://github.com/parrt/stratpd',
    license='MIT',
    py_modules=['stratpd'],
    packages=find_packages(),
    python_requires='>=3.6',
    author='Terence Parr',
    author_email='parrt@antlr.org',
    install_requires=['sklearn','pandas','numpy','matplotlib'],
    description='Model-independent net-effect (MINE) plots in Python 3',
    keywords='model-independent net-effect plots, visualization, partial dependence plots, partial derivative plots, ICE plots',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Intended Audience :: Developers']
)
