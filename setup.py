from setuptools import setup

setup(
    name='mine',
    version='0.1',
    url='https://github.com/parrt/mine',
    license='MIT',
    py_modules=['mine'],
    python_requires='>=3.6',
    author='Terence Parr',
    author_email='parrt@antlr.org',
    install_requires=['sklearn','pandas','numpy','matplotlib'],
    description='Model-independent net-effect in Python 3',
    keywords='model-independent net-effect plots, visualization, partial dependence plots, partial derivative plots, ICE plots',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Intended Audience :: Developers']
)
