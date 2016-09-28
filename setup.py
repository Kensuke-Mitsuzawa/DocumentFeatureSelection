# -*- coding: utf-8 -*-
"""Setup file for the document-feature-selection project.
"""

__author__ = 'kensuke-mi'
__version__ = '1.3'

import sys
from setuptools import setup, find_packages

# Flags to compile Cython code or use already compiled code
# --------------------------------------------------------------------------------------------------------
try:
    from Cython.Build import cythonize
    from distutils.extension import Extension
    from Cython.Distutils import build_ext
    import numpy
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }
ext_modules = [ ]
if use_cython:
    ext_modules += [
        Extension("DocumentFeatureSelection.pmi.pmi_cython", [ "DocumentFeatureSelection/pmi/pmi_cython.pyx" ]),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("DocumentFeatureSelection.pmi.pmi_cython", [ "DocumentFeatureSelection/pmi/pmi_cython.c" ]),
    ]
# --------------------------------------------------------------------------------------------------------


python_version = sys.version_info

if python_version >= (3, 0, 0):
    install_requires = ['six', 'setuptools>=1.0', 'joblib',
                        'scipy', 'nltk', 'scikit-learn', 'numpy', 'pypandoc', 'cython', 'scikit-learn']
else:
    raise Exception('This package does NOT support Python2.x')

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()


description = 'Various methods of feature selection from Text Data'

classifiers = [
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Natural Language :: Japanese",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.5"
        ]

setup(
    name='DocumentFeatureSelection',
    version=__version__,
    description=description,
    long_description=long_description,
    author=__author__,
    author_email='kensuke.mit@gmail.com',
    license='CeCILL-B',
    url='https://github.com/Kensuke-Mitsuzawa/DocumentFeatureSelection',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    test_suite='tests.all_tests.suite',
    install_requires=install_requires,
    setup_requires=['six', 'setuptools>=1.0'],
    classifiers=[],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    include_dirs = [numpy.get_include()]
)