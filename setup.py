# -*- coding: utf-8 -*-
"""Setup file for the document-feature-selection project.
"""

__author__ = 'kensuke-mi'
__version__ = '1.5'

import sys
import subprocess
from setuptools import setup, find_packages
from distutils.extension import Extension
python_version = sys.version_info
print('python version {}'.format(python_version))


# --------------------------------------------------------------------------------------------------------
# Flags to compile Cython code or use already compiled code
try:
    import Cython
except ImportError:
    subprocess.check_call(["python", '-m', 'pip', 'install', 'cython'])
    import Cython

if sys.version_info >= (3, 7):
    # if python >= 3.7, Cython must regenerate C++ code again.
    import os
    if os.path.exists('DocumentFeatureSelection/pmi/pmi_cython.c'):
        os.remove('DocumentFeatureSelection/pmi/pmi_cython.c')
    if os.path.exists('DocumentFeatureSelection/bns/bns_cython.c'):
        os.remove('DocumentFeatureSelection/bns/bns_cython.c')
    if os.path.exists('DocumentFeatureSelection/soa/soa_cython.c'):
        os.remove('DocumentFeatureSelection/soa/soa_cython.c')
    # if python >= 3.7, typing should be installed again.
    subprocess.check_call(["python", '-m', 'pip', 'install', 'typing'])

cmdclass = {}
ext_modules = []
from Cython.Distutils import build_ext

ext_modules += [
    Extension("DocumentFeatureSelection.pmi.pmi_cython", [ "DocumentFeatureSelection/pmi/pmi_cython.pyx" ],),
    Extension("DocumentFeatureSelection.soa.soa_cython", [ "DocumentFeatureSelection/soa/soa_cython.pyx" ],),
    Extension("DocumentFeatureSelection.bns.bns_cython", [ "DocumentFeatureSelection/bns/bns_cython.pyx" ],)
]
cmdclass.update({'build_ext': build_ext})


# --------------------------------------------------------------------------------------------------------
# try to install numpy automatically because sklearn requires the status where numpy is already installed
try:
    import numpy
except ImportError:
    use_numpy_include_dirs = False
    try:
        subprocess.check_call(["python", '-m', 'pip', 'install', 'numpy'])
        import numpy
    except Exception as e:
        raise Exception(e.__str__() + 'We failed to install numpy automatically. \
        Try installing numpy manually or Try anaconda distribution.')

# --------------------------------------------------------------------------------------------------------
# try to install scipy automatically because sklearn requires the status where scipy is already installed
try:
    import scipy
except ImportError:
    try:
        subprocess.check_call(["python", '-m', 'pip', 'install', 'scipy'])
        import scipy
    except Exception as e:
        raise Exception(e.__str__() + 'We failed to install scipy automatically. \
        Try installing scipy manually or Try anaconda distribution.')
# --------------------------------------------------------------------------------------------------------


install_requires = ['six', 'setuptools>=1.0', 'joblib', 'numpy',
                    'scipy', 'nltk', 'scikit-learn', 'pypandoc', 'cython', 'sqlitedict', 'nose',
                    'typing']

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
    tests_require=install_requires,
    setup_requires=['six', 'setuptools>=1.0', 'pip', 'typing', 'cython'],
    classifiers=classifiers,
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()]
)
