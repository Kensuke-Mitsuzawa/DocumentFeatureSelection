# -*- coding: utf-8 -*-
"""Setup file for the document-feature-selection project.
"""

__author__ = 'kensuke-mi'
__version__ = '1.0'

import sys
from setuptools import setup, find_packages

python_version = sys.version_info

if python_version >= (3, 0, 0):
    install_requires = ['six', 'setuptools>=1.0', 'joblib',
                        'scipy', 'nltk', 'scikit-learn', 'numpy', 'pypandoc']


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
)
