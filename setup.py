# -*- coding: utf-8 -*-
"""Setup file for the document-feature-selection project.
"""

__author__ = 'kensuke-mi'
__version__ = '0.4'

import codecs
import os.path
import re
import sys
import install_dependencies
from setuptools import setup, find_packages

python_version = sys.version_info

if python_version >= (3, 0, 0):
    install_requires = ['six', 'setuptools>=1.0', 'joblib', 'scipy', 'nltk']

# avoid a from document-feature-selection import __version__ as version (that compiles document-feature-selection.__init__ and is not compatible with bdist_deb)
for line in codecs.open(os.path.join('document_feature_selection', '__init__.py'), 'r', encoding='utf-8'):
    matcher = re.match(r"""^__version__\s*=\s*['"](.*)['"]\s*$""", line)

# get README content from README.md file
with codecs.open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as fd:
    long_description = fd.read()



entry_points = {u'console_scripts': [u'document_feature_selection = document_feature_selection.cli:main']}


setup(
    name='document-feature-selection',
    version=__version__,
    description='No description yet.',
    long_description=long_description,
    author=__author__,
    author_email='kensuke.mit@gmail.com',
    license='CeCILL-B',
    url='',
    entry_points=entry_points,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    test_suite='document-feature-selection.tests',
    install_requires=install_requires,
    setup_requires=['six', 'setuptools>=1.0'],
    classifiers=[],
)
