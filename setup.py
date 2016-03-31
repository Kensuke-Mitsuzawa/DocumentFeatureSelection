# -*- coding: utf-8 -*-
"""Setup file for the document-feature-selection project.
"""

import codecs
import os.path
import re
import sys
import install_dependencies
from setuptools import setup, find_packages


# avoid a from document-feature-selection import __version__ as version (that compiles document-feature-selection.__init__ and is not compatible with bdist_deb)
version = None
for line in codecs.open(os.path.join('document_feature_selection', '__init__.py'), 'r', encoding='utf-8'):
    matcher = re.match(r"""^__version__\s*=\s*['"](.*)['"]\s*$""", line)
    version = version or matcher and matcher.group(1)

# get README content from README.md file
with codecs.open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as fd:
    long_description = fd.read()

entry_points = {u'console_scripts': [u'document_feature_selection = document_feature_selection.cli:main']}


install_requires = ['six', 'setuptools>=1.0',
                    'nltk==3.0.1', 'scikit-learn==0.15.2', 'scipy', 'numpy']

setup(
    name='document-feature-selection',
    version=version,
    description='No description yet.',
    long_description=long_description,
    author='kensuke-mi',
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
