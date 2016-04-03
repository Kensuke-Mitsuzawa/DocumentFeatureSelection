#! -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
import sys
python_version = sys.version_info

if python_version > (3, 0, 0):
    from DocumentFeatureSelection.common.data_converter_python3 import DataConverter, DataCsrMatrix
else:
    raise SystemError('Not Implemented yet')