# -*- coding: utf-8 -*-
"""
Package gathering all unitary tests for document-feature-selection.
Module names must start with `test_` to be taken into account.

You should consider to install :mod:`Distribute` to run all tests with::

    $ python setup.py test

"""
from __future__ import unicode_literals
__author__ = 'kensuke-mi'
import unittest

if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
