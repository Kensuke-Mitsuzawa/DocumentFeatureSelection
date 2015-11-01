# -*- coding: utf-8 -*-
from __future__ import unicode_literals
__author__ = 'kensuke-mi'
__all__ = ['sample_function', 'sample_translation', ]


# write your actual code here.

def sample_function(first, second=4):
    """This is a sample function to demonstrate doctests
    of :mod:`document-feature-selection.code` and docs.
    It only return the sum of its two arguments.

    Args:
      :param first: (:class:`int`): first value to add
      :param second:  (:class:`int`): second value to add, 4 by default

    Returns:
      * :class:`int`: the sum of `first` and `second`.

    >>> sample_function(6, second=3)
    9
    >>> sample_function(6)
    10
    """
    return first + second


def sample_translation():
    """
    Simply return a constant string, that should be translated.
    """
    return _('This message should be translated')




if __name__ == '__main__':
    import doctest
    doctest.testmod()
