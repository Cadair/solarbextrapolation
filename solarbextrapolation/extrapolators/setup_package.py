# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import

import os
from distutils.extension import Extension

ROOT = os.path.relpath(os.path.dirname(__file__))


def get_extensions():
    sources = ["potential_field_extrapolator_cython.pyx"]
    include_dirs = ['numpy']

    exts = [
        Extension(name='solarbextrapolation.extrapolators.' + os.path.splitext(source)[0],
                  sources=[os.path.join(ROOT, source)],
                  include_dirs=include_dirs)
        for source in sources
    ]

    return exts


def requires_2to3():
    return False
