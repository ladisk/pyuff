#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2014-2017 Primož Čermelj, Matjaž Mršnik, Miha Pirnat, Janko Slavič, Blaž Starc (in alphabetic order)
# 
# This file is part of pyuff.
# 
# pyFRF is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# pyuff is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with pyuff.  If not, see <http://www.gnu.org/licenses/>.


desc = """\
UFF (Universal File Format) read/write.
=============
This module is part of the www.openmodal.com project and defines an UFF class to manipulate with the
UFF (Universal File Format) files

Read from and write of data-set types **151, 15, 55, 58, 58b, 82, 164** is supported.

For a showcase see: https://github.com/openmodal/pyuff/blob/master/pyuff%20Showcase.ipynb
"""

from setuptools import setup
setup(name='pyuff',
      version='1.20',
      author='Primož Čermelj, Janko Slavič',
      author_email='primoz.cermelj@gmail.com, janko.slavic@fs.uni-lj.si',
      description='UFF (Universal File Format) read/write.',
      url='https://github.com/openmodal/pyuff',
      py_modules=['pyuff'],
      long_description=desc,
      install_requires=['numpy']
      )