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
      version='1.0.0',
      author='Primož Čermelj',
      author_email='primoz.cermelj@gmail.com',
      url='https://github.com/openmodal/pyuff',
      py_modules=['pyuff'],
      long_description=desc,
      requires=['numpy']
      )