pyuff
=====

**Read and write UFF/UNV (Universal File Format) files in Python.**

The Universal File Format is a text (and partly binary) file format for test and analysis data
in structural dynamics: geometry, measured functions such as frequency response functions
(FRFs), mode shapes and analysis results. Files with the extensions ``.uff`` and ``.unv`` are
exported and imported by many experimental modal analysis, signal analysis and finite element
programs.

pyuff reads a file into a list of Python dictionaries, one dictionary for each data-set, and
writes such dictionaries back to a file. The numerical data are NumPy arrays.

.. code-block:: python

    import pyuff

    uff_file = pyuff.UFF('data/beam.uff')
    uff_file.get_set_types()   # array([ 151,  164, 2420, 2411,   58,   58,   58])

    frf = uff_file.read_sets(4)   # the first data-set 58: a dictionary
    frf['x']                      # frequency
    frf['data']                   # complex FRF

Installation
------------

.. code-block:: bash

    pip install pyuff

pyuff requires Python 3.13 or newer and NumPy.

Contents
--------

.. toctree::
    :maxdepth: 2

    showcase
    conventions
    the_uff_class
    Supported_datasets

- :doc:`showcase`: reading a file, plotting FRFs and writing measured data, a geometry and
  mode shapes.
- :doc:`conventions`: how a data-set is stored in a file and in a dictionary, required and
  optional fields, and the most common codes.
- :doc:`the_uff_class` and :doc:`Supported_datasets`: the reference of every method and
  of every field of the supported data-sets.

The specification of the data-sets is at https://www.ceas3.uc.edu/sdrluff/. The source code
and the issue tracker are at https://github.com/ladisk/pyuff.

History and license
-------------------

pyuff was first written by Primož Čermelj in 2007–2008, was adopted for Python 3 and
developed as part of the OpenModal project (2014–2017), and is maintained by Janko Slavič and
the `Laboratory for Dynamics of Machines and Structures (LADISK) <https://www.ladisk.si>`_,
University of Ljubljana. It is released under the MIT license.
