|pytest| |documentation| |pypi| |python|

pyuff
=====

Read and write UFF/UNV (Universal File Format) files in Python.

The Universal File Format is a text (and partly binary) file format for test and analysis data in
structural dynamics: geometry, measured functions such as frequency response functions (FRFs),
mode shapes and analysis results. Files with the extensions ``.uff`` and ``.unv`` are exported and
imported by many experimental modal analysis, signal analysis and finite element programs.

pyuff reads a file into a list of Python dictionaries, one dictionary for each data-set, and writes
such dictionaries back to a file. The numerical data are NumPy arrays.

Features
--------

- Read and write 15 data-set types, from geometry and units to measured functions and analysis
  results (see `Supported data-sets`_).
- Data-set 58 in ASCII and in binary form (58b), in single or double precision.
- Read all data-sets of a file, selected data-sets, or only their headers.
- ``prepare_*`` functions that check the values and build the dictionary of a data-set for writing.
- Text longer than its field, NaN and infinite values are handled with a warning when writing.

Installation
------------

.. code:: bash

    pip install pyuff

pyuff requires Python 3.13 or newer and NumPy.

Reading a file
--------------

Open a file and check which data-sets it contains:

.. code:: python

    import pyuff

    uff_file = pyuff.UFF('data/beam.uff')
    uff_file.get_n_sets()      # 7
    uff_file.get_set_types()   # array([ 151,  164, 2420, 2411,   58,   58,   58])
    uff_file.get_set_formats() # 0 for ASCII, 1 for binary data-sets

Read all data-sets (a list of dictionaries), or only some of them:

.. code:: python

    data = uff_file.read_sets()          # all data-sets
    header = uff_file.read_sets(0)       # one data-set: a dictionary
    frfs = uff_file.read_sets([4, 5, 6]) # selected data-sets: a list

Each data-set is a dictionary; its ``type`` key gives the data-set number. For a function in
data-set 58, ``x`` holds the abscissa (e.g. frequency) and ``data`` the ordinate (e.g. a complex
FRF):

.. code:: python

    import numpy as np
    import matplotlib.pyplot as plt

    frf = data[4]
    print(frf['type'], frf['id1'], frf['rsp_node'], frf['rsp_dir'], frf['ref_node'], frf['ref_dir'])

    plt.semilogy(frf['x'], np.abs(frf['data']))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('FRF magnitude')
    plt.show()

Writing a file
--------------

Each ``prepare_*`` function (e.g. ``prepare_58`` for data-set 58) checks the given values and returns
the dictionary of a data-set. ``write_sets`` writes one dictionary or a list of them; ``mode='add'``
appends to the file and ``mode='overwrite'`` starts a new file.

Measured FRFs (data-set 58)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    import numpy as np
    import pyuff

    frequency = np.arange(0, 1001)
    datasets = []
    for i, file in enumerate(['data/meas_point_1.txt', 'data/meas_point_2.txt', 'data/meas_point_3.txt']):
        accelerance = np.genfromtxt(file, dtype=complex)
        datasets.append(pyuff.prepare_58(
            func_type=4,                  # frequency response function
            rsp_node=1, rsp_dir=1,        # response: node 1, +X
            ref_node=i + 1, ref_dir=1,    # reference (excitation): node i + 1, +X
            data=accelerance, x=frequency,
            id1='Accelerance', rsp_ent_name='beam', ref_ent_name='beam',
            abscissa_spacing=1,           # even spacing of the abscissa
            abscissa_spec_data_type=18,   # frequency
            ordinate_spec_data_type=12,   # acceleration
            orddenom_spec_data_type=13))  # excitation force

    uff_file = pyuff.UFF('measurement.uff')
    uff_file.write_sets(datasets, mode='overwrite')

Data-set 58 is written in ASCII with double precision by default. Use ``prepare_58(..., binary=1)``
for the binary form (58b), and ``write_sets(..., force_double=False)`` for single precision.

Geometry and mode shapes (data-sets 2411, 2412 and 55)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    import numpy as np
    import pyuff

    node_nums = np.array([1, 2, 3, 4])
    nodes = pyuff.prepare_2411(
        node_nums=node_nums, def_cs=np.zeros(4, int), disp_cs=np.zeros(4, int), color=np.ones(4, int),
        x=np.array([0., 1., 1., 0.]), y=np.array([0., 0., 1., 1.]), z=np.zeros(4))

    quad = pyuff.prepare_2412(element_nums=1, fe_descriptor=44, phys_table=1, mat_table=1, color=1,
                              num_nodes=4, nodes_nums=[1, 2, 3, 4])
    elements = {'type': 2412, 44: [quad]}

    mode_shape = pyuff.prepare_55(
        model_type=1, analysis_type=2,   # structural, normal mode
        data_ch=2, spec_data_type=8,     # 3 DOF translation, displacement
        load_case=1, mode_n=1, freq=12.5,
        node_nums=node_nums,
        r1=np.zeros(4), r2=np.zeros(4), r3=np.array([0.1, 0.2, 0.2, 0.1]))  # X, Y and Z

    pyuff.UFF('model.uff').write_sets([nodes, elements, mode_shape], mode='overwrite')

Data-set 55 is written with as many values per node as response arrays are given: ``r1`` for scalar
data, ``r1`` to ``r3`` for translations, ``r1`` to ``r6`` for translations and rotations. Complex
arrays write complex data.

Supported data-sets
-------------------

=========  ====================================  =============================
Data-set   Name                                  Prepare function
=========  ====================================  =============================
15         Nodes                                 ``prepare_15``
18         Coordinate systems                    ``prepare_18``
55         Data at nodes (e.g. mode shapes)      ``prepare_55``
58, 58b    Function at nodal DOF (e.g. FRFs)     ``prepare_58``
82         Tracelines                            ``prepare_82``
151        Header                                ``prepare_151``
164        Units                                 ``prepare_164``
1858       Data-set 58 qualifiers                ``prepare_1858``
2400       Model header                          ``prepare_2400``
2411       Nodes, double precision               ``prepare_2411``
2412       Elements                              ``prepare_2412``
2414       Analysis data                         ``prepare_2414``
2420       Coordinate systems                    ``prepare_2420``
2429       Permanent groups                      ``prepare_2429``
2467       Permanent groups                      ``prepare_2467``
=========  ====================================  =============================

Other data-sets in a file are skipped when reading: ``read_sets`` returns only their ``type``.
The fields of each data-set are described in the docstrings of the ``prepare_*`` functions and in
the `documentation <https://pyuff.readthedocs.io/en/latest/Supported_datasets.html>`_; the
description of a data-set from the specification is printed by e.g.
``pyuff.datasets.dataset_58.get_structure_58()``.

Notes on writing
----------------

- Text longer than its fixed-width field is truncated, with a warning.
- NaN and infinite values in ``data`` (e.g. of data-set 58) are written as 0 and as the largest
  finite float, with a warning.
- Data-set 58 is written in double precision by default, also if it was read in single precision;
  use ``force_double=False`` for single precision.

Documentation
-------------

The documentation, with a showcase and the description of every data-set, is at
https://pyuff.readthedocs.io. The specification of the data-sets is at
https://www.ceas3.uc.edu/sdrluff/.

Development
-----------

.. code:: bash

    git clone https://github.com/ladisk/pyuff.git
    cd pyuff
    pip install -e .[dev]
    pytest

Bug reports and pull requests are welcome at https://github.com/ladisk/pyuff/issues.

History and license
-------------------

pyuff was first written by Primož Čermelj in 2007–2008, was adopted for Python 3 and developed as
part of the OpenModal project (2014–2017), and is maintained by Janko Slavič and the
`Laboratory for Dynamics of Machines and Structures (LADISK) <https://www.ladisk.si>`_,
University of Ljubljana. It is released under the MIT license.

.. |pytest| image:: https://github.com/ladisk/pyuff/actions/workflows/python-package.yml/badge.svg
    :target: https://github.com/ladisk/pyuff/actions
.. |documentation| image:: https://readthedocs.org/projects/pyuff/badge/?version=latest
    :target: https://pyuff.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. |pypi| image:: https://img.shields.io/pypi/v/pyuff.svg
    :target: https://pypi.org/project/pyuff/
    :alt: PyPI version
.. |python| image:: https://img.shields.io/pypi/pyversions/pyuff.svg
    :target: https://pypi.org/project/pyuff/
    :alt: Python versions
