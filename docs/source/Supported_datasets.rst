Supported data-sets
===================

Each supported data-set has a ``prepare_*`` function that builds the dictionary of a data-set
for writing, checks the given values and documents every field: what it means, its codes,
whether it is required and its default value. The general conventions are explained in
:doc:`conventions`; the specification of the data-sets is at https://www.ceas3.uc.edu/sdrluff/.

.. list-table::
    :header-rows: 1
    :widths: 12 26 62

    * - Data-set
      - Name
      - Content
    * - :ref:`15 <dataset-15>`
      - Nodes
      - Node coordinates in single precision; 2411 is used more often.
    * - :ref:`18 <dataset-18>`
      - Coordinate systems
      - Local coordinate systems (older form of 2420).
    * - :ref:`55 <dataset-55>`
      - Data at nodes
      - Values at nodes, e.g. mode shapes or operating deflection shapes.
    * - :ref:`58, 58b <dataset-58>`
      - Function at nodal DOF
      - Measured or computed functions, e.g. FRFs, time histories, spectra.
    * - :ref:`82 <dataset-82>`
      - Tracelines
      - Lines between nodes, for displaying a wireframe model.
    * - :ref:`151 <dataset-151>`
      - Header
      - Model name, description and the programs that wrote the file.
    * - :ref:`164 <dataset-164>`
      - Units
      - Unit system of the file.
    * - :ref:`1858 <dataset-1858>`
      - Data-set 58 qualifiers
      - Additional information on data-set 58: e.g. windows, averaging, units.
    * - :ref:`2400 <dataset-2400>`
      - Model header
      - Model of a finite element program.
    * - :ref:`2411 <dataset-2411>`
      - Nodes (double precision)
      - Node coordinates.
    * - :ref:`2412 <dataset-2412>`
      - Elements
      - Finite elements and the nodes they connect.
    * - :ref:`2414 <dataset-2414>`
      - Analysis data
      - Results at nodes or elements, e.g. from a finite element analysis.
    * - :ref:`2420 <dataset-2420>`
      - Coordinate systems
      - Local coordinate systems.
    * - :ref:`2429 <dataset-2429>`
      - Permanent groups
      - Named groups of nodes, elements and other entities.
    * - :ref:`2467 <dataset-2467>`
      - Permanent groups
      - Named groups of entities (newer form of 2429).

.. _dataset-15:

Data-set 15: Nodes
------------------

.. autofunction:: pyuff.prepare_15

.. _dataset-18:

Data-set 18: Coordinate systems
-------------------------------

.. autofunction:: pyuff.prepare_18

.. _dataset-55:

Data-set 55: Data at nodes
--------------------------

.. autofunction:: pyuff.prepare_55

.. _dataset-58:

Data-set 58, 58b: Function at nodal DOF
---------------------------------------

.. autofunction:: pyuff.prepare_58

.. _dataset-82:

Data-set 82: Tracelines
-----------------------

.. autofunction:: pyuff.prepare_82

.. _dataset-151:

Data-set 151: Header
--------------------

.. autofunction:: pyuff.prepare_151

.. _dataset-164:

Data-set 164: Units
-------------------

.. autofunction:: pyuff.prepare_164

.. _dataset-1858:

Data-set 1858: Data-set 58 qualifiers
-------------------------------------

.. autofunction:: pyuff.prepare_1858

.. _dataset-2400:

Data-set 2400: Model header
---------------------------

.. autofunction:: pyuff.prepare_2400

.. _dataset-2411:

Data-set 2411: Nodes (double precision)
---------------------------------------

.. autofunction:: pyuff.prepare_2411

.. _dataset-2412:

Data-set 2412: Elements
-----------------------

.. autofunction:: pyuff.prepare_2412

.. _dataset-2414:

Data-set 2414: Analysis data
----------------------------

.. autofunction:: pyuff.prepare_2414

.. _dataset-2420:

Data-set 2420: Coordinate systems
---------------------------------

.. autofunction:: pyuff.prepare_2420

.. _dataset-2429:

Data-set 2429: Permanent groups
-------------------------------

.. autofunction:: pyuff.prepare_2429

.. autofunction:: pyuff.prepare_group

.. _dataset-2467:

Data-set 2467: Permanent groups
-------------------------------

.. autofunction:: pyuff.prepare_2467

.. autofunction:: pyuff.datasets.dataset_2467.prepare_group
