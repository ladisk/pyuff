Data-sets and fields
====================

This page explains how a data-set is stored in a universal file and in a pyuff dictionary,
and lists the codes that are used most often.

Files, data-sets, records and fields
------------------------------------

A universal file is a sequence of **data-sets**. Each data-set starts and ends with a line
with ``-1`` (right-aligned in six characters); the line after the start gives the data-set
**type** (e.g. ``58``, or ``58b`` for the
binary form of data-set 58). A file can hold data-sets of different types in any order: a
measurement file has, e.g., a header (151), units (164), nodes (2411) and many functions (58).

A data-set consists of **records** (one or more lines), and a record consists of **fields** of
fixed width. The specification of each data-set, at https://www.ceas3.uc.edu/sdrluff/, numbers
the records and the fields; the documentation of the ``prepare_*`` functions gives this
position for every field, e.g. "(record 6, field 7)". The specification text of a data-set is
also printed by e.g. ``pyuff.datasets.dataset_58.get_structure_58()``.

For example, a data-set 58 with a complex FRF at two frequency lines, as written by pyuff:

.. code-block:: text

        -1
        58
    Accelerance
    NONE
    NONE
    NONE
    NONE
        4         0    0         0       beam         1   1       beam         2   1
             6         2         1  0.00000e+00  1.00000e+00  0.00000e+00
            18    0    0    0 NONE                 Hz
            12    0    0    0 NONE                 m/s^2
            13    0    0    0 NONE                 N
             0    0    0    0 NONE                 NONE
       1.00000000000e-01   2.00000000000e-01   3.00000000000e-01  -1.00000000000e-01
        -1

Its records and the dictionary keys of their fields:

.. list-table::
    :header-rows: 1
    :widths: 12 30 58

    * - Record
      - Content
      - Keys
    * - 1–5
      - ID lines
      - ``id1`` (``'Accelerance'``), ``id2``, ..., ``id5``
    * - 6
      - DOF identification
      - ``func_type`` (4: FRF), ``func_id``, ``ver_num``, ``load_case_id``,
        ``rsp_ent_name``, ``rsp_node`` (1), ``rsp_dir`` (1: +X), ``ref_ent_name``,
        ``ref_node`` (2), ``ref_dir`` (1: +X)
    * - 7
      - data form
      - ``ord_data_type`` (6: complex, double precision), ``num_pts`` (2),
        ``abscissa_spacing`` (1: even), ``abscissa_min`` (0), ``abscissa_inc`` (1),
        ``z_axis_value``
    * - 8
      - abscissa
      - ``abscissa_spec_data_type`` (18: frequency), ``abscissa_len_unit_exp``,
        ``abscissa_force_unit_exp``, ``abscissa_temp_unit_exp``, ``abscissa_axis_lab``,
        ``abscissa_axis_units_lab`` (``'Hz'``)
    * - 9
      - ordinate (numerator)
      - ``ordinate_spec_data_type`` (12: acceleration), ... as for the abscissa
    * - 10
      - ordinate denominator
      - ``orddenom_spec_data_type`` (13: excitation force), ...
    * - 11
      - z-axis
      - ``z_axis_spec_data_type`` (0: not used), ...
    * - 12
      - data values
      - ``data``: the real and imaginary parts of each value; ``x`` is not stored, as the
        abscissa is even (``abscissa_min`` + i ``abscissa_inc``)

Dictionaries
------------

:meth:`pyuff.UFF.read_datasets` returns one dictionary for each data-set, and
:meth:`pyuff.UFF.write_datasets` writes such dictionaries:

- the key ``type`` is the data-set type (an integer, e.g. ``58``);
- the other keys are the fields of the data-set, named as the parameters of the
  ``prepare_*`` function of the data-set (see :doc:`Supported_datasets`);
- numerical data (e.g. ``x`` and ``data`` of data-set 58, the coordinates of nodes) are NumPy
  arrays.

A dictionary can be written directly, but the ``prepare_*`` functions are recommended: their
parameters are documented, and they check the given values before writing.

Required, optional and computed fields
--------------------------------------

The fields of a data-set are of three kinds; the documentation of each ``prepare_*``
function says which kind every field is.

Required
    Fields without a reasonable default, e.g. the response node and the function type of
    data-set 58. Writing a data-set without them raises an exception.

Optional
    Fields with a default value, marked *optional*, with the default given, e.g. the ID
    lines of data-set 58 (default ``'NONE'``). A field that is missing or ``None`` gets the
    default when writing.

Computed
    Fields that pyuff computes from the data when writing, e.g. the number of points of
    data-set 58 from ``data``. Their values in the dictionary are not used for writing, but
    they are set when reading, so the dictionary describes the file.

By default, a ``prepare_*`` function returns only the given fields. With
``return_full_dict=True`` it returns all fields of the data-set, with ``None`` for those not
given; this shows which fields exist, and the dictionary can be written as it is.

Common codes
------------

Many fields are integer codes from the specification. The codes used most often, for data-set
58 and others, are given here; the documentation of each ``prepare_*`` function lists all codes
of its fields.

.. rubric:: Directions (``rsp_dir``, ``ref_dir``)

.. list-table::
    :header-rows: 1
    :widths: 15 35 15 35

    * - Code
      - Direction
      - Code
      - Direction
    * - 0
      - scalar
      -
      -
    * - 1, −1
      - +X, −X translation
      - 4, −4
      - +X, −X rotation
    * - 2, −2
      - +Y, −Y translation
      - 5, −5
      - +Y, −Y rotation
    * - 3, −3
      - +Z, −Z translation
      - 6, −6
      - +Z, −Z rotation

.. rubric:: Function types (``func_type`` of data-set 58), the most common

.. list-table::
    :header-rows: 1
    :widths: 15 85

    * - Code
      - Function
    * - 0
      - general or unknown
    * - 1
      - time response
    * - 2
      - auto spectrum
    * - 3
      - cross spectrum
    * - 4
      - frequency response function
    * - 5
      - transmissibility
    * - 6
      - coherence
    * - 9
      - power spectral density (PSD)
    * - 12
      - spectrum
    * - 18
      - mode indicator function

.. rubric:: Specific data types (abscissa, ordinate, ordinate denominator and z-axis of data-set 58)

The specific data type gives the physical quantity of an axis. A function that is a ratio,
such as an FRF, has an ordinate numerator and an ordinate denominator: for an accelerance
:math:`H = a/F`, the ordinate (``ordinate_spec_data_type``) is acceleration (12) and the
ordinate denominator (``orddenom_spec_data_type``) is excitation force (13). For a function
that is not a ratio, the ordinate denominator is 0.

.. list-table::
    :header-rows: 1
    :widths: 15 35 15 35

    * - Code
      - Quantity
      - Code
      - Quantity
    * - 0
      - unknown
      - 12
      - acceleration
    * - 1
      - general
      - 13
      - excitation force
    * - 2
      - stress
      - 15
      - pressure
    * - 3
      - strain
      - 16
      - mass
    * - 5
      - temperature
      - 17
      - time
    * - 6
      - heat flux
      - 18
      - frequency
    * - 8
      - displacement
      - 19
      - rpm
    * - 9
      - reaction force
      - 20
      - order
    * - 11
      - velocity
      -
      -

.. rubric:: Units exponents

The fields ``*_len_unit_exp``, ``*_force_unit_exp`` and ``*_temp_unit_exp`` give the exponents
of length, force and temperature in the unit of an axis; e.g. for a pressure (force per
area) they are −2, 1 and 0. They are used only when the specific data type is general (1), or
when the direction is scalar; otherwise the unit follows from the specific data type and the
units of the file (data-set 164).

Precision of data-set 58
------------------------

The ordinate data type (``ord_data_type``) of data-set 58 gives whether the data are real or
complex, in single or double precision: 2 real single, 4 real double, 5 complex single, 6
complex double. pyuff writes in double precision by default
(:meth:`~pyuff.UFF.write_datasets` with ``force_double=True``), also if the data were read in single
precision; with ``force_double=False``, the precision is given by ``ord_data_type``.
