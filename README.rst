|pytest| |documentation| 

pyuff
=====

Universal File Format read and write
------------------------------------
This module defines an UFF class to manipulate with the UFF (Universal File Format) files.

Read from and write of data-set types **15, 55, 58, 58b, 82, 151, 164, 2411, 2412, 2414, 2420, 2429, 2467** are supported.

Check out the `documentation <https://pyuff.readthedocs.io/en/latest/index.html>`_.

To install the package, run:

.. code:: python

   pip install pyuff

Showcase
---------

To analyse UFF file we first load the uff module and example file:

.. code:: python

    import pyuff
    uff_file = pyuff.UFF('data/beam.uff')

To check which datasets are written in the file use:

.. code:: python

    uff_file.get_set_types()

Reading from the UFF file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To load all datasets from the UFF file to data object use:

.. code:: python

    data = uff_file.read_sets()


The first dataset 58 contains following keys:

.. code:: python

    data[4].keys()

Most important keys are ``x``: x-axis and ``data``: y-axis that define the stored response:

.. code:: python

    plt.semilogy(data[4]['x'], np.abs(data[4]['data']))
    plt.xlabel('Frequency  [Hz]')
    plt.ylabel('FRF Magnitude [dB m/N]')
    plt.xlim([0,1000])
    plt.show()


Writing measurement data to UFF file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Loading the accelerance data:

.. code:: python

    measurement_point_1 = np.genfromtxt('data/meas_point_1.txt', dtype=complex)
    measurement_point_2 = np.genfromtxt('data/meas_point_2.txt', dtype=complex)
    measurement_point_3 = np.genfromtxt('data/meas_point_3.txt', dtype=complex)

.. code:: python

    measurement_point_1[0] = np.nan*(1+1.j)

.. code:: python

    measurement = [measurement_point_1, measurement_point_2, measurement_point_3]

Creating the UFF file where we add dataset 58 for measurement consisting of the dictionary-like keys containing the measurement data and the information about the measurement:

.. code:: python

    for i in range(3):
        print('Adding point {:}'.format(i + 1))
        response_node = 1
        response_direction = 1
        reference_node = i + 1
        reference_direction = 1
        acceleration_complex = measurement[i]
        frequency = np.arange(0, 1001)
        name = 'TestCase'
        data = {'type':58,
                'func_type': 4,
                'rsp_node': response_node,
                'rsp_dir': response_direction,
                'ref_dir': reference_direction,
                'ref_node': reference_node,
                'data': acceleration_complex,
                'x': frequency,
                'id1': 'id1',
                'rsp_ent_name': name,
                'ref_ent_name': name,
                'abscissa_spacing':1,
                'abscissa_spec_data_type':18,
                'ordinate_spec_data_type':12,
                'orddenom_spec_data_type':13}
        uffwrite = pyuff.UFF('./data/measurement.uff')
        uffwrite.write_set(data,'add')

Or we can use support function ``prepare_58`` to prepare the dictionary for creating the UFF file. Functions for other datasets can be found in `supported datasets <https://pyuff.readthedocs.io/en/latest/Supported_datasets.html>`_.

.. code:: python

    for i in range(3):
    print('Adding point {:}'.format(i + 1))
    response_node = 1
    response_direction = 1
    reference_node = i + 1
    reference_direction = 1
    acceleration_complex = measurement[i]
    frequency = np.arange(0, 1001)
    name = 'TestCase'
    pyuff.prepare_58(func_type=4,
                rsp_node=response_node,
                rsp_dir=response_direction,
                ref_dir=reference_direction,
                ref_node=reference_node,
                data=acceleration_complex,
                x=frequency,
                id1='id1',
                rsp_ent_name=name,
                ref_ent_name=name,
                abscissa_spacing=1,
                abscissa_spec_data_type=18,
                ordinate_spec_data_type=12,
                orddenom_spec_data_type=13)

.. |pytest| image:: https://github.com/ladisk/pyuff/actions/workflows/python-package.yml/badge.svg
    :target: https://github.com/ladisk/pyuff/actions
.. |documentation| image:: https://readthedocs.org/projects/pyuff/badge/?version=latest
    :target: https://pyuff.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
