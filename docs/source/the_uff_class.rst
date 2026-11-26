The ``UFF`` class
=================

A :class:`~pyuff.UFF` object gives access to one universal file: it indexes the data-sets of
the file, reads them into dictionaries and writes dictionaries to the file.

.. autoclass:: pyuff.UFF
    :members: read_datasets, write_datasets, get_n_datasets, get_dataset_types, get_dataset_formats,
        get_supported_datasets, get_file_name, file_exists, get_status, refresh
