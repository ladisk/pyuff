The ``UFF`` class
=================

A :class:`~pyuff.UFF` object gives access to one universal file: it indexes the data-sets of
the file, reads them into dictionaries and writes dictionaries to the file.

.. autoclass:: pyuff.UFF
    :members: read_sets, write_sets, get_n_sets, get_set_types, get_set_formats,
        get_supported_sets, get_file_name, file_exists, get_status, refresh
