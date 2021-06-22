Supported datasets
==================

More info on specific datasets can be obtained `here <https://www.ceas3.uc.edu/sdrluff>`_.


Dict function showcase
-----------------------

Dict functions are used for preparing the dictionaries for writing UFF files.

Parameters labeled ``optional`` designate optional fields, that are not needed when writing to a file, as they have a default value.
Parameters labeled ``ignored`` designate fields that are not needed at all, as these fields are defined automatically.

Some fields are data dependent, meaning they are only used/available for some specific data-type. For example ``modal_damp_vis`` in dataset 55.

Parameter ``return_full_dict`` determines if all keys or only specified arguments are returned.

.. code:: python
    
    import pyuff

    pyuff.dict_151(model_name='NewModel', date_db_created='17-Jun-21', time_db_created='12:49:33', version_db1= 0, version_db2= 0, file_type= 0)








Dataset 15
----------

.. autofunction:: pyuff.dict_15


Dataset 55
----------

.. autofunction:: pyuff.dict_55


Dataset 58<b>
-------------

.. autofunction:: pyuff.dict_58


Dataset 82
----------

.. autofunction:: pyuff.dict_82


Dataset 151
-----------

.. autofunction:: pyuff.dict_151

Dataset 164
-----------

.. autofunction:: pyuff.dict_164


Dataset 2411
-------------

.. autofunction:: pyuff.dict_2411


Dataset 2412
-------------

.. autofunction:: pyuff.dict_2412


Dataset 2414
-------------

.. autofunction:: pyuff.dict_2414


Dataset 2420
-------------

.. autofunction:: pyuff.dict_2420

















