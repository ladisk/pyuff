import numpy as np

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none, _fit_string


def get_structure_2400(raw=False):
    """(source: https://www.ceas3.uc.edu/sdrluff/"""
    out = """
Universal Dataset Number: 2400

Name:   Model Header
-----------------------------------------------------------------------

Record 1:       FORMAT(I12,2I6,I12)
                Field 1      -- model UID
                Field 2      -- entity type
                Field 3      -- entity subtype
                Field 4      -- version number
Record 2:       FORMAT(40A2)
                Field 1      -- entity name
Record 3:       FORMAT(40A2)
                Field 1      -- part number
Record 4:       FORMAT(32I2)
                Field 1      -- status mask (32 values)
Record 5:       FORMAT(10A1,10A1,3I12)
                Field 1      -- date (DD-MMM-YY)
                Field 2      -- time (HH:MM:SS)
                Field 3      -- IDM version ID
                Field 4      -- IDM item ID
                Field 5      -- primary parent UID
Record 6:       FORMAT(I12)
                Field 1      -- optimization switches
                                =0  geometry OFF, analysis OFF
                                =1  geometry ON,  analysis OFF
                                =2  geometry OFF, analysis ON
                                =3  geometry ON,  analysis ON

-----------------------------------------------------------------
"""
    if raw:
        return out
    else:
        print(out)


def _write2400(fh, dset):
    """Writes dset data - data-set 2400 - to an open file fh."""
    try:
        dset = _opt_fields(dset, {
            'model_uid': 0,
            'entity_type': 0,
            'entity_subtype': 0,
            'version_number': 0,
            'entity_name': '',
            'part_number': '',
            'status_mask': [0] * 32,
            'date': '',
            'time': '',
            'idm_version_id': 0,
            'idm_item_id': 0,
            'primary_parent_uid': 0,
            'optimization_switches': 0,
        })
        fh.write('%6i\n%6i%74s\n' % (-1, 2400, ' '))
        fh.write('%12i%6i%6i%12i\n' % (dset['model_uid'], dset['entity_type'],
                                        dset['entity_subtype'], dset['version_number']))
        fh.write('%-80s\n' % _fit_string(dset['entity_name'], 80, 'entity_name', 2400))
        fh.write('%-80s\n' % _fit_string(dset['part_number'], 80, 'part_number', 2400))
        fh.write('%s\n' % ''.join(['%2i' % v for v in dset['status_mask']]))
        fh.write('%-10s%-10s%12i%12i%12i\n' % (_fit_string(dset['date'], 10, 'date', 2400),
                                                 _fit_string(dset['time'], 10, 'time', 2400),
                                                 dset['idm_version_id'], dset['idm_item_id'],
                                                 dset['primary_parent_uid']))
        fh.write('%12i\n' % dset['optimization_switches'])
        fh.write('%6i\n' % -1)
    except KeyError as msg:
        raise Exception('The required key \'' + msg.args[0] + '\' not present when writing data-set #2400')
    except:
        raise Exception('Error writing data-set #2400')


def _extract2400(block_data):
    """Extract dset data - data-set 2400."""
    dset = {'type': 2400}
    try:
        split_data = block_data.splitlines(True)
        dset.update(_parse_header_line(split_data[2], 4, [12, 6, 6, 12], [2, 2, 2, 2],
                                       ['model_uid', 'entity_type', 'entity_subtype', 'version_number']))
        dset.update(_parse_header_line(split_data[3], 1, [80], [1], ['entity_name']))
        dset.update(_parse_header_line(split_data[4], 1, [80], [1], ['part_number']))
        # Record 4: 32 consecutive I2 fields
        status_line = split_data[5]
        dset['status_mask'] = [int(status_line[i*2:(i*2)+2]) for i in range(32)]
        dset.update(_parse_header_line(split_data[6], 5, [10, 10, 12, 12, 12], [1, 1, 2, 2, 2],
                                       ['date', 'time', 'idm_version_id', 'idm_item_id', 'primary_parent_uid']))
        dset.update(_parse_header_line(split_data[7], 1, [12], [2], ['optimization_switches']))
    except:
        raise Exception('Error reading data-set #2400')
    return dset


def prepare_2400(
        model_uid=None,
        entity_type=None,
        entity_subtype=None,
        version_number=None,
        entity_name=None,
        part_number=None,
        status_mask=None,
        date=None,
        time=None,
        idm_version_id=None,
        idm_item_id=None,
        primary_parent_uid=None,
        optimization_switches=None,
        return_full_dict=False):
    """Prepare data-set 2400 (model header) for writing.

    The model header identifies the model (finite element or test model) to
    which the following data-sets belong: its unique identifier, name, part
    number, creation date and parent entities in the data management system
    of the program that wrote the file. Most fields matter only to that
    program; all are optional when writing. For a general file header see
    data-set 151.

    Parameters
    ----------
    model_uid : int, optional
        Model unique identifier (record 1, field 1). Default: 0.
    entity_type : int, optional
        Entity type (record 1, field 2). Default: 0.
    entity_subtype : int, optional
        Entity subtype (record 1, field 3). Default: 0.
    version_number : int, optional
        Version number (record 1, field 4). Default: 0.
    entity_name : str, optional
        Entity (model) name, up to 80 characters; longer text is truncated
        with a warning (record 2, field 1). Default: empty.
    part_number : str, optional
        Part number, up to 80 characters; longer text is truncated with a
        warning (record 3, field 1). Default: empty.
    status_mask : list of int, optional
        Status mask, a list of exactly 32 integers (record 4, field 1).
        Default: 32 zeros.
    date : str, optional
        Date in the form ``'DD-MMM-YY'``, e.g. ``'27-FEB-26'``, up to 10
        characters (record 5, field 1). Default: empty.
    time : str, optional
        Time in the form ``'HH:MM:SS'``, up to 10 characters (record 5,
        field 2). Default: empty.
    idm_version_id : int, optional
        IDM (data management) version ID (record 5, field 3). Default: 0.
    idm_item_id : int, optional
        IDM item ID (record 5, field 4). Default: 0.
    primary_parent_uid : int, optional
        Unique identifier of the primary parent entity (record 5, field 5).
        Default: 0.
    optimization_switches : {0, 1, 2, 3}, optional
        Optimization switches (record 6, field 1): 0 geometry off and
        analysis off, 1 geometry on and analysis off, 2 geometry off and
        analysis on, 3 geometry on and analysis on. Default: 0.
    return_full_dict : bool, optional
        If True, the dictionary has all keys of the data-set, with ``None`` for
        the values not given; if False (default), only the given values.

    Returns
    -------
    dict
        The data-set, to be written with :meth:`pyuff.UFF.write_sets`.

    Examples
    --------
    >>> import pyuff
    >>> header = pyuff.prepare_2400(
    ...     model_uid=1,
    ...     entity_name='Beam',
    ...     part_number='PART-001',
    ...     date='27-FEB-26',
    ...     time='12:00:00')
    >>> pyuff.UFF('model_header.uff').write_sets(header, mode='overwrite')
    """
    if np.array(model_uid).dtype != int and model_uid != None:
        raise TypeError('model_uid must be integer')
    if np.array(entity_type).dtype != int and entity_type != None:
        raise TypeError('entity_type must be integer')
    if np.array(entity_subtype).dtype != int and entity_subtype != None:
        raise TypeError('entity_subtype must be integer')
    if np.array(version_number).dtype != int and version_number != None:
        raise TypeError('version_number must be integer')
    if type(entity_name) != str and entity_name != None:
        raise TypeError('entity_name must be string')
    if type(part_number) != str and part_number != None:
        raise TypeError('part_number must be string')
    if status_mask != None and (not hasattr(status_mask, '__len__') or len(status_mask) != 32):
        raise ValueError('status_mask must be a list of 32 integers')
    if type(date) != str and date != None:
        raise TypeError('date must be string')
    if type(time) != str and time != None:
        raise TypeError('time must be string')
    if np.array(idm_version_id).dtype != int and idm_version_id != None:
        raise TypeError('idm_version_id must be integer')
    if np.array(idm_item_id).dtype != int and idm_item_id != None:
        raise TypeError('idm_item_id must be integer')
    if np.array(primary_parent_uid).dtype != int and primary_parent_uid != None:
        raise TypeError('primary_parent_uid must be integer')
    if np.array(optimization_switches).dtype != int and optimization_switches != None:
        raise TypeError('optimization_switches must be integer')

    dataset = {
        'type': 2400,
        'model_uid': model_uid,
        'entity_type': entity_type,
        'entity_subtype': entity_subtype,
        'version_number': version_number,
        'entity_name': entity_name,
        'part_number': part_number,
        'status_mask': status_mask,
        'date': date,
        'time': time,
        'idm_version_id': idm_version_id,
        'idm_item_id': idm_item_id,
        'primary_parent_uid': primary_parent_uid,
        'optimization_switches': optimization_switches,
    }

    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)

    return dataset
