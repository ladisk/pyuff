import numpy as np

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none


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
        fh.write('%-80s\n' % dset['entity_name'])
        fh.write('%-80s\n' % dset['part_number'])
        fh.write('%s\n' % ''.join(['%2i' % v for v in dset['status_mask']]))
        fh.write('%-10s%-10s%12i%12i%12i\n' % (dset['date'], dset['time'],
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
    """Name: Model Header

    R-Record, F-Field

    :param model_uid: R1 F1, Model UID
    :param entity_type: R1 F2, Entity type
    :param entity_subtype: R1 F3, Entity subtype
    :param version_number: R1 F4, Version number
    :param entity_name: R2 F1, Entity name
    :param part_number: R3 F1, Part number
    :param status_mask: R4 F1, Status mask (list of 32 integers)
    :param date: R5 F1, Date (DD-MMM-YY)
    :param time: R5 F2, Time (HH:MM:SS)
    :param idm_version_id: R5 F3, IDM version ID
    :param idm_item_id: R5 F4, IDM item ID
    :param primary_parent_uid: R5 F5, Primary parent UID
    :param optimization_switches: R6 F1, Optimization switches
    :param return_full_dict: If True full dict with all keys is returned, else only specified arguments are included

    **Test prepare_2400**

    >>> import pyuff
    >>> import os
    >>> save_to_file = 'test_pyuff'
    >>> dataset = pyuff.prepare_2400(
    >>>     model_uid=1,
    >>>     entity_type=7,
    >>>     entity_subtype=-1,
    >>>     version_number=0,
    >>>     entity_name='MyModel',
    >>>     part_number='PART-001',
    >>>     status_mask=[0]*32,
    >>>     date='27-FEB-26',
    >>>     time='12:00:00',
    >>>     idm_version_id=0,
    >>>     idm_item_id=0,
    >>>     primary_parent_uid=0,
    >>>     optimization_switches=0)
    >>> if save_to_file:
    >>>     if os.path.exists(save_to_file):
    >>>         os.remove(save_to_file)
    >>>     uffwrite = pyuff.UFF(save_to_file)
    >>>     uffwrite._write_set(dataset, 'add')
    >>> dataset
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
