import numpy as np
import os

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none

def get_structure_164(raw=False):
    """(source: https://www.ceas3.uc.edu/sdrluff/"""
    out = """
Universal Dataset Number: 164

Name:   Units
-----------------------------------------------------------------------

Record 1:       FORMAT(I10,20A1,I10)
                Field 1      -- units code
                                = 1 - SI: Meter (newton)
                                = 2 - BG: Foot (pound f)
                                = 3 - MG: Meter (kilogram f)
                                = 4 - BA: Foot (poundal)
                                = 5 - MM: mm (milli newton)
                                = 6 - CM: cm (centi newton)
                                = 7 - IN: Inch (pound f)
                                = 8 - GM: mm (kilogram f)
                                = 9 - US: USER_DEFINED
                Field 2      -- units description (used for
                                documentation only)
                Field 3      -- temperature mode
                                = 1 - absolute
                                = 2 - relative
Record 2:       FORMAT(3D25.17)
                Unit factors for converting universal file units to SI.
                To convert from universal file units to SI divide by
                the appropriate factor listed below.
                Field 1      -- length
                Field 2      -- force
                Field 3      -- temperature
                Field 4      -- temperature offset

-----------------------------------------------------------------
"""

    if raw:
        return out
    else:
        print(out)   

def _write164(fh, dset):
    """Writes units data - data-set 164 - to an open file fh."""
    try:
        # handle optional fields
        dset = _opt_fields(dset, {'units_description': 'User unit system',
                                        'temp_mode': 1})
        # write strings to the file
        fh.write('%6i\n%6i%74s\n' % (-1, 164, ' '))
        fh.write('%10i%20s%10i\n' % (dset['units_code'], dset['units_description'], dset['temp_mode']))
        str = '%25.16e%25.16e%25.16e\n%25.16e\n' % (
            dset['length'], dset['force'], dset['temp'], dset['temp_offset'])
        str = str.replace('e+', 'D+')
        str = str.replace('e-', 'D-')
        fh.write(str)
        fh.write('%6i\n' % -1)
    except KeyError as msg:
        raise Exception('The required key \'' + msg.args[0] + '\' not present when writing data-set #164')
    except:
        raise Exception('Error writing data-set #164')


def _extract164(block_data):
    """Extract units data - data-set 164."""
    dset = {'type': 164}
    try:
        split_data = block_data.splitlines(True)
        dset.update(_parse_header_line(split_data[2], 1, [10, 20, 10], [2, 1, 2],
                                            ['units_code', 'units_description', 'temp_mode']))
        split_data = ''.join(split_data[3:])
        split_data = split_data.split()
        dset['length'] = float(split_data[0].lower().replace('d', 'e'))
        dset['force'] = float(split_data[1].lower().replace('d', 'e'))
        dset['temp'] = float(split_data[2].lower().replace('d', 'e'))
        dset['temp_offset'] = float(split_data[3].lower().replace('d', 'e'))
    except:
        raise Exception('Error reading data-set #164')
    return dset


def prepare_164(
        units_code=None,
        units_description=None,
        temp_mode=None,
        length=None,
        force=None,
        temp=None,
        temp_offset=None,
        return_full_dict=False):
    """Name: Units

    R-Record, F-Field

    :param units_code: R1 F1, Units code: 
        1 - SI: Meter (newton),
        2 - BG: Foot (pound f),
        3 - MG: Meter (kilogram f),
        4 - BA: Foot (poundal),
        5 - MM: mm (milli newton),
        6 - CM: cm (centi newton),
        7 - IN: Inch (pound f),
        8 - GM: mm (kilogram f),
        9 - US: USER_DEFINED
    :param units_description: R1 F2, Units cription, optional
    :param temp_mode: R1 F3, Temperature mode (1-absolute, 2-relative), optional
    :param length: R2 F1, Length
    :param force: R2 F2, Force
    :param temp: R2 F3, Temperature
    :param temp_offset: R2 F4, Temperature offset
    :param return_full_dict: If True full dict with all keys is returned, else only specified arguments are included

    **Test prepare_164**

    >>> save_to_file = 'test_pyuff'
    >>> dataset = pyuff.prepare_164(
    >>>     units_code=1,
    >>>     units_description='SI units',
    >>>     temp_mode=1,
    >>>     length=3.28083989501312334,
    >>>     force=2.24808943099710480e-01,
    >>>     temp=1.8,
    >>>     temp_offset=459.67)
    >>> if save_to_file:
    >>>     if os.path.exists(save_to_file):
    >>>         os.remove(save_to_file)
    >>>     uffwrite = pyuff.UFF(save_to_file)
    >>>     uffwrite._write_set(dataset, 'add')
    >>> dataset
    """

    if units_code not in np.arange(10) and units_code != None:
        raise ValueError('units code must be integer between 0 and 9')
    if type(units_description) != str and units_description != None:
        raise TypeError('units_decription must be string')
    if temp_mode not in (1, 2, None):
        raise ValueError('temp_mode can be either 1 or 2')
    if type(length) != float and length != None:
        raise TypeError('length must be float')
    if type(force) != float and force != None:
        raise TypeError('force must be float')
    if type(temp) != float and temp != None:
        raise TypeError('temp must be float')
    if type(temp_offset) != float and temp_offset != None:
        raise TypeError('temp_offset must be float')

    dataset={
        'type': 164,
        'units_code':units_code,
        'units_description':units_description,
        'temp_mode':temp_mode,
        'length':length,
        'force':force,
        'temp':temp,
        'temp_offset':temp_offset
        }


    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)

    return dataset


