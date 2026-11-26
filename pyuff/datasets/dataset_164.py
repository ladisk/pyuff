import numpy as np
import os

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none, _fit_string

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
        fh.write('%10i%20s%10i\n' % (dset['units_code'],
                                       _fit_string(dset['units_description'], 20, 'units_description', 164),
                                       dset['temp_mode']))
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


def _extract164(dataset_data):
    """Extract units data - data-set 164."""
    dset = {'type': 164}
    try:
        split_data = dataset_data.splitlines(True)
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
    """Prepare data-set 164 (units) for writing.

    Data-set 164 states the unit system in which the values of the other
    data-sets in the file are given (coordinates, responses, forces). Besides
    the unit system code, it gives the factors for converting the file units
    to SI: a value in SI units is the value in file units divided by the
    factor. For SI units all three factors are 1.

    Parameters
    ----------
    units_code : int
        Unit system (record 1, field 1):

        - 1 SI: meter (newton)
        - 2 BG: foot (pound force)
        - 3 MG: meter (kilogram force)
        - 4 BA: foot (poundal)
        - 5 MM: mm (millinewton)
        - 6 CM: cm (centinewton)
        - 7 IN: inch (pound force)
        - 8 GM: mm (kilogram force)
        - 9 US: user defined
    units_description : str, optional
        Description of the unit system, for documentation only, up to 20
        characters; longer text is truncated with a warning (record 1,
        field 2). Default: ``'User unit system'``.
    temp_mode : {1, 2}, optional
        Temperature mode: 1 absolute, 2 relative (record 1, field 3).
        Default: 1.
    length : float
        Length conversion factor, e.g. 3.28083989501312 for feet (record 2,
        field 1).
    force : float
        Force conversion factor, e.g. 0.224808943099710 for pound force
        (record 2, field 2).
    temp : float
        Temperature conversion factor, e.g. 1.8 for degrees Fahrenheit
        (record 2, field 3).
    temp_offset : float
        Temperature offset, e.g. 273.15 for Celsius or 459.67 for Fahrenheit
        (record 2, field 4).
    return_full_dict : bool, optional
        If True, the dictionary has all keys of the data-set, with ``None`` for
        the values not given; if False (default), only the given values.

    Returns
    -------
    dict
        The data-set, to be written with :meth:`pyuff.UFF.write_datasets`.

    Notes
    -----
    ``length``, ``force``, ``temp`` and ``temp_offset`` must be Python
    ``float`` values (``1.0``, not ``1``).

    Examples
    --------
    >>> import pyuff
    >>> units = pyuff.prepare_164(
    ...     units_code=1,
    ...     units_description='SI units',
    ...     temp_mode=2,
    ...     length=1.0,
    ...     force=1.0,
    ...     temp=1.0,
    ...     temp_offset=273.15)
    >>> pyuff.UFF('units.uff').write_datasets(units, mode='overwrite')
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


