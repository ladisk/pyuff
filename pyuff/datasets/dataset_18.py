import numpy as np

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none


def get_structure_18(raw=False):
    """(source: https://www.ceas3.uc.edu/sdrluff/"""
    out = """
Universal Dataset Number: 18

Name:   Coordinate Systems
-----------------------------------------------------------------------

Record 1:        FORMAT(5I10)
                 Field 1       -- coordinate system number
                 Field 2       -- coordinate system type
                 Field 3       -- reference coordinate system number
                 Field 4       -- color
                 Field 5       -- method of definition
                               = 1 - origin, +x axis, +xz plane

Record 2:        FORMAT(20A2)
                 Field 1       -- coordinate system name

Record 3:        FORMAT(1P6E13.5)
                 Total of 9 coordinate system definition parameters.
                 Fields 1-3    -- origin of new system specified in
                                  reference system
                 Fields 4-6    -- point on +x axis of the new system
                                  specified in reference system
                 Fields 7-9    -- point on +xz plane of the new system
                                  specified in reference system

Records 1 thru 3 are repeated for each coordinate system in the model.

-----------------------------------------------------------------
"""

    if raw:
        return out
    else:
        print(out)


def _write18(fh, dset):
    """Writes dset data - data-set 18 - to an open file fh."""
    try:
        n = len(dset['cs_num'])
        dset = _opt_fields(dset, {
            'cs_type': np.zeros(n, dtype=int),
            'color': np.zeros(n, dtype=int),
            'method': np.ones(n, dtype=int),
            'cs_name': [''] * n,
        })
        fh.write('%6i\n%6i%74s\n' % (-1, 18, ' '))
        for i in range(n):
            fh.write('%10i%10i%10i%10i%10i\n' % (
                int(dset['cs_num'][i]),
                int(dset['cs_type'][i]),
                int(dset['ref_cs_num'][i]),
                int(dset['color'][i]),
                int(dset['method'][i])))
            fh.write('%-40s\n' % dset['cs_name'][i])
            fh.write('%13.5e%13.5e%13.5e%13.5e%13.5e%13.5e\n' % (
                dset['ref_o'][i][0], dset['ref_o'][i][1], dset['ref_o'][i][2],
                dset['x_point'][i][0], dset['x_point'][i][1], dset['x_point'][i][2]))
            fh.write('%13.5e%13.5e%13.5e\n' % (
                dset['xz_point'][i][0], dset['xz_point'][i][1], dset['xz_point'][i][2]))
        fh.write('%6i\n' % -1)
    except KeyError as msg:
        raise Exception('The required key \'' + msg.args[0] + '\' not present when writing data-set #18')
    except:
        raise Exception('Error writing data-set #18')


def _extract18(block_data):
    '''Extract local CS definitions -- data-set 18.'''
    dset = {'type': 18}
    try:
        split_data = block_data.splitlines()

        # -- Get Record 1
        rec_1 = np.array(list(map(float, ''.join(split_data[2::4]).split())))

        dset['cs_num'] = rec_1[::5]
        dset['cs_type'] = rec_1[1::5]
        dset['ref_cs_num'] = rec_1[2::5]
        dset['color'] = rec_1[3::5]
        dset['method'] = rec_1[4::5]

        # -- Get Record 2
        dset['cs_name'] = [s.strip() for s in split_data[3::4]]

        # -- Get Record 31 and 32
        # ... these are the origins of cs defined in ref
        line_data = ''.join(split_data[4::4])
        rec_31 = [float(line_data[i * 13:(i + 1) * 13]) for i in range(int(len(line_data) / 13))]
        dset['ref_o'] = np.vstack((np.array(rec_31[::6]),
                                    np.array(rec_31[1::6]),
                                    np.array(rec_31[2::6]))).transpose()

        # ... these are points on the x axis of cs defined in ref
        dset['x_point'] = np.vstack((np.array(rec_31[3::6]),
                                        np.array(rec_31[4::6]),
                                        np.array(rec_31[5::6]))).transpose()

        # ... these are the points on the xz plane
        line_data = ''.join(split_data[5::4])
        rec_32 = [float(line_data[i * 13:(i + 1) * 13]) for i in range(int(len(line_data) / 13))]
        dset['xz_point'] = np.vstack((np.array(rec_32[::3]),
                                        np.array(rec_32[1::3]),
                                        np.array(rec_32[2::3]))).transpose()
    except:
        raise Exception('Error reading data-set #18')
    return dset


def prepare_18(
        cs_num=None,
        cs_type=None,
        ref_cs_num=None,
        color=None,
        method=None,
        cs_name=None,
        ref_o=None,
        x_point=None,
        xz_point=None,
        return_full_dict=False):
    """Name: Coordinate Systems

    R-Record, F-Field

    :param cs_num: R1 F1, Coordinate system number (array of int)
    :param cs_type: R1 F2, Coordinate system type (array of int)
    :param ref_cs_num: R1 F3, Reference coordinate system number (array of int)
    :param color: R1 F4, Color (array of int)
    :param method: R1 F5, Method of definition (array of int, 1=origin+x+xz)
    :param cs_name: R2 F1, Coordinate system name (list of str)
    :param ref_o: R3 F1-3, Origin of new system in reference system (Nx3 array)
    :param x_point: R3 F4-6, Point on +x axis in reference system (Nx3 array)
    :param xz_point: R3 F7-9, Point on +xz plane in reference system (Nx3 array)
    :param return_full_dict: If True full dict with all keys is returned, else only specified arguments are included

    **Test prepare_18**

    >>> import pyuff
    >>> import os
    >>> save_to_file = 'test_pyuff'
    >>> dataset = pyuff.prepare_18(
    >>>     cs_num=[1, 2],
    >>>     cs_type=[0, 0],
    >>>     ref_cs_num=[0, 0],
    >>>     color=[1, 1],
    >>>     method=[1, 1],
    >>>     cs_name=['CS1', 'CS2'],
    >>>     ref_o=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    >>>     x_point=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    >>>     xz_point=[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    >>> if save_to_file:
    >>>     if os.path.exists(save_to_file):
    >>>         os.remove(save_to_file)
    >>>     uffwrite = pyuff.UFF(save_to_file)
    >>>     uffwrite._write_set(dataset, 'add')
    >>> dataset
    """
    dataset = {
        'type': 18,
        'cs_num': cs_num,
        'cs_type': cs_type,
        'ref_cs_num': ref_cs_num,
        'color': color,
        'method': method,
        'cs_name': cs_name,
        'ref_o': ref_o,
        'x_point': x_point,
        'xz_point': xz_point,
    }

    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)

    return dataset
