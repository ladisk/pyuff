from os import write
import numpy as np

from ..tools import UFFException, _opt_fields, _parse_header_line, _write_record

# def _write15(fh, dset):
#     """Writes coordinate data - data-set 15 - to an open file fh"""
#     try:
#         n = len(dset['node_nums'])
#         # handle optional fields
#         dset = _opt_fields(dset, {'def_cs': np.asarray([0 for ii in range(0, n)], 'i'),
#                                         'disp_cs': np.asarray([0 for ii in range(0, n)], 'i'),
#                                         'color': np.asarray([0 for ii in range(0, n)], 'i')})
#         # write strings to the file
#         fh.write('%6i\n%6i%74s\n' % (-1, 15, ' '))
#         for ii in range(0, n):
#             fh.write('%10i%10i%10i%10i%13.5e%13.5e%13.5e\n' % (
#                 dset['node_nums'][ii], dset['def_cs'][ii], dset['disp_cs'][ii], dset['color'][ii],
#                 dset['x'][ii], dset['y'][ii], dset['z'][ii]))
#         fh.write('%6i\n' % -1)
#     except KeyError as msg:
#         raise UFFException('The required key \'' + msg.args[0] + '\' not present when writing data-set #15')
#     except:
#         raise UFFException('Error writing data-set #15')

FORMATS = [
    ['10.0f', '10.0f', '10.0f', '10.0f', '13.5f', '13.5f', '13.5f'],
]

def _write15(fh, dset):
    """Writes coordinate data - data-set 15 - to an open file fh"""
    try:
        n = len(dset['node_nums'])
        # handle optional fields
        dset = _opt_fields(dset, {'def_cs': np.asarray([0 for ii in range(0, n)], 'i'),
                                        'disp_cs': np.asarray([0 for ii in range(0, n)], 'i'),
                                        'color': np.asarray([0 for ii in range(0, n)], 'i')})
        # write strings to the file
        _write_record(fh, [-1, 15, ' '], formats=['6.0f', '6.0f', '74s'], multiline=True)
        
        for ii in range(0, n):
            _write_record(fh, 
                values=[dset['node_nums'][ii], dset['def_cs'][ii], dset['disp_cs'][ii], dset['color'][ii], dset['x'][ii], dset['y'][ii], dset['z'][ii]], 
                formats=FORMATS[0])

        _write_record(fh, -1, '6.0f')
    except KeyError as msg:
        raise UFFException('The required key \'' + msg.args[0] + '\' not present when writing data-set #15')
    except:
        raise UFFException('Error writing data-set #15')


def _extract15(blockData):
    """Extract coordinate data - data-set 15."""
    dset = {'type': 15}
    try:
        # Body
        split_data = blockData.splitlines()
        split_data = ''.join(split_data[2:]).split()
        split_data = [float(_) for _ in split_data]

        dset['node_nums'] = split_data[::7]
        dset['def_cs'] = split_data[1::7]
        dset['disp_cs'] = split_data[2::7]
        dset['color'] = split_data[3::7]
        dset['x'] = split_data[4::7]
        dset['y'] = split_data[5::7]
        dset['z'] = split_data[6::7]
    except:
        raise UFFException('Error reading data-set #15')
    return dset