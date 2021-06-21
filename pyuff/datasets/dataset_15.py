from os import write
import os
import numpy as np

from ..tools import _opt_fields, _parse_header_line, _write_record, check_dict_for_none
from .. import pyuff

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
#         raise Exception('The required key \'' + msg.args[0] + '\' not present when writing data-set #15')
#     except:
#         raise Exception('Error writing data-set #15')

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
        raise Exception('The required key \'' + msg.args[0] + '\' not present when writing data-set #15')
    except:
        raise Exception('Error writing data-set #15')


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
        raise Exception('Error reading data-set #15')
    return dset


def dict_15(node_nums=None, def_cs=None, disp_cs=None, color=None, x=None,y=None,z=None,return_full_dict=False):
    """Name: Nodes
    
    R-Record, F-Field

    :param node_nums: R1 F1, node label
    :param <def_cs>: R1 F2, deformation coordinate system numbers 
    :param disp_cs: R1 F3, displacement coordinate system numbers
    :param color: R1 F4, color
    :param x: R1 F5, Dimensional coordinate of node in the definition system
    :param y: R1 F6, Dimensional coordinate of node in the definition system
    :param z: R1 F7, Dimensional coordinate of node in the definition system
    
    :param return_full_dict: If True full dict with all keys is returned, else only specified arguments are included
    """
    dataset={'type': 15,
        'node_nums': node_nums,
        'def_cs': def_cs, 
        'disp_cs': disp_cs,  
        'color': color,  
        'x': x,  
        'y': y,  
        'z': z }

    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)

    return dataset


def prepare_test_15(save_to_file=''):
    dataset = {'type': 15,  # Nodes
               'node_nums': [16, 17, 18, 19, 20],  # I10, node label
               'def_cs': [11, 11, 11, 12, 12],  # I10, definition coordinate system number
               'disp_cs': [16, 16, 17, 18, 19],  # I10, displacement coordinate system number
               'color': [1, 3, 4, 5, 6],  # I10, color
               'x': [0.0, 1.53, 0.0, 1.53, 0.0],  # E13.5
               'y': [0.0, 0.0, 3.84, 3.84, 0.0],  # E13.5
               'z': [0.0, 0.0, 0.0, 0.0, 1.83]}  # E13.5
    dataset_out = dataset.copy()

    if save_to_file:
        if os.path.exists(save_to_file):
            os.remove(save_to_file)
        uffwrite = pyuff.UFF(save_to_file)
        uffwrite._write_set(dataset, 'add')

    return dataset_out