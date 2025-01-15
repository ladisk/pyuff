from os import write
import numpy as np

from ..tools import _opt_fields, _parse_header_line, _write_record, check_dict_for_none

def get_structure_15(raw=False):
    """(source: https://www.ceas3.uc.edu/sdrluff/"""
    out = """
Universal Dataset Number: 15

Name:   Nodes
-----------------------------------------------------------------------
 
             Record 1: FORMAT(4I10,1P3E13.5)
                       Field 1 -    node label
                       Field 2 -    definition coordinate system number
                       Field 3 -    displacement coordinate system number
                       Field 4 -    color
                       Field 5-7 -  3 - Dimensional coordinates of node
                                    in the definition system
 
             NOTE:  Repeat record for each node
 
------------------------------------------------------------------------------
"""

    if raw:
        return out
    else:
        print(out)   

FORMATS = [
    ['10.0f', '10.0f', '10.0f', '10.0f', '13.5E', '13.5E', '13.5E'],
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
        _write_record(fh, [-1, 15], formats=['6.0f', '6.0f'], multiline=True)
        
        for ii in range(0, n):
            _write_record(fh, 
                values=[dset['node_nums'][ii], dset['def_cs'][ii], dset['disp_cs'][ii], dset['color'][ii], dset['x'][ii], dset['y'][ii], dset['z'][ii]], 
                formats=FORMATS[0])

        _write_record(fh, -1, '6.0f')
    except KeyError as msg:
        raise Exception('The required key \'' + msg.args[0] + '\' not present when writing data-set #15')
    except:
        raise Exception('Error writing data-set #15')


def _extract15(block_data):
    """Extract coordinate data - data-set 15."""
    dset = {'type': 15}
    try:
        # Body
        split_data = block_data.splitlines()
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


def prepare_15(
        node_nums=None,
        def_cs=None, 
        disp_cs=None, 
        color=None, 
        x=None,
        y=None,
        z=None,
        return_full_dict=False
        ):
    """Name: Nodes
    
    R-Record, F-Field

    :param node_nums: R1 F1, node label
    :param def_cs: R1 F2, deformation coordinate system numbers, optional
    :param disp_cs: R1 F3, displacement coordinate system numbers, optional
    :param color: R1 F4, color, optional
    :param x: R1 F5, Dimensional coordinate of node in the definition system
    :param y: R1 F6, Dimensional coordinate of node in the definition system
    :param z: R1 F7, Dimensional coordinate of node in the definition system
    
    :param return_full_dict: If True full dict with all keys is returned, else only specified arguments are included

    **Test prepare_15**

    >>> save_to_file = 'test_pyuff'
    >>> dataset = pyuff.prepare_15(
    >>>     node_nums=[16, 17, 18, 19, 20],
    >>>     def_cs=[11, 11, 11, 12, 12],
    >>>     disp_cs=[16, 16, 17, 18, 19],
    >>>     color=[1, 3, 4, 5, 6],  # I10,
    >>>     x=[0.0, 1.53, 0.0, 1.53, 0.0],
    >>>     y=[0.0, 0.0, 3.84, 3.84, 0.0],
    >>>     z=[0.0, 0.0, 0.0, 0.0, 1.83])
    >>> if save_to_file:
    >>>     if os.path.exists(save_to_file):
    >>>         os.remove(save_to_file)
    >>>     uffwrite = pyuff.UFF(save_to_file)
    >>>     uffwrite._write_set(dataset, 'add')
    >>> dataset
    """

    if type(node_nums) not in (list, tuple, np.ndarray) and node_nums != None:
        raise TypeError('node_nums must be either list, tuple or numpy.ndarray')
    if np.array(node_nums).dtype != int and node_nums != None:
        raise TypeError('node_nums must be integers.')
    if np.array(def_cs).dtype != int and def_cs != None:
        raise TypeError('def_cs must be integers.')
    if np.array(disp_cs).dtype != int and disp_cs != None:
        raise TypeError('disp_cs must be integers.')
    if np.array(color).dtype != int and color != None:
        raise TypeError('color must be integers.')
    if np.array(x).dtype != float and x != None:
        raise TypeError('x must be float.')
    if np.array(y).dtype != float and y != None:
        raise TypeError('y must be float.')
    if np.array(z).dtype != float and z != None:
        raise TypeError('z must be float.')
    

    dataset={
        'type': 15,
        'node_nums': node_nums,
        'def_cs': def_cs, 
        'disp_cs': disp_cs,  
        'color': color,  
        'x': x,  
        'y': y,  
        'z': z 
        }

    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)

    return dataset
