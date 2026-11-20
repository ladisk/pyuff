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
    """Prepare data-set 15 (nodes) for writing.

    Data-set 15 holds the node labels and their coordinates in single
    precision; it is the older, simpler alternative to data-set 2411, which
    stores the coordinates in double precision. The nodes are the points of
    the geometry to which, for example, the response degrees of freedom of
    data-set 58 refer and between which trace lines (data-set 82) are drawn.
    All arrays describe the same nodes and must have the same length.

    Parameters
    ----------
    node_nums : array_like of int
        Node labels (record 1, field 1).
    def_cs : array_like of int, optional
        Definition coordinate system number of each node, i.e. the coordinate
        system (data-set 18 or 2420) in which ``x``, ``y`` and ``z`` are given;
        0 is the global system. Same length as ``node_nums`` (record 1,
        field 2). Default: 0 for all nodes.
    disp_cs : array_like of int, optional
        Displacement coordinate system number of each node, i.e. the coordinate
        system in which the displacements (responses) at the node are expressed;
        0 is the global system. Same length as ``node_nums`` (record 1,
        field 3). Default: 0 for all nodes.
    color : array_like of int, optional
        Color number of each node, used only for display. Same length as
        ``node_nums`` (record 1, field 4). Default: 0 for all nodes.
    x : array_like of float
        x coordinate of each node in its definition coordinate system. Same
        length as ``node_nums`` (record 1, field 5).
    y : array_like of float
        y coordinate of each node (record 1, field 6).
    z : array_like of float
        z coordinate of each node (record 1, field 7).
    return_full_dict : bool, optional
        If True, the dictionary has all keys of the data-set, with ``None`` for
        the values not given; if False (default), only the given values.

    Returns
    -------
    dict
        The data-set, to be written with :meth:`pyuff.UFF.write_sets`.

    Examples
    --------
    >>> import numpy as np
    >>> import pyuff
    >>> nodes = pyuff.prepare_15(
    ...     node_nums=np.array([1, 2, 3, 4]),
    ...     x=np.array([0.0, 0.5, 1.0, 1.5]),
    ...     y=np.zeros(4),
    ...     z=np.zeros(4))
    >>> pyuff.UFF('beam_nodes.uff').write_sets(nodes, mode='overwrite')
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
