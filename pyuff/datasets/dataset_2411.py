import numpy as np

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none, _column_arrays
from .._vector_text import format_rows

def get_structure_2411(raw=False):
    """(source: https://www.ceas3.uc.edu/sdrluff/"""
    out = """
Universal Dataset Number: 2411

Name:   Nodes - Double Precision
----------------------------------------------------------------------------

Record 1:        FORMAT(4I10)
                 Field 1       -- node label
                 Field 2       -- export coordinate system number
                 Field 3       -- displacement coordinate system number
                 Field 4       -- color
Record 2:        FORMAT(1P3D25.16)
                 Fields 1-3    -- node coordinates in the part coordinate
                                  system
 
Records 1 and 2 are repeated for each node in the model.
  
----------------------------------------------------------------------------
"""

    if raw:
        return out
    else:
        print(out)   

def _write2411(fh, dset):
    try:
        dict = {}

        dset = _opt_fields(dset, dict)
        fh.write('%6i\n%6i%74s\n' % (-1, 2411, ' '))
        
        n = len(dset['node_nums'])
        node, def_cs, disp_cs, color, x, y, z = _column_arrays(
            n, dset['node_nums'], dset['def_cs'], dset['disp_cs'], dset['color'], dset['x'], dset['y'], dset['z'])
        if node.dtype.kind not in 'biuf':
            node = [int(node_id) for node_id in node]
        fh.write(format_rows([('%10i', node), ('%10i', def_cs), ('%10i', disp_cs), ('%10i', color), '\n',
                              ('%25.16e', x), ('%25.16e', y), ('%25.16e', z), '\n'], n))

        fh.write('%6i\n' % -1)

    except:
        raise Exception('Error writing data-set #2411')


def _extract2411(block_data):
    """Extract coordinate data - data-set 2411."""
    dset = {'type': 2411}
    try:
        # Body
        split_data = block_data.splitlines(True)  # Keep the line breaks!
        split_data = ''.join(split_data[2:])  # ..as they are again needed
        split_data = split_data.replace("D", "E").replace("d", "E").split()
        # replace to support D or d notation as an exponential notation (typically form Unigraphics IDEAS)
        values = np.array(split_data, 'd')
        dset['node_nums'] = values[::7].copy()
        dset['def_cs'] = values[1::7].copy()
        dset['disp_cs'] = values[2::7].copy()
        dset['color'] = values[3::7].copy()
        dset['x'] = values[4::7].copy()
        dset['y'] = values[5::7].copy()
        dset['z'] = values[6::7].copy()
    except:
        raise Exception('Error reading data-set #2411')
    return dset


def prepare_2411(
        node_nums=None,
        def_cs=None,
        disp_cs=None,
        color=None,
        x=None,
        y=None,
        z=None,
        return_full_dict=False):
    """Prepare data-set 2411 (nodes, double precision) for writing.

    The data-set holds the nodes of a model: the label, the coordinate systems and the color of
    each node and its coordinates, written in double precision. The nodes are the points at which
    data-sets 55 and 58 give the response (e.g. the measurement points of a modal test); they are
    connected into a mesh by the elements of data-set 2412. All arrays have one value per node,
    shape ``(n,)``.

    Each node refers to two coordinate systems by their labels (as defined e.g. in data-set 2420):
    the export coordinate system, in which the node coordinates are given, and the displacement
    coordinate system, in which the directions of the degrees of freedom at the node are given.
    Programs use 0 or 1 for the global Cartesian coordinate system.

    Parameters
    ----------
    node_nums : array_like of int
        Node labels (record 1, field 1).
    def_cs : array_like of int
        Export coordinate system number of each node, the coordinate system of ``x``, ``y`` and
        ``z`` (record 1, field 2).
    disp_cs : array_like of int
        Displacement coordinate system number of each node, the coordinate system of the degrees
        of freedom at the node (record 1, field 3).
    color : array_like of int
        Color number of each node, used by pre- and post-processors only, e.g. 8 (record 1,
        field 4).
    x : array_like of float
        X coordinates of the nodes (record 2, field 1).
    y : array_like of float
        Y coordinates of the nodes (record 2, field 2).
    z : array_like of float
        Z coordinates of the nodes (record 2, field 3).
    return_full_dict : bool, optional
        If True, the dictionary has all keys of the data-set, with ``None`` for
        the values not given; if False (default), only the given values.

    Returns
    -------
    dict
        The data-set, to be written with :meth:`pyuff.UFF.write_sets`.

    Examples
    --------
    Four nodes of a square plate, in the global coordinate system:

    >>> import numpy as np
    >>> import pyuff
    >>> nodes = pyuff.prepare_2411(
    ...     node_nums=np.array([1, 2, 3, 4]),
    ...     def_cs=np.zeros(4, dtype=int),
    ...     disp_cs=np.zeros(4, dtype=int),
    ...     color=np.full(4, 8),
    ...     x=np.array([0.0, 1.0, 1.0, 0.0]),
    ...     y=np.array([0.0, 0.0, 1.0, 1.0]),
    ...     z=np.zeros(4))
    >>> pyuff.UFF('nodes.uff').write_sets(nodes, mode='overwrite')
    """
    # **Test prepare_2411**
    #>>> save_to_file = 'test_pyuff'
    #>>> dataset = pyuff.prepare_2411(
    #>>>     node_nums=np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]),
    #>>>     def_cs=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    #>>>     disp_cs=np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]),
    #>>>     color=np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8]),
    #>>>     x=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
    #>>>     y=np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
    #>>>     z=np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
    #>>> if save_to_file:
    #>>>     if os.path.exists(save_to_file):
    #>>>         os.remove(save_to_file)
    #>>>     uffwrite = pyuff.UFF(save_to_file)
    #>>>     uffwrite.write_sets(dataset, mode='add')
    #>>> dataset

    if np.array(node_nums).dtype != int and node_nums != None:
        raise TypeError('node_nums must be integer')
    if np.array(def_cs).dtype != int and def_cs != None:
        raise TypeError('def_cs must be integer')
    if np.array(disp_cs).dtype != int and disp_cs != None:
        raise TypeError('disp_cs must be integer')
    if np.array(color).dtype != int and color != None:
        raise TypeError('color must be integer')
    if np.array(x).dtype != float and x != None:
        raise TypeError('x must be float')
    if np.array(y).dtype != float and y != None:
        raise TypeError('y must be float')
    if np.array(z).dtype != float and z != None:
        raise TypeError('z must be float')

    dataset={
        'type': 2411,
        'node_nums': node_nums,
        'def_cs': def_cs,
        'disp_cs': disp_cs,
        'color': color,
        'x':  x,
        'y': y,
        'z': z
        }
    
    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)

    return dataset
