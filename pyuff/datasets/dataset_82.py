import numpy as np
import os

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none, _fit_string

def get_structure_82(raw=False):
    """(source: https://www.ceas3.uc.edu/sdrluff/"""
    out = """
Universal Dataset Number: 82

Name:   Tracelines
-----------------------------------------------------------------------
 
             Record 1: FORMAT(3I10)
                       Field 1 -    trace line number
                       Field 2 -    number of nodes defining trace line
                                    (maximum of 250)
                       Field 3 -    color
 
             Record 2: FORMAT(80A1)
                       Field 1 -    Identification line
 
             Record 3: FORMAT(8I10)
                       Field 1 -    nodes defining trace line
                               =    > 0 draw line to node
                               =    0 move to node (a move to the first
                                    node is implied)
             Notes: 1) MODAL-PLUS node numbers must not exceed 8000.
                    2) Identification line may not be blank.
                    3) Systan only uses the first 60 characters of the
                       identification text.
                    4) MODAL-PLUS does not support trace lines longer than
                       125 nodes.
                    5) Supertab only uses the first 40 characters of the
                       identification line for a name.
                    6) Repeat Datasets for each Trace_Line
 
------------------------------------------------------------------------------
"""

    if raw:
        return out
    else:
        print(out)   

def _write82(fh, dset):
    """Writes line data - data-set 82 - to an open file fh"""
    try:
        # handle optional fields
        dset = _opt_fields(dset, {'id': 'NONE',
                                        'color': 0})
        # write strings to the file
        # removed jul 2017: unique_nodes = set(dset['nodes'])
        # removed jul 2017:if 0 in unique_nodes: unique_nodes.remove(0)
        # number of changes of node need to
        # n_nodes = len(dset['nodes'])
        n_nodes = np.sum((dset['nodes'][1:] - dset['nodes'][:-1]) != 0) + 1
        fh.write('%6i\n%6i%74s\n' % (-1, 82, ' '))
        fh.write('%10i%10i%10i\n' % (dset['trace_num'], n_nodes, dset['color']))
        fh.write('%-80s\n' % _fit_string(dset['id'], 80, 'id', 82))
        sl = 0
        n8_blocks = n_nodes // 8
        rem_lines = n_nodes % 8
        if n8_blocks:
            for ii in range(0, n8_blocks):
                #                 fh.write( string.join(['%10i'%line_n for line_n in dset['lines'][sl:sl+8]],'')+'\n' )
                fh.write(''.join(['%10i' % line_n for line_n in dset['nodes'][sl:sl + 8]]) + '\n')
                sl += 8
        if rem_lines > 0:
            fh.write(''.join(['%10i' % line_n for line_n in dset['nodes'][sl:]]) + '\n')
        #                 fh.write( string.join(['%10i'%line_n for line_n in dset['lines'][sl:]],'')+'\n' )
        fh.write('%6i\n' % -1)
    except KeyError as msg:
        raise Exception('The required key \'' + msg.args[0] + '\' not present when writing data-set #82')
    except:
        raise Exception('Error writing data-set #82')


def _extract82(block_data):
    """Extract line data - data-set 82."""
    dset = {'type': 82}
    try:
        split_data = block_data.splitlines(True)
        dset.update(
            _parse_header_line(split_data[2], 3, [10, 10, 10], [2, 2, 2], ['trace_num', 'n_nodes', 'color']))
        dset.update(_parse_header_line(split_data[3], 1, [80], [1], ['id']))
        split_data = ''.join(split_data[4:])
        split_data = split_data.split()
        dset['nodes'] = np.asarray([float(str) for str in split_data])
    except:
        raise Exception('Error reading data-set #82')
    return dset


def prepare_82(
        trace_num=None,
        n_nodes=None,
        color=None,
        id=None,
        nodes=None,
        return_full_dict=False):

    """Prepare data-set 82 (trace lines) for writing.

    A trace line connects nodes (data-set 15 or 2411) with straight lines to
    draw the wireframe of the measured structure, for example for animating
    mode shapes. Each data-set 82 holds one trace line; write a list of
    data-sets for several lines. ``pyuff.tools.convert_dataset_2412_to_82``
    creates trace lines from the elements of data-set 2412.

    Parameters
    ----------
    trace_num : int
        Trace line number (record 1, field 1).
    n_nodes : int, optional
        Number of nodes defining the trace line (maximum 250) (record 1,
        field 2). Ignored when writing: the writer computes it from ``nodes``.
    color : int, optional
        Color number, used only for display (record 1, field 3). Default: 0.
    id : str, optional
        Identification line, i.e. the name of the trace line, up to 80
        characters; longer text is truncated with a warning. The specification
        does not allow a blank line (record 2, field 1). Default: ``'NONE'``.
    nodes : numpy.ndarray of int
        Node labels along the line: a positive label draws a line to that
        node, 0 starts a new segment (move to the next node without drawing);
        a move to the first node is implied (record 3). Must be a NumPy array,
        not a list.
    return_full_dict : bool, optional
        If True, the dictionary has all keys of the data-set, with ``None`` for
        the values not given; if False (default), only the given values.

    Returns
    -------
    dict
        The data-set, to be written with :meth:`pyuff.UFF.write_sets`.

    Examples
    --------
    A rectangle through nodes 1 to 4 and a separate line from node 5 to 6:

    >>> import numpy as np
    >>> import pyuff
    >>> line = pyuff.prepare_82(
    ...     trace_num=1,
    ...     id='Frame',
    ...     nodes=np.array([1, 2, 3, 4, 1, 0, 5, 6]))
    >>> pyuff.UFF('trace_lines.uff').write_sets(line, mode='overwrite')
    """

    if np.array(trace_num).dtype != int and trace_num != None:
        raise TypeError('trace_num must be integer')
    if np.array(n_nodes).dtype != int and n_nodes != None:
        raise TypeError('n_nodes must be integer')
    if np.array(color).dtype != int and color != None:
        raise TypeError('color must be integer')
    if type(id) != str and id != None:
        raise TypeError(('id must be string'))
    if np.array(nodes).dtype != int and nodes != None:
        raise TypeError('nodes must be integers')
    

    dataset={
        'type': 82,
        'trace_num': trace_num,
        'n_nodes': n_nodes,
        'color': color,
        'id': id,
        'nodes': nodes 
        }


    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)


    return dataset

