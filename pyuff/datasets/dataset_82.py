import numpy as np
import os

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none

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
        fh.write('%-80s\n' % dset['id'])
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

    """Name: Tracelines

    R-Record, F-Field

    :param trace_num: R1 F1, Trace line number
    :param n_nodes: R1 F2, number of nodes defining trace line (maximum of 250), ignored
    :param color: R1 F3, color, optional
    :param id: R2 F1, identification line, optional
    :param nodes: R3 F1, nodes defining trace line (0 move to node, >0 draw line to node)
    
    :param return_full_dict: If True full dict with all keys is returned, else only specified arguments are included
    
    **Test prepare_82**

    >>> save_to_file = 'test_pyuff'
    >>> dataset = pyuff.prepare_82(
    >>>     trace_num=2,
    >>>     n_nodes=7,
    >>>     color=30,
    >>>     id='Identification line',
    >>>     nodes=np.array([0, 10, 13, 14, 15, 16, 17]))
    >>> if save_to_file:
    >>>     if os.path.exists(save_to_file):
    >>>         os.remove(save_to_file)
    >>>     uffwrite = pyuff.UFF(save_to_file)
    >>>     uffwrite._write_set(dataset, 'add')
    >>> dataset

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

