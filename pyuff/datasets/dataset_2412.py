import numpy as np
import itertools

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none

def get_structure_2412(raw=False):
    """(source: https://www.ceas3.uc.edu/sdrluff/"""
    out = """
Universal Dataset Number: 2412

Name:   Elements
-----------------------------------------------------------------------
 
Record 1:        FORMAT(6I10)
                 Field 1       -- element label
                 Field 2       -- fe descriptor id
                 Field 3       -- physical property table number
                 Field 4       -- material property table number
                 Field 5       -- color
                 Field 6       -- number of nodes on element
 
Record 2:  *** FOR NON-BEAM ELEMENTS ***
                 FORMAT(8I10)
                 Fields 1-n    -- node labels defining element
 
Record 2:  *** FOR BEAM ELEMENTS ONLY ***
                 FORMAT(3I10)
                 Field 1       -- beam orientation node number
                 Field 2       -- beam fore-end cross section number
                 Field 3       -- beam  aft-end cross section number
 
Record 3:  *** FOR BEAM ELEMENTS ONLY ***
                 FORMAT(8I10)
                 Fields 1-n    -- node labels defining element
 
Records 1 and 2 are repeated for each non-beam element in the model.
Records 1 - 3 are repeated for each beam element in the model.
 
------------------------------------------------------------------------------
"""

    if raw:
        return out
    else:
        print(out)   

def _write2412(fh, dset):
    try:
        fh.write('%6i\n%6i%74s\n' % (-1, 2412, ' '))
        for el_type in dset:
            if type(el_type) is not int:
                # Skip 'type', 'triangle' and 'quad' indices
                continue
            for elem in dset[el_type]:
                fh.write('%10i%10i%10i%10i%10i%10i\n' % (
                    elem['element_nums'],
                    elem['fe_descriptor'],
                    elem['phys_table'],
                    elem['mat_table'],
                    elem['color'],
                    elem['num_nodes'],
                ))
                if elem['fe_descriptor'] == 11:
                # Rods have to be written in 3 lines - ad additional line here
                    fh.write('%10i%10i%10i\n' % (
                        elem['beam_orientation'],
                        elem['beam_foreend_cross'],
                        elem['beam_aftend_cross']
                    ))
                for ii in elem['nodes_nums']:
                    fh.write('%10i' % ii)
                fh.write('\n')
        fh.write('%6i\n' % -1)
    except:
        raise Exception('Error writing data-set #2412')


def _extract2412(block_data):
    """Extract element data - data-set 2412."""
    dset = {'type': 2412}
    # Define dictionary of possible elements types for legacy interface
    elt_type_dict = {41: 'triangle', 44: 'quad'}
    # Elements that are seen as rods and read as 3 lines
    rods_dict = {11, 21, 22, 23, 24}

    # Read data
    try:
        split_data = block_data.splitlines()
        split_data = [a.split() for a in split_data][2:]

        # Extract Records
        i = 0
        while i < len(split_data):
            dict_tmp = dict()
            line = split_data[i]
            dict_tmp['element_nums'] = int(line[0])
            dict_tmp['fe_descriptor'] = int(line[1])
            dict_tmp['phys_table'] = int(line[2])
            dict_tmp['mat_table'] = int(line[3])
            dict_tmp['color'] = int(line[4])
            dict_tmp['num_nodes'] = int(line[5])
            if dict_tmp['fe_descriptor'] in rods_dict:
                # element is a rod and covers 3 lines
                dict_tmp['beam_orientation'] = int(split_data[i+1][0])
                dict_tmp['beam_foreend_cross'] = int(split_data[i + 1][1])
                dict_tmp['beam_aftend_cross'] = int(split_data[i + 1][2])
                dict_tmp['nodes_nums'] = [int(e) for e in split_data[i+2]]
                i += 3
            else:
                # element is no rod and covers 2 lines
                dict_tmp['nodes_nums'] = [int(e) for e in split_data[i+1]]
                i += 2
            desc = dict_tmp['fe_descriptor']
            if not desc in dset:
                dset[desc] = []
            dset[desc].append(dict_tmp)
        for num, name in elt_type_dict.items():
            # if we have one of the keys that are enabled for the legacy interface, add everything for that here
            if num in dset.keys():
                dset[name] = {
                    'element_nums': [e['element_nums'] for e in dset[num]],
                    'fe_descriptor': [e['fe_descriptor'] for e in dset[num]],
                    'phys_table': [e['phys_table'] for e in dset[num]],
                    'mat_table': [e['mat_table'] for e in dset[num]],
                    'color': [e['color'] for e in dset[num]],
                    'num_nodes': [e['num_nodes'] for e in dset[num]],
                    'nodes_nums': [e['nodes_nums'] for e in dset[num]],
                }

        return dset

    except:
        raise Exception('Error reading data-set #2412')


def prepare_2412(
        element_nums=None,
        fe_descriptor=None,
        phys_table=None,
        mat_table=None,
        color=None,
        num_nodes=None,
        nodes_nums=None,
        beam_orientation=None,
        beam_foreend_cross=None,
        beam_aftend_cross=None,
        return_full_dict=False):
    """Name: Elements

    R-Record, F-Field

    :param element_nums: R1 F1, List of n element numbers
    :param fe_descriptor: R1 F2, Fe descriptor id
    :param phys_table: R1 F3, Physical property table number
    :param mat_table: R1 F4, Material property table number
    :param color: R1 F5, Color, optional
    :param num_nodes: R1 F6, Number of nodes on element
    :param nodes_nums: R2 F1 (R3 FOR RODS), Node labels defining element
    :param beam_orientation: R2 F1 FOR RODS ONLY, beam orientation node number
    :param beam_foreend_cross: R2 F2 FOR RODS ONLY, beam fore-end cross section number
    :param beam_aftend_cross: R2 F3 FOR RODS ONLY, beam aft-end cross section number
    :param return_full_dict: If True full dict with all keys is returned, else only specified arguments are included

    **Test prepare_2412**
    
    >>> save_to_file = 'test_pyuff'
    >>> data = pyuff.prepare_2412(
    >>>     element_nums=np.array([69552, 98919, 69304]),
    >>>     fe_descriptor=np.array([94, 94, 94]),
    >>>     phys_table=np.array([2, 2, 2]),
    >>>     mat_table=np.array([1, 1, 1]),
    >>>     color=np.array([2, 2, 2]),
    >>>     nodes_nums=np.array([[   29,  2218,  2219,30],[81619, 83403,  2218,    29],[   30,  2219,  2119,    31],[  31, 2119, 1659,   32]]))
    >>> dataset = {'type':2412, 'quad':data}
    >>> if save_to_file:
    >>>     if os.path.exists(save_to_file):
    >>>         os.remove(save_to_file)
    >>>     uffwrite = pyuff.UFF(save_to_file)
    >>>     uffwrite.write_sets(dataset, mode='add')
    >>> dataset
    """
    if np.array(element_nums).dtype != int and element_nums != None:
        raise TypeError('element_nums must be integer')
    if np.array(fe_descriptor).dtype != int and fe_descriptor != None:
        raise TypeError('fe_descriptor must be integer')
    if np.array(phys_table).dtype != int and phys_table != None:
        raise TypeError('phys_table must be integer')
    if np.array(mat_table).dtype != int and mat_table != None:
        raise TypeError('mat_table must be integer')
    if np.array(color).dtype != int and color != None:
        raise TypeError('color must be integer')
    if np.array(num_nodes).dtype != int and num_nodes != None:
        raise TypeError('num_nodes must be integer')
    if np.array(nodes_nums).dtype != int and nodes_nums != None:
        raise TypeError('nodes_nums must be integer')
    if np.array(beam_orientation).dtype != int and beam_orientation != None:
        raise TypeError('beam_orientation must be integer')
    if np.array(beam_foreend_cross).dtype != int and beam_foreend_cross != None:
        raise TypeError('beam_foreend_cross must be integer')
    if np.array(beam_aftend_cross).dtype != int and beam_aftend_cross != None:
        raise TypeError('beam_aftend_cross must be integer')


    dataset={
        'type': 2412,
        'element_nums': element_nums,
        'fe_descriptor': fe_descriptor,
        'phys_table': phys_table,
        'mat_table': mat_table,
        'color': color,
        'num_nodes': num_nodes,
        'beam_orientation': beam_orientation,
        'beam_foreend_cross': beam_foreend_cross,
        'beam_aftend_cross': beam_aftend_cross,
        'nodes_nums': nodes_nums
        }
    
    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)

    return dataset

