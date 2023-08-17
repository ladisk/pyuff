import numpy as np

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none

def _write2412(fh, dset):
    try:
        elt_type_dict = {'triangle': 3, 'quad': 4}
        fh.write('%6i\n%6i%74s\n' % (-1, 2412, ' '))
        for elt_type in dset:
            if elt_type == "type":
                pass
            else:
                #for i in range(len(dset[elt_type]['element_nums'])):
                for elem in dset['all']:
                    fh.write('%10i%10i%10i%10i%10i%10i\n' % (
                        elem['element_nums'],
                        elem['fe_descriptor'],
                        elem['phys_table'],
                        elem['mat_table'],
                        elem['color'],
                        elem[elt_type],
                    ))
                    if elem['f_descriptor'] == 11:
                    # rods have to be written in 3 lines
                        fh.write('%10i%10i%10i\n' % (
                            elem['beam_orientation'],
                            elem['beam_foreend_cross'],
                            elem['beam_aftend_cross']
                        ))
                    for ii in elem['nodes_nums']:
                        #fh.write('%10i' % dset[elt_type]['nodes_nums'][i][ii])
                        fh.write(' '.join(elem))
                    fh.write('\n')
        fh.write('%6i\n' % -1)

    except:
        raise Exception('Error writing data-set #2412')


def _extract2412(block_data):
    """Extract element data - data-set 2412."""
    dset = {'type': 2412, 'all': None}
    # Define dictionary of possible elements types
    # Only 2D non-quadratic elements are supported
    elt_type_dict = {'3': 'triangle', '4': 'quad'}
    # Read data
    try:
        split_data = block_data.splitlines()
        split_data = [a.split() for a in split_data][2:]
        # Extract Records
        rec1 = np.array([])
        rec2 = []
        dataset = []
        i = 0
        while i < len(split_data):
            dict_tmp = dict()
            line = split_data[i]
            dict_tmp['element_num'] = int(line[0])
            dict_tmp['f_descriptor'] = int(line[1])
            dict_tmp['phys_table'] = int(line[2])
            dict_tmp['mat_table'] = int(line[3])
            dict_tmp['color'] = int(line[4])
            dict_tmp['num_nodes'] = int(line[5])
            if dict_tmp['f_descriptor'] == 11:
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
            desc = dict_tmp['f_descriptor']
            if not desc in dset:
                dset[desc] = []
            dset[desc].append(dict_tmp)
            dataset.append(dict_tmp)
        dset['all'] = dataset
        return dset
        #test = [e['f_descriptor'] for e in dataset]
        #test = set(test)
        #test = list(test)
        desc_list = [e['f_descriptor'] for e in dataset]
        elts_types = list(set(desc_list))
        for elt_type in elts_types:
            ind = list(np.where(np.array(desc_list) == elt_type)[0])
            dict_tmp = dict()
            #test = [e for dataset if ]    #dataset[ind]
            test = dataset[ind]
            dict_tmp['element_nums'] = [e['element_num'] for e in dataset[ind]]#]dataset[ind]['element_num'].copy()
            dict_tmp['fe_descriptor'] = dataset[ind, 'fe_descriptor'].copy()
            dict_tmp['phys_table'] = dataset[ind, 'phys_table'].copy()
            dict_tmp['mat_table'] = dataset[ind, 'mat_table'].copy()
            dict_tmp['color'] = dataset[ind, 'color'].copy()
            #dict_tmp['nodes_nums'] = np.array([rec2[i] for i in ind], dtype=int).copy().reshape((-1, elt_type))
            dict_tmp['nodes_nums'] = dataset[ind, 'color'].copy()
            #dset[elt_type_dict[str(elt_type)]] = dict_tmp
            dset[elt_type] = dict_tmp
        return dataset


        #rec1 = np.array(split_data[::2], dtype=int)
        # Extract Record 2
        #rec2 = split_data[1::2]
        # Look for the different types of elements stored in the dataset
        elts_types = list(set(rec1[:,5]))
        for elt_type in elts_types:
            ind = np.where(np.array(rec1[:,5]) == elt_type)[0]
            dict_tmp = dict()
            dict_tmp['element_nums'] = rec1[ind,0].copy()
            dict_tmp['fe_descriptor'] = rec1[ind,1].copy()
            dict_tmp['phys_table'] = rec1[ind,2].copy()
            dict_tmp['mat_table'] = rec1[ind,3].copy()
            dict_tmp['color'] = rec1[ind,4].copy()
            dict_tmp['nodes_nums'] =  np.array([rec2[i] for i in ind], dtype=int).copy().reshape((-1,elt_type))
            dset[elt_type_dict[str(elt_type)]] = dict_tmp
    except:
        raise Exception('Error reading data-set #2412')
    return dset


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
        'nodes_nums': nodes_nums
        }
    
    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)

    return dataset

