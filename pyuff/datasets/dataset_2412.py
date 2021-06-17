import numpy as np

from ..tools import UFFException, _opt_fields, _parse_header_line

def _write2412(fh, dset):
    try:
        elt_type_dict = {'triangle': 3, 'quad': 4}
        fh.write('%6i\n%6i%74s\n' % (-1, 2412, ' '))
        for elt_type in dset:
            if elt_type == "type":
                pass
            else:
                for i in range(len(dset[elt_type]['element_nums'])):
                    fh.write('%10i%10i%10i%10i%10i%10i\n' % (
                        dset[elt_type]['element_nums'][i],
                        dset[elt_type]['fe_descriptor'][i],
                        dset[elt_type]['phys_table'][i],
                        dset[elt_type]['mat_table'][i],
                        dset[elt_type]['color'][i],
                        elt_type_dict[elt_type],
                    ))
                    for ii in range(elt_type_dict[elt_type]):
                        fh.write('%10i' % dset[elt_type]['nodes_nums'][i][ii])
                    fh.write('\n')
        fh.write('%6i\n' % -1)

    except:
        raise UFFException('Error writing data-set #2412')


def _extract2412(blockData):
    # Extract element data - data-set 2412.
    dset = {'type': 2412}
    # Define dictionary of possible elements types
    # Only 2D non-quadratic elements are supported
    elt_type_dict = {'3': 'triangle', '4': 'quad'}
    # Read data
    try:
        splitData = blockData.splitlines()
        splitData = [a.split() for a in splitData][2:]
        # Extract Record 1
        rec1 = np.array(splitData[::2], dtype=int) 
        # Extract Record 2
        rec2 = splitData[1::2] 
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
        raise UFFException('Error reading data-set #2412')
    return dset