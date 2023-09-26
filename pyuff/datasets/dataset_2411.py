import numpy as np

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none

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
        
        for node,node_id in enumerate(dset['node_nums']):
            fh.write('%10i%10i%10i%10i\n' % (int(node_id), dset['def_cs'][node], dset['disp_cs'][node],
                                            dset['color'][node]))

            fh.write('%25.16e%25.16e%25.16e\n' % (dset['x'][node], dset['y'][node], dset['z'][node]))

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
        split_data = split_data.split()
        # replace to support D or d notation as an exponential notation (typically form Unigraphics IDEAS)
        values = np.asarray([float(str.replace("D", "E").replace("d", "E")) for str in split_data], 'd')
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
    """Name: Nodes - Double Precision

    R-Record, F-Field
    
    :param node_nums: R1 F1, Node label
    :param def_cs: R1 F2, Export coordinate system number
    :param disp_cs: R1 F3, Displacement coordinate system number
    :param color: R1 F4, Color
    :param x: R2 F1, Node coordinates in the part coordinate system
    :param y: R2 F2, Node coordinates in the part coordinate system
    :param z: R2 F3, Node coordinates in the part coordinate system
    :param return_full_dict: If True full dict with all keys is returned, else only specified arguments are included 

    Records 1 and 2 are repeated for each node in the model.
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
