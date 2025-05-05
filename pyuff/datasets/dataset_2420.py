import numpy as np

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none

def get_structure_2420(raw=False):
    """(source: https://www.ceas3.uc.edu/sdrluff/"""
    out = """
Universal Dataset Number: 2420

Name:   Coordinate Systems
-----------------------------------------------------------------------

Record 1:        FORMAT (2I10)
                 Field 1       -- Part UID

Record 2:        FORMAT (40A2)
                 Field 1       -- Part Name

Record 3:        FORMAT (4I10)
                 Field 1       -- Coordinate System Label
                 Field 2       -- Coordinate System Type
                                  = 0, Cartesian
                                  = 1, Cylindrical
                                  = 2, Spherical
                 Field 3       -- Coordinate System Color

Record 4:        FORMAT (40A2)
                 Field 1       -- Coordinate System Name

Record 5:        FORMAT (1P3D25.16)
                 Field 1-3     -- Transformation Matrix Row 1

Record 6:        FORMAT (1P3D25.16)
                 Field 1-3     -- Transformation Matrix Row 2

Record 7:        FORMAT (1P3D25.16)
                 Field 1-3     -- Transformation Matrix Row 3

Record 8:        FORMAT (1P3D25.16)
                 Field 1-3     -- Transformation Matrix Row 4

Records 3 thru 8 are repeated for each Coordinate System in the Part.

-----------------------------------------------------------------------
"""

    if raw:
        return out
    else:
        print(out)   

def _write2420(fh, dset):
    try:
        dict = {'Part_UID': 1,
                'Part_Name': 'None',
                'CS_types': [0],
                'CS_colors': [8]}
        dset = _opt_fields(dset, dict)
        fh.write('%6i\n%6i%74s\n' % (-1, 2420, ' '))
        fh.write('%10i\n' % (dset['Part_UID']))
        fh.write('%-80s\n' % (dset['Part_Name']))

        # -- Check Dimensions of input arrays
        num_CS = len(dset['CS_sys_labels'])
        if not all([len(e) == num_CS for e in [dset['CS_types'], dset['CS_colors'], dset['CS_names'], dset['CS_matrices']]]):
            raise IndexError("Values missing for one or more CS")
        for node in range(num_CS):
            fh.write('%10i%10i%10i\n' % (dset['CS_sys_labels'][node], dset['CS_types'][node], dset['CS_colors'][node]))
            fh.write('%s\n' % dset['CS_names'][node])
            fh.write('%25.16e%25.16e%25.16e\n' % tuple(dset['CS_matrices'][node][0, :]))
            fh.write('%25.16e%25.16e%25.16e\n' % tuple(dset['CS_matrices'][node][1, :]))
            fh.write('%25.16e%25.16e%25.16e\n' % tuple(dset['CS_matrices'][node][2, :]))
            fh.write('%25.16e%25.16e%25.16e\n' % tuple(dset['CS_matrices'][node][3, :]))
        fh.write('%6i\n' % -1)
    except:
        raise Exception('Error writing data-set #2420')

def _extract2420(block_data):
    '''Extract local CS/transforms -- data-set 2420.'''
    dset = {'type': 2420}
    #        try:
    split_data = block_data.splitlines(True)

    # -- Get Record 1
    dset['Part_UID'] = int(split_data[2])

    # -- Get Record 2
    dset['Part_Name'] = split_data[3].rstrip()

    # -- Get Record 3
    rec_3 = list(map(int, ''.join(split_data[4::6]).split()))
    dset['CS_sys_labels'] = rec_3[::3]
    dset['CS_types'] = rec_3[1::3]
    dset['CS_colors'] = rec_3[2::3]

    # -- Get Record 4
    dset['CS_names'] = list(map(str.rstrip, split_data[5::6]))

    # !! The following part should be made smoother
    # -- Get Record 5
    def replace_d_float(s):
        return float(s.replace('d', 'e').replace('D', 'E'))
    row1 = list(map(replace_d_float, ''.join(split_data[6::6]).split()))
    row2 = list(map(replace_d_float, ''.join(split_data[7::6]).split()))
    row3 = list(map(replace_d_float, ''.join(split_data[8::6]).split()))
    row4 = list(map(replace_d_float, ''.join(split_data[9::6]).split()))
    dset['CS_matrices'] = [np.vstack((row1[i:(i + 3)], row2[i:(i + 3)], row3[i:(i + 3)], row4[i:(i + 3)])) \
                            for i in np.arange(0, len(row1), 3)]

    return dset


def prepare_2420(
        Part_UID=None,
        Part_Name=None,
        CS_sys_labels=None,
        CS_types=None,
        CS_colors=None,
        CS_names=None,
        CS_matrices=None,
        return_full_dict=False):
    """Name: Coordinate Systems

    R-Record, F-Field

    :param Part_UID: R1 F1, Part UID
    :param Part_Name: R2 F1, Part Name
    :param CS_sys_labels: R3 F1, Coordinate System Label
    :param CS_types: R3 F2, Coordinate System Type (0-Cartesian, 1-Cylindrical, 2-Spherical)
    :param CS_colors: R3 F3, Coordinate System Color
    :param CS_names: R4 F1, Coordinate System Name
    :param CS_matrices: R5-8 F1-3, Transformation Matrix
    :param return_full_dict: If True full dict with all keys is returned, else only specified arguments are included
    """
    # **Test prepare_2420**
    #save_to_file = 'test_pyuff'
    #dataset = pyuff.prepare_2420(
    #    Part_UID = 1,
    #    Part_Name = 'None',
    #    CS_sys_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    #    CS_types = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #    CS_colors = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    #    CS_names = ['CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6', 'CS7', 'CS8', 'CS9', 'CS10'],
    #    CS_matrices = [np.array([[-0.44807362, 0., 0.89399666], [-0., 1., 0.], [-0.89399666, -0., -0.44807362]]),
    #                    np.array([[-0.44807362,  0.,  0.89399666], [-0.,  1.,  0.], [-0.89399666, -0., -0.44807362]]),
    #                    np.array([[-0.44807362,  0.,  0.89399666], [-0.,  1.,  0.], [-0.89399666, -0., -0.44807362]]),
    #                    np.array([[-0.44807362,  0.,  0.89399666], [-0.,  1., 0.], [-0.89399666, -0., -0.44807362]]),
    #                    np.array([[-0.44807362,  0.,  0.89399666], [-0., 1., 0.], [-0.89399666, -0., -0.44807362]]),
    #                    np.array([[-0.44807362,  0.,  0.89399666], [-0., 1., 0.], [-0.89399666, -0., -0.44807362]]),
    #                    np.array([[-0.44807362,  0.,  0.89399666], [-0., 1., 0.], [-0.89399666, -0., -0.44807362]]),
    #                    np.array([[-0.44807362,  0.,  0.89399666], [-0., 1., 0.], [-0.89399666, -0., -0.44807362]]),
    #                    np.array([[-0.44807362,  0.,  0.89399666], [-0., 1., 0.], [-0.89399666, -0., -0.44807362]]),
    #                    np.array([[-0.44807362,  0.,  0.89399666], [-0., 1., 0.], [-0.89399666, -0., -0.44807362]])])
    #if save_to_file:
    #    if os.path.exists(save_to_file):
    #        os.remove(save_to_file)
    #    uffwrite = pyuff.UFF(save_to_file)
    #    uffwrite.write_sets(dataset, mode='add')
    #dataset

    if np.array(Part_UID).dtype != int and Part_UID != None:
        raise TypeError('Part_UID must be integer')
    if type(Part_Name) != str and Part_Name != None:
        raise TypeError('Part_Name must be str')
    if np.array(CS_sys_labels).dtype != int and CS_sys_labels != None:
        raise TypeError('CS_sys_labels must be integer')
    if type(CS_types) in (np.ndarray, list, tuple):
        for i in CS_types:
            if i not in (0, 1, 2):
                raise ValueError('CS_types can be 0, 1, 2')
    if np.array(CS_types).dtype != int and CS_types != None:
        raise ValueError('CS_types can be 0, 1, 2')
    if np.array(CS_colors).dtype != int and CS_colors != None:
        raise TypeError('CS_colors must be integer')
    if type(CS_names) not in (np.ndarray, list, str) and CS_names != None:
        raise TypeError('CS_names datatype must be str')
    if type(CS_names) == np.ndarray:
        for i in CS_names:
            if type(i) != np.str_: 
                raise TypeError('CS_names datatype must be str')
    if type(CS_names) == list:
        for i in CS_names:
            if type(i) != str: 
                raise TypeError('CS_names datatype must be str')
    if np.array(CS_matrices).dtype != float and CS_matrices != None:
        raise TypeError('CS_matrices must be float')

    dataset={
        'type': 2420,
        'Part_UID':Part_UID,
        'Part_Name': Part_Name,
        'CS_sys_labels': CS_sys_labels,
        'CS_types': CS_types,
        'CS_colors': CS_colors,
        'CS_names': CS_names,
        'CS_matrices': CS_matrices
        }

    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)

    return dataset

