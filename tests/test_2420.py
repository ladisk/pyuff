import numpy as np
import pyuff

def test_prepare_2420():
    dict_2420 = pyuff.prepare_2420(return_full_dict=True)

    x = sorted(list(dict_2420.keys()))
    y = sorted(['type',
                'Part_UID',
                'Part_Name',
                'CS_sys_labels',
                'CS_types',
                'CS_colors',
                'CS_names',
                'CS_matrices'])
    np.testing.assert_array_equal(x,y)

    #empty dictionary test
    x2=pyuff.prepare_2420()
    if 'type' not in x2.keys():
        raise Exception('Not correct keys')
    if x2['type'] != 2420:
        raise Exception('Not correct type')
