import numpy as np
import pyuff

def test_prepare_2411():
    dict_2411 = pyuff.prepare_2411(return_full_dict=True)

    x = sorted(list(dict_2411.keys()))
    y = sorted(['type', 'node_nums', 'def_cs', 'disp_cs', 'color', 'x', 'y', 'z'])
    np.testing.assert_array_equal(x,y)

    #empty dictionary test
    x2=pyuff.prepare_2411()
    if 'type' not in x2.keys():
        raise Exception('Not correct keys')
    if x2['type'] != 2411:
        raise Exception('Not correct type')

