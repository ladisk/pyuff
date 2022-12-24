import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff

def test_2411():
    uff_file = pyuff.UFF('./data/2411 and 2414.uff')
    uff_file.read_sets()

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

def test_write_2411():
    # Read dataset 2411 in test file
    uff_ascii = pyuff.UFF('./data/2411 and 2414.uff')
    a = uff_ascii.read_sets(1)
    # Test
    np.testing.assert_array_equal(a['node_nums'], np.arange(1,442))
    
    # Write dataset 2414
    uff_write = pyuff.UFF('./data/tmp.uff')
    uff_write._write_set(a,'overwrite')

    # Read dataset 2414 in written file
    uff_ascii = pyuff.UFF('./data/tmp.uff')
    b = uff_ascii.read_sets(0)
    # Test
    np.testing.assert_array_equal(a['x'], b['x'])
    np.testing.assert_array_equal(a['y'], b['y'])
    np.testing.assert_array_equal(a['z'], b['z'])

if __name__ == '__main__':
    test_write_2411()