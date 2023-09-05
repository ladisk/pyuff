import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff

def test_read_2412():
    uff_ascii = pyuff.UFF('./data/heat_engine_housing.uff')
    a = uff_ascii.read_sets(3)
    np.testing.assert_array_equal(a[111][-1]['nodes_nums'], np.array([1, 3, 9, 10]))
    np.testing.assert_array_equal(a[91][-1]['nodes_nums'], np.array([2, 3, 8]))

def test_read_write_2412_mixed():
    # Read dataset 2412 in test file
    #uff_ascii = pyuff.UFF('./data/mesh_test_uff2412_mixed.uff')
    uff_ascii = pyuff.UFF('./data/heat_engine_housing.uff')
    a = uff_ascii.read_sets(3)
    # Test
    np.testing.assert_array_equal(a[111][-1]['nodes_nums'], np.array([1, 3, 9, 10]))
    np.testing.assert_array_equal(a[91][-1]['nodes_nums'], np.array([2, 3, 8]))
    
    # Write dataset 2412
    uff_write = pyuff.UFF('./data/tmp.uff')
    uff_write._write_set(a,'overwrite')

    # Read dataset 2412 in written file
    uff_ascii = pyuff.UFF('./data/tmp.uff')
    b = uff_ascii.read_sets(0)
    # Test
    np.testing.assert_array_equal(a[111], b[111])
    np.testing.assert_array_equal(a[91], b[91])

def test_prepare_2412():
    dict_2412 = pyuff.prepare_2412(return_full_dict=True)

    x = sorted(list(dict_2412.keys()))
    y = sorted(['beam_aftend_cross',
                'beam_foreend_cross',
                'beam_orientation',
                'color',
                'element_nums',
                'fe_descriptor',
                'mat_table',
                'nodes_nums',
                'num_nodes',
                'phys_table',
                'type'])
    np.testing.assert_array_equal(x,y)

    #empty dictionary test
    x2=pyuff.prepare_2412()
    if 'type' not in x2.keys():
        raise Exception('Not correct keys')
    if x2['type'] != 2412:
        raise Exception('Not correct type')

if __name__ == '__main__':
    # test_read_2412()
    test_prepare_2412()
    test_read_write_2412_mixed()
