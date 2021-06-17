# -*- coding: utf-8 -*-

import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../../')

import pyuff

def test_read_2412():
    uff_ascii = pyuff.UFF('./data/mesh_Oros-modal_uff15_uff2412.uff')
    a = uff_ascii.read_sets(2)
    np.testing.assert_array_equal(a['quad']['nodes_nums'][0], np.array([1, 25, 48, 24]))
    np.testing.assert_array_equal(a['quad']['nodes_nums'][-1], np.array([50, 74, 73, 49]))

def test_read_write_2412_mixed():
    # Read dataset 2412 in test file
    uff_ascii = pyuff.UFF('./data/mesh_test_uff2412_mixed.uff')
    a = uff_ascii.read_sets(2)
    # Test
    np.testing.assert_array_equal(a['triangle']['nodes_nums'][-1], np.array([3, 6, 11]))
    np.testing.assert_array_equal(a['quad']['nodes_nums'][-1], np.array([3, 4, 5, 6]))
    
    # Write dataset 2412
    uff_write = pyuff.UFF('./data/tmp.uff')
    uff_write._write_set(a,'overwrite')

    # Read dataset 2412 in written file
    uff_ascii = pyuff.UFF('./data/tmp.uff')
    b = uff_ascii.read_sets(0)
    # Test
    np.testing.assert_array_equal(a['triangle']['nodes_nums'], b['triangle']['nodes_nums'])
    np.testing.assert_array_equal(a['quad']['nodes_nums'], b['quad']['nodes_nums'])


if __name__ == '__main__':
    # test_read_2412()
    test_read_write_2412_mixed()