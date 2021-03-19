# -*- coding: utf-8 -*-

import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff

def test_read_2412():
    uff_ascii = pyuff.UFF('./data/mesh_Oros-modal_uff15_uff2412.unv')
    a = uff_ascii.read_sets(2)
    np.testing.assert_array_equal(a['quad']['nodes_nums'][0], np.array([1, 25, 48, 24]))
    np.testing.assert_array_equal(a['quad']['nodes_nums'][-1], np.array([50, 74, 73, 49]))

def test_read_write_2412_mixed():
    uff_ascii = pyuff.UFF('./data/mesh_test_uff2412_mixed.unv')
    a = uff_ascii.read_sets(2)
    np.testing.assert_array_equal(a['triangle']['nodes_nums'][-1], np.array([3, 6, 11]))
    np.testing.assert_array_equal(a['quad']['nodes_nums'][-1], np.array([3, 4, 5, 6]))
    
    uff_write = pyuff.UFF('./data/tmp.unv')
    uff_write._write_set(a,'add')

if __name__ == '__main__':
    # test_read_2412()
    test_read_write_2412_mixed()