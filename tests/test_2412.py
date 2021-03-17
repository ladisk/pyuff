# -*- coding: utf-8 -*-

import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff

def test_read_2412():
    uff_ascii = pyuff.UFF('./data/mesh_Oros-modal_uff15_uff2412.unv')
    a = uff_ascii.read_sets(2)
    np.testing.assert_array_equal(a['nodes_nums'][0], np.array([1, 25, 48, 24]))
    np.testing.assert_array_equal(a['nodes_nums'][-1],[50, 74, 73, 49])

if __name__ == '__main__':
    test_read_2412()