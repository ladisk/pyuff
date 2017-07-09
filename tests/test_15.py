"""
Unit test for lvm_read.py
"""

import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff

def test_read_15():
    uff_ascii = pyuff.UFF('../data/Artemis export - Geometry RPBC_setup_05_14102016_105117.uff')
    a = uff_ascii.read_sets(0)
    np.testing.assert_array_equal(a['node_nums'][:4],[16.0, 17.0, 18.0, 19.0])
    x = [0.0, 1.53, 0.0, 1.53, 0.0, 1.53, 0.0, 1.53, 4.296, 5.616, 4.296]
    y = [0.0, 0.0, 3.84, 3.84, 0.0, 0.0, 3.84, 3.84, 0.382, 0.382, 1.102]
    z = [0.0, 0.0, 0.0, 0.0, 1.83, 1.83, 1.83, 1.83, 0.4]
    np.testing.assert_array_equal(a['x'][:len(x)],x)
    np.testing.assert_array_equal(a['y'][:len(y)],y)
    np.testing.assert_array_equal(a['z'][:len(z)],z)


if __name__ == '__mains__':
    np.testing.run_module_suite()