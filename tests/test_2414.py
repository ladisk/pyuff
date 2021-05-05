# -*- coding: utf-8 -*-
import numpy as np
import sys, os
# my_path = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, my_path + '/../')

import pyuff

def test_read_2414():
    uff_ascii = pyuff.UFF('./data/DS2414_disp_file.uff')
    a = uff_ascii.read_sets(3)
    np.testing.assert_array_equal(a['freq'], np.array([100]))
    np.testing.assert_array_equal(a['x'][3], np.array([3.33652e-09-1j*9.17913e-13]))

def test_write_2414():
    # Read dataset 2414 in test file
    uff_ascii = pyuff.UFF('./data/DS2414_disp_file.uff')
    a = uff_ascii.read_sets(3)
    # Test
    np.testing.assert_array_equal(a['freq'], np.array([100]))
    np.testing.assert_array_equal(a['x'][3], np.array([3.33652e-09-1j*9.17913e-13]))
    
    # Write dataset 2414
    uff_write = pyuff.UFF('./data/tmp.uff')
    uff_write._write_set(a,'overwrite')

    # Read dataset 2414 in written file
    uff_ascii = pyuff.UFF('./data/tmp.uff')
    b = uff_ascii.read_sets(0)
    # Test
    np.testing.assert_array_equal(a['freq'], b['freq'])
    np.testing.assert_array_equal(a['z'], b['z'])

if __name__ == '__main__':
    #test_read_2412()
    test_write_2414()