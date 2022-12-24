import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff

def test_read_58_non_ascii():
    uff_ascii = pyuff.UFF('./data/non_ascii_header.uff')
    a = uff_ascii.read_sets(0)
    #print(uff_ascii.read_sets(0)['id1'])
    np.testing.assert_string_equal(a['id1'],'ref6_23_Mar')
    np.testing.assert_array_equal(a['rsp_dir'],0)
    length = a['num_pts']
    dt = a['x'][1]
    time = np.arange(length)*dt
    np.testing.assert_array_equal(a['x'],time)
    np.testing.assert_array_equal(len(a['data']),length)
    first_last = np.array([a['data'][0],a['data'][-1]])
    np.testing.assert_array_equal(first_last,np.array([0.407994+0.j     , 3.75037 +2.93363j]))


def test_read_58_ascii():
    uff_ascii = pyuff.UFF('./data/Sample_UFF58_ascii.uff')
    a = uff_ascii.read_sets(0)
    #print(uff_ascii.read_sets(0)['id1'])
    np.testing.assert_string_equal(a['id1'],'Mic 01.0Scalar')
    np.testing.assert_array_equal(a['rsp_dir'],1)
    length = a['num_pts']
    dt = a['x'][1]
    time = np.arange(length)*dt
    np.testing.assert_array_equal(a['x'],time)
    np.testing.assert_array_equal(len(a['data']),length)
    first_last = np.array([a['data'][0],a['data'][-1]])
    np.testing.assert_array_equal(first_last,np.array([-1.47553E-02,-4.31469E-03]))

def test_read_58b_binary_vs_58_ascii():
    uff_ascii = pyuff.UFF('./data/Sample_UFF58_ascii.uff')
    a = uff_ascii.read_sets(0)
    uff_bin = pyuff.UFF('./data/Sample_UFF58b_bin.uff')
    b = uff_bin.read_sets(0)
    #print(uff_ascii.read_sets(0)['id1'])
    np.testing.assert_string_equal(a['id1'],b['id1'])
    np.testing.assert_array_equal(a['rsp_dir'],b['rsp_dir'])
    np.testing.assert_array_equal(a['x'],b['x'])
    np.testing.assert_array_almost_equal(a['data'],b['data'])

if __name__ == '__main__':
    test_read_58_non_ascii()

if __name__ == '__mains__':
    np.testing.run_module_suite()