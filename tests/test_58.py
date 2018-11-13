#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff

def test_read_write_read_given_data():
    test_read_write_read_given_data_base('./data/BK_4_channels.uff')
    test_read_write_read_given_data_base('./data/Sample_UFF58_ascii.uff')
    test_read_write_read_given_data_base('./data/Sample_UFF58b_bin.uff')
    data_at_the_end = np.array([-5.48363E-004,-7.51019E-004,-6.07967E-004,-0.00103712])
    test_read_write_read_given_data_base('./data/no_spacing2_UFF58_ascii.uff',data_at_the_end)

def test_read_write_read_given_data_base(file='', data_at_the_end=None):
    if file=='':
        return
    #read from file
    uff_read = pyuff.UFF(file)

    a = uff_read.read_sets()
    if type(a)==list:
        a = [_ for _ in a if _['type']==58]
        a = a[0]

    #write to file
    save_to_file = './data/temp58.uff'
    if os.path.exists(save_to_file):
        os.remove(save_to_file)
    _ = pyuff.UFF(save_to_file)
    _.write_sets(a, 'add')

    #read back
    uff_read = pyuff.UFF(save_to_file)
    b = uff_read.read_sets(0)

    if os.path.exists(save_to_file):
        os.remove(save_to_file)

    labels = [_ for _ in a.keys() if any(_[-len(w):]==w for w in ['_lab', '_name', '_description'])]
    string_keys = ['id1', 'id2', 'id3', 'id4', 'id5']
    exclude_keys = ['orddenom_spec_data_type', 'abscissa_spec_data_type',
                    'spec_data_type', 'units_description',
                    'version_num']

    string_keys = list(set(string_keys).union(set(labels)).difference(set(exclude_keys)))
    numeric_keys = list((set(a.keys()).difference(set(string_keys)).difference(set(exclude_keys))))

    for k in numeric_keys:
        print('Testing: ', k)
        np.testing.assert_array_almost_equal(a[k], b[k], decimal=3)
    for k in string_keys:
        print('Testing string: ', k, a[k])
        np.testing.assert_string_equal(a[k], b[k])

    print('Testing data: ')
    np.testing.assert_array_almost_equal(a['data'], b['data'])

    if data_at_the_end is not None:
        print('Testing last data line: ')
        np.testing.assert_array_almost_equal(a['data'][-len(data_at_the_end):], data_at_the_end)



def test_write_read_58():
    save_to_file = './data/measurement.uff'
    uff_dataset_origin = pyuff.prepare_test_58(save_to_file=save_to_file)
    uff_read = pyuff.UFF(save_to_file)
    uff_dataset_read = uff_read.read_sets()
    if os.path.exists(save_to_file):
        os.remove(save_to_file)


    string_keys = ['id1', 'rsp_ent_name', 'ref_ent_name']
    numeric_keys = list(set(uff_dataset_origin[0].keys()) - set(string_keys))

    for a, b in zip(uff_dataset_origin, uff_dataset_read):
        for k in numeric_keys:
            print('Testing: ', k)
            np.testing.assert_array_almost_equal(a[k], b[k])
        for k in string_keys:
            np.testing.assert_string_equal(a[k], b[k])

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

def test_non_ascii_in_header():
    uff_ascii = pyuff.UFF('./data/Artemis export - data and dof 05_14102016_105117.uff')
    a = uff_ascii.read_sets(0)
    np.testing.assert_string_equal(a['id1'], 'Channel 1 [20, -X]')
    np.testing.assert_almost_equal(a['x'][-1], 1.007999609375e+03)
    np.testing.assert_almost_equal(np.sum(a['data']), 172027.83668466809)
    np.testing.assert_string_equal(a['ordinate_axis_units_lab'][:3], 'm/s')
    num_pts = a['num_pts']
    np.testing.assert_equal(num_pts, len(a['x']))


if __name__ == '__main__':
    test_read_58b_binary_vs_58_ascii()

if __name__ == '__mains__':
    np.testing.run_module_suite()