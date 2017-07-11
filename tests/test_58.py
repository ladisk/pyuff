"""
Unit test for lvm_read.py
"""

import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff

def prepare_58():
    # delete prior existing files
    if os.path.exists('./data/measurement.uff'):
        os.remove('./data/measurement.uff')
    uff_dataset = []
    measurement_point_1 = np.genfromtxt('./data/meas_point_1.txt', dtype=complex)
    measurement_point_2 = np.genfromtxt('./data/meas_point_2.txt', dtype=complex)
    measurement_point_3 = np.genfromtxt('./data/meas_point_3.txt', dtype=complex)
    #measurement_point_1[0] = np.nan * (1 + 1.j) #addina np.nan for testing (should be handled ok)
    measurement = [measurement_point_1, measurement_point_2, measurement_point_3]
    print()
    binary = [0,1,0]
    for i in range(3):
        print('Adding point {}'.format(i+1))
        response_node = 1
        response_direction = 1
        reference_node = i + 1
        reference_direction = 1
        acceleration_complex = measurement[i]
        frequency = np.arange(0, 1001)
        name = 'TestCase'
        data = {'type': 58,
                'binary': binary[i],
                'func_type': 4,
                'rsp_node': response_node,
                'rsp_dir': response_direction,
                'ref_dir': reference_direction,
                'ref_node': reference_node,
                'data': acceleration_complex,
                'x': frequency,
                'id1': 'id1',
                'rsp_ent_name': name,
                'ref_ent_name': name,
                'abscissa_spacing': 1,
                'abscissa_spec_data_type': 18,
                'ordinate_spec_data_type': 12,
                'orddenom_spec_data_type': 13}
        uff_dataset.append(data.copy())
        uffwrite = pyuff.UFF('./data/measurement.uff')
        uffwrite._write_set(data, 'add')
    return uff_dataset

def test_write_read_58():
    uff_dataset_origin = prepare_58()
    uff_read = pyuff.UFF('./data/measurement.uff')
    uff_dataset_read = uff_read.read_sets()

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


if __name__ == '__mains__':
    np.testing.run_module_suite()