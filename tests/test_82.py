import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff


def test_read_write_read_given_data():
    #read from file
    uff_read = pyuff.UFF('./data/Artemis export - Geometry RPBC_setup_05_14102016_105117.uff')
    a = uff_read.read_sets(1)

    #write to file
    save_to_file = './data/trace_lines.uff'
    if os.path.exists(save_to_file):
        os.remove(save_to_file)
    _ = pyuff.UFF(save_to_file)
    _.write_sets(a, 'add')

    #read back
    uff_read = pyuff.UFF(save_to_file)
    b = uff_read.read_sets(0)

    if os.path.exists(save_to_file):
        os.remove(save_to_file)

    string_keys = ['id']
    numeric_keys = list(set(a.keys()) - set(string_keys))

    for k in numeric_keys:
        print('Testing: ', k)
        np.testing.assert_array_almost_equal(a[k], b[k])
    for k in string_keys:
        np.testing.assert_string_equal(a[k], b[k])

def test_read_write_read_given_data2():
    datasets = [4, 5, 6]
    for ds in datasets:
        uff_read = pyuff.UFF('./data/TestLab 161 164 18 15 82.uff')
        a = uff_read.read_sets(ds)

        #write to file
        save_to_file = './data/trace_lines.uff'
        if os.path.exists(save_to_file):
            os.remove(save_to_file)
        _ = pyuff.UFF(save_to_file)
        _.write_sets(a, 'add')

        #read back
        uff_read = pyuff.UFF(save_to_file)
        b = uff_read.read_sets(0)

        if os.path.exists(save_to_file):
            os.remove(save_to_file)

        string_keys = ['id']
        numeric_keys = list(set(a.keys()) - set(string_keys))

        for k in numeric_keys:
            print('Testing: ', k)
            np.testing.assert_array_almost_equal(a[k], b[k])
        for k in string_keys:
            np.testing.assert_string_equal(a[k], b[k])
		
def test_write_read_test_data():
    save_to_file = './data/trace_lines.uff'

    dataset = pyuff.prepare_82(
        trace_num=2,
        n_nodes=7,
        color=30,
        id='Identification line',
        nodes=np.array([0, 10, 13, 14, 15, 16, 17]))
    
    dataset_out = dataset.copy()
    if save_to_file:
        if os.path.exists(save_to_file):
            os.remove(save_to_file)
        uffwrite = pyuff.UFF(save_to_file)
        uffwrite._write_set(dataset, 'add')
    
    uff_dataset_origin = dataset_out
    uff_read = pyuff.UFF(save_to_file)
    uff_dataset_read = uff_read.read_sets()
    if os.path.exists(save_to_file):
        os.remove(save_to_file)

    string_keys = ['id']
    numeric_keys = list(set(uff_dataset_origin.keys()) - set(string_keys))

    a, b = uff_dataset_origin, uff_dataset_read
    for k in numeric_keys:
        print('Testing: ', k)
        np.testing.assert_array_almost_equal(a[k], b[k])
    for k in string_keys:
        np.testing.assert_string_equal(a[k], b[k])

def test_prepare_82():
    dict_82 = pyuff.prepare_82(
        trace_num=2,
        n_nodes=7,
        color=30,
        id='Identification line',
        nodes=np.array([0, 10, 13, 14, 15, 16, 17]))
    x = sorted(list(dict_82.keys()))
    y=sorted(['type', 'trace_num', 'n_nodes', 'color', 'id', 'nodes'])
    np.testing.assert_array_equal(x,y)

    #empty dictionary test
    x2=pyuff.prepare_82()
    if 'type' not in x2.keys():
        raise Exception('Not correct keys')
    if x2['type'] != 82:
        raise Exception('Not correct type')

if __name__ == '__mains__':
    np.testing.run_module_suite()