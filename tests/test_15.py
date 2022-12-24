import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff

def test_read_write_read_given_data():
    test_read_write_read_given_data_base('./data/Artemis export - Geometry RPBC_setup_05_14102016_105117.uff')

def test_read_write_read_given_data_base(file=''):
    if file=='':
        return
    #read from file
    uff_read = pyuff.UFF(file)

    a = uff_read.read_sets()
    if type(a)==list:
        types = np.array([_['type'] for _ in a])
        a = a[np.argwhere(types==15)[0][0]]

    #write to file
    save_to_file = './data/temp.uff'
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
    string_keys = []
    exclude_keys = []

    string_keys = list(set(string_keys).union(set(labels)).difference(set(exclude_keys)))
    numeric_keys = list((set(a.keys()).difference(set(string_keys)).difference(set(exclude_keys))))


    for k in numeric_keys:
        print('Testing: ', k)
        np.testing.assert_array_almost_equal(a[k], b[k], decimal=3)
    for k in string_keys:
        print('Testing string: ', k, a[k])
        np.testing.assert_string_equal(a[k], b[k])


def test_read():
    uff_ascii = pyuff.UFF('./data/Artemis export - Geometry RPBC_setup_05_14102016_105117.uff')
    a = uff_ascii.read_sets(0)
    np.testing.assert_array_equal(a['node_nums'][:4],[16.0, 17.0, 18.0, 19.0])
    x = [0.0, 1.53, 0.0, 1.53, 0.0, 1.53, 0.0, 1.53, 4.296, 5.616, 4.296]
    y = [0.0, 0.0, 3.84, 3.84, 0.0, 0.0, 3.84, 3.84, 0.382, 0.382, 1.102]
    z = [0.0, 0.0, 0.0, 0.0, 1.83, 1.83, 1.83, 1.83, 0.4]
    np.testing.assert_array_equal(a['x'][:len(x)],x)
    np.testing.assert_array_equal(a['y'][:len(y)],y)
    np.testing.assert_array_equal(a['z'][:len(z)],z)

def test_write_read():
    save_to_file = './data/nodes.uff'
    
    dataset = pyuff.prepare_15(
        node_nums=[16, 17, 18, 19, 20],
        def_cs=[11, 11, 11, 12, 12],
        disp_cs=[16, 16, 17, 18, 19],
        color=[1, 3, 4, 5, 6],
        x=[0.0, 1.53, 0.0, 1.53, 0.0],
        y=[0.0, 0.0, 3.84, 3.84, 0.0],
        z=[0.0, 0.0, 0.0, 0.0, 1.83])

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

    numeric_keys = uff_dataset_origin.keys()

    a, b = uff_dataset_origin, uff_dataset_read
    for k in numeric_keys:
        print('Testing: ', k)
        np.testing.assert_array_almost_equal(a[k], b[k])

def test_prepare_15():
    dict_15=pyuff.prepare_15(
        node_nums=[16, 17, 18, 19, 20],  
        def_cs=[11, 11, 11, 12, 12], 
        disp_cs=[16, 16, 17, 18, 19],
        color=[1, 3, 4, 5, 6],
        x=[0.0, 1.53, 0.0, 1.53, 0.0],
        y=[0.0, 0.0, 3.84, 3.84, 0.0],
        z=[0.0, 0.0, 0.0, 0.0, 1.83])
    x = sorted(list(dict_15.keys()))
    y = sorted(['type', 'node_nums', 'def_cs', 'disp_cs', 'color', 'x', 'y','z'])
    np.testing.assert_array_equal(x,y)

    #empty dictionary test
    x2=pyuff.prepare_15()
    if 'type' not in x2.keys():
        raise Exception('Not correct keys')
    if x2['type'] != 15:
        raise Exception('Not correct type')

if __name__ == '__mains__':
    np.testing.run_module_suite()