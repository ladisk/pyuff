import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff

def test_read_write_read_given_data():
    test_read_write_read_given_data_base('./data/beam.uff')

def test_read_write_read_given_data_base(file=''):
    if file=='':
        return
    #read from file
    uff_read = pyuff.UFF(file)

    a = uff_read.read_sets()
    if type(a)==list:
        types = np.array([_['type'] for _ in a])
        a = a[np.argwhere(types==164)[0][0]]

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

def test_write_read_164():
    save_to_file = './data/test.uff'

    dataset = pyuff.prepare_164(
        units_code=1,
        units_description='SI units',
        temp_mode=1,
        length=3.28083989501312334,
        force=2.24808943099710480e-01,
        temp=1.8,
        temp_offset=459.67)
    
    dataset_out = dataset.copy()
    if save_to_file:
        if os.path.exists(save_to_file):
            os.remove(save_to_file)
        uffwrite = pyuff.UFF(save_to_file)
        uffwrite._write_set(dataset, 'add')
    
    a = dataset_out
    uff_read = pyuff.UFF(save_to_file)
    b = uff_read.read_sets()
    if os.path.exists(save_to_file):
        os.remove(save_to_file)

    labels = [_ for _ in a.keys() if any(_[-len(w):]==w for w in ['_lab', '_name', '_description'])]
    string_keys = []
    exclude_keys = []

    string_keys = list(set(string_keys).union(set(labels)).difference(set(exclude_keys)))
    numeric_keys = list((set(a.keys()).difference(set(string_keys)).difference(set(exclude_keys))))

    for k in numeric_keys:
        print('Testing: ', k)
        np.testing.assert_array_almost_equal(a[k], b[k])
    for k in string_keys:
        np.testing.assert_string_equal(a[k], b[k])

def test_prepare_164():
    dict_164 = pyuff.prepare_164(
        units_code=1,
        units_description='SI units',
        temp_mode=1,
        length=3.28083989501312334,
        force=2.24808943099710480e-01,
        temp=1.8,
        temp_offset=459.67)
    
    x = sorted(list(dict_164.keys()))
    y = sorted(['type',
                'units_code',
                'units_description',
                'temp_mode',
                'length',
                'force',
                'temp',
                'temp_offset'])
    np.testing.assert_array_equal(x,y)

    #empty dictionary test
    x2=pyuff.prepare_164()
    if 'type' not in x2.keys():
        raise Exception('Not correct keys')
    if x2['type'] != 164:
        raise Exception('Not correct type')


if __name__ == '__mains__':
    np.testing.run_module_suite()