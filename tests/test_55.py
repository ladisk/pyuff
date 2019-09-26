import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')
import pyuff

def test_read_write_read_given_data():
    test_read_write_read_given_data_base('./data/tree_structure_mini.uff')

def test_read_write_read_given_data_base(file='', data_at_the_end=None):
    if file=='':
        return
    #read from file
    uff_read = pyuff.UFF(file)

    a = uff_read.read_sets()
    if type(a)==list:
        a = [_ for _ in a if _['type']==55]
        a = a[0]
    #write to file
    save_to_file = './data/temp55.uff'
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
    exclude_keys = ['spec_data_type', 'version_num', 'units_description', 'data_type']
    string_keys = list(set(string_keys).union(set(labels)).difference(set(exclude_keys)))
    numeric_keys = list((set(a.keys()).difference(set(string_keys)).difference(set(exclude_keys))))
    #print(a['n_bytes'], b['n_bytes'])
    for k in numeric_keys:
        print('Testing: ', k)
        np.testing.assert_array_almost_equal(a[k], b[k], decimal=3)
    for k in string_keys:
        print('Testing string: ', k, a[k])
        np.testing.assert_string_equal(a[k], b[k])

#    print('Testing data: ')
#    np.testing.assert_array_almost_equal(a['data'], b['data'], decimal=4)

    if data_at_the_end is not None:
        print('Testing last data line: ')
        np.testing.assert_array_almost_equal(a['data'][-len(data_at_the_end):], data_at_the_end)

    
def test_write_read_55():
    save_to_file = './data/measurement.uff'
    uff_dataset_origin = pyuff.prepare_test_55(save_to_file=save_to_file)
    uff_read = pyuff.UFF(save_to_file)
    uff_dataset_read = uff_read.read_sets()
    if os.path.exists(save_to_file):
        os.remove(save_to_file)

    string_keys = ['id1']
    numeric_keys = list(set(uff_dataset_origin[0].keys()) - set(string_keys))

    for a, b in zip(uff_dataset_origin, uff_dataset_read):
        for k in numeric_keys:
            print('Testing: ', k)
            np.testing.assert_array_almost_equal(a[k], b[k], decimal=3)
        for k in string_keys:
            np.testing.assert_string_equal(a[k], b[k])

if __name__ == '__main__':
    test_read_write_read_given_data()

if __name__ == '__mains__':
    np.testing.run_module_suite()
