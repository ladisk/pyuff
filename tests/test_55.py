import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff

def test_read_write_read_given_data():
    test_read_write_read_given_data_base('./data/uff55 sample_test record10.uff')
    test_read_write_read_given_data_base('./data/uff55_translation.uff')
    test_read_write_read_given_data_base('./data/uff55_translation_rotation.uff')

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

    for k in numeric_keys:
        print('Testing: ', k)
        np.testing.assert_array_almost_equal(a[k], b[k], decimal=3)
    for k in string_keys:
        print('Testing string: ', k, a[k])
        np.testing.assert_string_equal(a[k], b[k])


def test_write_read_55():
    save_to_file = './data/measurement.uff'
    
    if save_to_file:
        if os.path.exists(save_to_file):
            os.remove(save_to_file)
    
    uff_datasets = []
    modes = [1, 2, 3]
    node_nums = [1, 2, 3, 4]
    freqs = [10.0, 12.0, 13.0]
    for i, b in enumerate(modes):
        mode_shape = np.random.normal(size=len(node_nums))
        name = 'TestCase'
        data=pyuff.prepare_55(
            model_type=1,
            id1='NONE',
            id2='NONE',
            id3='NONE',
            id4='NONE',
            id5='NONE',
            analysis_type=2,
            data_ch=2,
            spec_data_type=8,
            data_type=2,
            r1=mode_shape,
            r2=mode_shape,
            r3=mode_shape,
            n_data_per_node=3,
            node_nums=[1, 2, 3, 4],
            load_case=1,
            mode_n=i + 1,
            modal_m= 0,
            freq=freqs[i],
            modal_damp_vis=0.,
            modal_damp_his=0.)
        
        uff_datasets.append(data.copy())
        if save_to_file:
            uffwrite = pyuff.UFF(save_to_file)
            uffwrite._write_set(data, 'add')
    
    uff_dataset_origin = uff_datasets
    uff_read = pyuff.UFF(save_to_file)
    uff_dataset_read = uff_read.read_sets()
    if os.path.exists(save_to_file):
        os.remove(save_to_file)

    string_keys = ['id1', 'id2', 'id3', 'id4', 'id5']
    numeric_keys = list(set(uff_dataset_origin[0].keys()) - set(string_keys))

    for a, b in zip(uff_dataset_origin, uff_dataset_read):
        for k in numeric_keys:
            print('Testing: ', k)
            np.testing.assert_array_almost_equal(a[k], b[k], decimal=5)
        for k in string_keys:
            np.testing.assert_string_equal(a[k], b[k])

def test_prepare_55():
    uff_datasets = []
    modes = [1, 2, 3]
    node_nums = [1, 2, 3, 4]
    freqs = [10.0, 12.0, 13.0]
    for i, b in enumerate(modes):
        mode_shape = np.random.normal(size=len(node_nums))
        name = 'TestCase'
        data=pyuff.prepare_55(
            model_type=1,
            id1='NONE',
            id2='NONE',
            id3='NONE',
            id4='NONE',
            id5='NONE',
            analysis_type=2,
            data_ch=2,
            spec_data_type=8,
            data_type=2,
            r1=mode_shape,
            r2=mode_shape,
            r3=mode_shape,
            n_data_per_node=3,
            node_nums=[1, 2, 3, 4],
            load_case=1,
            mode_n=i + 1,
            modal_m=0,
            freq=freqs[i],
            modal_damp_vis=0.,
            modal_damp_his=0.)
        
        uff_datasets.append(data.copy())
    
    x = sorted(list(uff_datasets[0].keys()))
    y = sorted(['type',
                'id1',
                'id2',
                'id3',
                'id4',
                'id5',
                'model_type',
                'analysis_type',
                'data_ch',
                'spec_data_type',
                'data_type',
                'n_data_per_node',
                'r1',
                'r2',
                'r3',
                'load_case',
                'mode_n',
                'freq',
                'modal_m',
                'modal_damp_vis',
                'modal_damp_his',
                'node_nums'])
    np.testing.assert_array_equal(x,y)

    #empty dictionary test
    x2=pyuff.prepare_55()
    if 'type' not in x2.keys():
        raise Exception('Not correct keys')
    if x2['type'] != 55:
        raise Exception('Not correct type')


def test_write_read_55_complex_eigenvalues():
    """Test round-trip for complex eigenvalue data (analysis_type=3). Ref: GitHub issue #98."""
    save_to_file = './data/temp55_complex.uff'

    if os.path.exists(save_to_file):
        os.remove(save_to_file)

    node_nums = [1, 2, 3, 4]

    # Complex eigenvalue and modal parameters
    eig = 1.5 + 2.3j
    modal_a = 0.1 + 0.2j
    modal_b = 0.3 + 0.4j

    # Complex mode shape data
    r1 = np.array([1.0+0.5j, 2.0+1.0j, 3.0+1.5j, 4.0+2.0j])
    r2 = np.array([0.5+0.1j, 1.5+0.2j, 2.5+0.3j, 3.5+0.4j])
    r3 = np.array([0.1+0.9j, 0.2+0.8j, 0.3+0.7j, 0.4+0.6j])

    data = pyuff.prepare_55(
        model_type=1,
        id1='Complex eigenvalue test',
        id2='NONE',
        id3='NONE',
        id4='NONE',
        id5='NONE',
        analysis_type=3,
        data_ch=2,
        spec_data_type=8,
        data_type=5,
        r1=r1,
        r2=r2,
        r3=r3,
        n_data_per_node=3,
        node_nums=node_nums,
        load_case=1,
        mode_n=1,
        eig=eig,
        modal_a=modal_a,
        modal_b=modal_b)

    uffwrite = pyuff.UFF(save_to_file)
    uffwrite._write_set(data, 'add')

    uff_read = pyuff.UFF(save_to_file)
    b = uff_read.read_sets(0)

    if os.path.exists(save_to_file):
        os.remove(save_to_file)

    # Verify complex eigenvalue parameters
    np.testing.assert_almost_equal(b['eig'], eig, decimal=3)
    np.testing.assert_almost_equal(b['modal_a'], modal_a, decimal=3)
    np.testing.assert_almost_equal(b['modal_b'], modal_b, decimal=3)

    # Verify complex mode shape data
    np.testing.assert_array_almost_equal(b['r1'], r1, decimal=3)
    np.testing.assert_array_almost_equal(b['r2'], r2, decimal=3)
    np.testing.assert_array_almost_equal(b['r3'], r3, decimal=3)


if __name__ == '__main__':
    test_read_write_read_given_data()
    test_write_read_55()

if __name__ == '__mains__':
    np.testing.run_module_suite()