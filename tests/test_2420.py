import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff

def test_prepare_2420():
    dict_2420 = pyuff.prepare_2420(return_full_dict=True)

    x = sorted(list(dict_2420.keys()))
    y = sorted(['type',
                'Part_UID',
                'Part_Name',
                'CS_sys_labels',
                'CS_types',
                'CS_colors',
                'CS_names',
                'CS_matrices'])
    np.testing.assert_array_equal(x,y)

    # -- Test empty dictionary
    x2=pyuff.prepare_2420()
    if 'type' not in x2.keys():
        raise Exception('Not correct keys')
    if x2['type'] != 2420:
        raise Exception('Not correct type')

def test_read_2420():
    uff_ascii = pyuff.UFF('./data/beam.uff')
    a = uff_ascii.read_sets(2)
    print(a)
    assert a['type'] == 2420
    assert a['Part_UID'] == 1
    assert a['Part_Name'] == 'None'
    np.testing.assert_array_equal(a['CS_sys_labels'], np.array(range(1, 11)))
    np.testing.assert_array_equal(a['CS_types'], np.array([0]*10))
    np.testing.assert_array_equal(a['CS_colors'], np.array([8]*10))
    np.testing.assert_array_equal(a['CS_names'], np.array([f'CS{i}' for i in range(1, 11)]))
    assert len(a['CS_names']) == 10
    assert np.shape(a['CS_matrices']) == (10, 4, 3)

def test_write_2420():
    # -- Prepare test dataset
    a = pyuff.prepare_2420(
            Part_UID=5,
            Part_Name='Testname',
            CS_sys_labels=[1, 2, 3, 4, 5],
            CS_types=[0]*5,
            CS_colors=[8]*5,
            CS_names=['One', 'Two', 'Three', 'Four', 'Five'],
            CS_matrices=[np.random.rand(4, 3), np.random.rand(4, 3), np.random.rand(4, 3), np.random.rand(4, 3), np.random.rand(4, 3)],
            return_full_dict=True)
    # -- Write dataset 2420
    uff_write = pyuff.UFF('./data/tmp.uff')
    uff_write._write_set(a, 'overwrite')

    # -- Read dataset 2420 in written file
    uff_ascii = pyuff.UFF('./data/tmp.uff')
    b = uff_ascii.read_sets(0)

    # -- Test equality
    for key in ['type', 'Part_UID', 'Part_Name', 'CS_sys_labels', 'CS_types', 'CS_colors', 'CS_names']:
        assert a[key] == b[key]
    np.testing.assert_array_equal(a['CS_matrices'], b['CS_matrices'])
