import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff


def test_prepare_18():
    dataset = pyuff.prepare_18(
        cs_num=[1, 2],
        cs_type=[0, 0],
        ref_cs_num=[0, 0],
        color=[1, 1],
        method=[1, 1],
        cs_name=['CS1', 'CS2'],
        ref_o=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        x_point=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        xz_point=[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])

    assert dataset['type'] == 18
    assert dataset['cs_num'] == [1, 2]
    assert dataset['cs_type'] == [0, 0]
    assert dataset['ref_cs_num'] == [0, 0]
    assert dataset['color'] == [1, 1]
    assert dataset['method'] == [1, 1]
    assert dataset['cs_name'] == ['CS1', 'CS2']
    assert dataset['ref_o'] == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert dataset['x_point'] == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    assert dataset['xz_point'] == [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]

    # empty dict test
    x = pyuff.prepare_18()
    assert 'type' in x
    assert x['type'] == 18


def test_read_18():
    uff_file = pyuff.UFF('./data/TestLab 161 164 18 15 82.uff')
    sets = uff_file.read_sets()
    dset = [s for s in sets if s['type'] == 18][0]

    assert dset['cs_num'][0] == 1.0
    assert dset['ref_cs_num'][0] == 0.0
    assert dset['cs_type'][0] == 0.0
    assert dset['color'][0] == 8.0
    assert dset['method'][0] == 1.0
    assert dset['cs_name'][0].strip() == 'SYS1'
    np.testing.assert_array_almost_equal(dset['ref_o'][0], [-2.4, -0.95, 0.0], decimal=5)
    np.testing.assert_array_almost_equal(dset['x_point'][0], [-3.4, -0.95, -8.74228e-08], decimal=5)
    np.testing.assert_array_almost_equal(dset['xz_point'][0], [-3.4, -0.95, -1.0], decimal=5)
    assert len(dset['cs_num']) == 36


def test_write_read_18():
    save_to_file = './data/temp18.uff'
    if os.path.exists(save_to_file):
        os.remove(save_to_file)

    dataset = pyuff.prepare_18(
        cs_num=[1, 2],
        cs_type=[0, 1],
        ref_cs_num=[0, 1],
        color=[8, 4],
        method=[1, 1],
        cs_name=['SYS1', 'SYS2'],
        ref_o=[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        x_point=[[1.0, 0.0, 0.0], [2.0, 2.0, 3.0]],
        xz_point=[[0.0, 0.0, 1.0], [1.0, 2.0, 4.0]])

    uffwrite = pyuff.UFF(save_to_file)
    uffwrite._write_set(dataset, 'add')

    uff_read = pyuff.UFF(save_to_file)
    b = uff_read.read_sets(0)

    if os.path.exists(save_to_file):
        os.remove(save_to_file)

    assert b['type'] == 18
    assert b['cs_num'][0] == 1.0
    assert b['cs_num'][1] == 2.0
    assert b['cs_type'][0] == 0.0
    assert b['cs_type'][1] == 1.0
    assert b['ref_cs_num'][0] == 0.0
    assert b['ref_cs_num'][1] == 1.0
    assert b['color'][0] == 8.0
    assert b['color'][1] == 4.0
    assert b['method'][0] == 1.0
    assert b['method'][1] == 1.0
    assert b['cs_name'][0].strip() == 'SYS1'
    assert b['cs_name'][1].strip() == 'SYS2'
    np.testing.assert_array_almost_equal(b['ref_o'][0], [0.0, 0.0, 0.0], decimal=5)
    np.testing.assert_array_almost_equal(b['ref_o'][1], [1.0, 2.0, 3.0], decimal=5)
    np.testing.assert_array_almost_equal(b['x_point'][0], [1.0, 0.0, 0.0], decimal=5)
    np.testing.assert_array_almost_equal(b['x_point'][1], [2.0, 2.0, 3.0], decimal=5)
    np.testing.assert_array_almost_equal(b['xz_point'][0], [0.0, 0.0, 1.0], decimal=5)
    np.testing.assert_array_almost_equal(b['xz_point'][1], [1.0, 2.0, 4.0], decimal=5)


if __name__ == '__main__':
    test_prepare_18()
    test_read_18()
    test_write_read_18()
