import numpy as np
import pytest
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


def _header_55(data_ch, data_type, n_data_per_node):
    """Records 1 to 8 of a normal mode data-set 55."""
    ids = ''.join('%-80s\n' % s for s in ['id1', 'id2', 'id3', 'id4', 'id5'])
    return ('    -1\n    55\n' + ids
            + '%10i%10i%10i%10i%10i%10i\n' % (1, 2, data_ch, 8, data_type, n_data_per_node)
            + '%10i%10i%10i%10i\n' % (2, 4, 1, 1)
            + '%13.5e%13.5e%13.5e%13.5e\n' % (100., 1., 0., 0.))


def _write_text(path, text):
    with open(path, 'w') as fh:
        fh.write(text)
    return path


def test_read_55_one_value_per_node(tmp_path):
    nodes = [(1, 0.1), (2, 0.2), (3, 0.3)]
    text = _header_55(1, 2, 1) + ''.join('%10i\n%13.5e\n' % node for node in nodes) + '    -1\n'
    dset = pyuff.UFF(_write_text(tmp_path / 'one.unv', text)).read_sets(0)

    np.testing.assert_array_equal(dset['node_nums'], [1, 2, 3])
    np.testing.assert_array_equal(dset['r1'], [0.1, 0.2, 0.3])
    assert 'r2' not in dset


def test_read_55_lines_filled_to_80_characters(tmp_path):
    # e.g. written by SDynPy: every line is filled with spaces up to 80 characters
    text = _header_55(2, 2, 3) + '%-80s\n%-80s\n' % ('%10i' % 1, '%13.5e%13.5e%13.5e' % (1., 2., 3.)) + '    -1\n'
    dset = pyuff.UFF(_write_text(tmp_path / 'filled.unv', text)).read_sets(0)

    np.testing.assert_array_equal([dset['r1'], dset['r2'], dset['r3']], [[1.], [2.], [3.]])


def test_read_55_count_from_data_characteristic(tmp_path):
    # e.g. written by SDynPy: 3 complex values per node, but record 6 gives 1 value per node
    values = np.arange(1, 7) / 4
    text = _header_55(2, 5, 1) + '%10i\n' % 1 + '%13.5e' * 6 % tuple(values) + '\n' + '    -1\n'
    with pytest.warns(UserWarning, match='record 6 gives 1 values per node'):
        dset = pyuff.UFF(_write_text(tmp_path / 'count.unv', text)).read_sets(0)

    assert dset['n_data_per_node'] == 3
    np.testing.assert_array_equal([dset['r1'], dset['r2'], dset['r3']], (values[0::2] + 1j * values[1::2])[:, None])


def test_read_55_double_precision(tmp_path):
    # real, double precision (4E20.12, Fortran exponent D)
    text = (_header_55(2, 4, 3) + '%10i\n%s\n' % (7, '  1.234567890123D+00 -2.000000000000D-01  3.000000000000D+02')
            + '    -1\n')
    dset = pyuff.UFF(_write_text(tmp_path / 'real.unv', text)).read_sets(0)
    np.testing.assert_array_equal(dset['node_nums'], [7])
    np.testing.assert_array_equal([dset['r1'], dset['r2'], dset['r3']], [[1.234567890123], [-0.2], [300.]])

    # complex, double precision: 6 values in two lines
    values = np.arange(1, 7) / 3
    text = (_header_55(2, 6, 3) + '%10i\n' % 7 + '%20.11e' * 4 % tuple(values[:4]) + '\n'
            + '%20.11e' * 2 % tuple(values[4:]) + '\n' + '    -1\n')
    dset = pyuff.UFF(_write_text(tmp_path / 'complex.unv', text)).read_sets(0)
    np.testing.assert_allclose([dset['r1'], dset['r2'], dset['r3']], (values[0::2] + 1j * values[1::2])[:, None], rtol=1e-11)


def test_read_55_wrong_number_of_values(tmp_path):
    text = _header_55(2, 2, 3) + '%10i\n%13.5e%13.5e\n' % (1, 1., 2.) + '    -1\n'
    with pytest.raises(Exception) as error:
        pyuff.UFF(_write_text(tmp_path / 'wrong.unv', text)).read_sets(0)
    assert 'expected 3 values at each of the 1 nodes' in str(error.value.__cause__.__cause__)


@pytest.mark.parametrize('n_data_per_node', [1, 3, 6])
@pytest.mark.parametrize('data_type', [2, 4, 5, 6])
def test_write_read_55_values_per_node(tmp_path, n_data_per_node, data_type):
    rng = np.random.default_rng(0)
    responses = {}
    for i in range(1, n_data_per_node + 1):
        responses['r%i' % i] = rng.normal(size=5)
        if data_type in (5, 6):
            responses['r%i' % i] = responses['r%i' % i] + 1j * rng.normal(size=5)
    data = pyuff.prepare_55(model_type=1, analysis_type=2, data_ch=2, spec_data_type=8, data_type=data_type,
                            node_nums=np.arange(1, 6), load_case=1, mode_n=1, freq=10., **responses)
    path = tmp_path / 'values.uff'
    pyuff.UFF(path).write_sets(data, 'add')
    dset = pyuff.UFF(path).read_sets(0)

    assert dset['data_type'] == data_type
    assert dset['n_data_per_node'] == n_data_per_node
    assert 'r%i' % (n_data_per_node + 1) not in dset
    np.testing.assert_array_equal(dset['node_nums'], np.arange(1, 6))
    rtol = 1e-10 if data_type in (4, 6) else 1e-5
    for key, value in responses.items():
        np.testing.assert_allclose(dset[key], value, rtol=rtol)


def test_write_55_full_dict(tmp_path):
    # r4, r5 and r6 are None: three values per node
    data = pyuff.prepare_55(model_type=1, analysis_type=2, data_ch=2, spec_data_type=8, node_nums=np.arange(1, 3),
                            load_case=1, mode_n=1, freq=10., modal_m=0, modal_damp_vis=0., modal_damp_his=0.,
                            r1=np.ones(2), r2=np.ones(2), r3=np.ones(2), return_full_dict=True)
    path = tmp_path / 'full.uff'
    pyuff.UFF(path).write_sets(data, 'add')
    assert pyuff.UFF(path).read_sets(0)['n_data_per_node'] == 3


if __name__ == '__main__':
    test_read_write_read_given_data()
    test_write_read_55()

if __name__ == '__mains__':
    np.testing.run_module_suite()