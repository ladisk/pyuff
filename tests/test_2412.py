import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff

def test_read_2412():
    # -- Load small test file
    uff_ascii = pyuff.UFF('./data/mesh_Oros-modal_uff15_uff2412.uff')
    a = uff_ascii.read_sets(2)

    # -- Test legacy interface
    np.testing.assert_array_equal(a['quad']['nodes_nums'][0], np.array([1, 25, 48, 24]))
    np.testing.assert_array_equal(a['quad']['nodes_nums'][-1], np.array([50, 74, 73, 49]))
    # -- Test new interface
    np.testing.assert_array_equal(a[44][0]['nodes_nums'], np.array([1, 25, 48, 24]))
    np.testing.assert_array_equal(a[44][-1]['nodes_nums'], np.array([50, 74, 73, 49]))

    # -- Load bigger test file with rods
    uff_ascii = pyuff.UFF('./data/simcenter_exported_result.uff')
    a = uff_ascii.read_sets(2)

    # -- Test new interface
    assert all(['type' in a, 21 in a, 91 in a, 94 in a, len(a) == 4])
    np.testing.assert_array_equal(a[21][0]['nodes_nums'], np.array([1894, 1443]))
    np.testing.assert_array_equal(a[94][0]['nodes_nums'], np.array([1870, 1046, 1, 946]))
    assert sorted(list(a[21][0].keys())) == ['beam_aftend_cross', 'beam_foreend_cross', 'beam_orientation', 'color', 'element_nums', 'fe_descriptor', 'mat_table', 'nodes_nums', 'num_nodes', 'phys_table']



def test_read_write_2412_mixed():
    # Read dataset 2412 in test file
    uff_ascii = pyuff.UFF('./data/mesh_test_uff2412_mixed.uff')
    a = uff_ascii.read_sets(2)

    # Test
    np.testing.assert_array_equal(a['triangle']['nodes_nums'][-1], np.array([3, 6, 11]))
    np.testing.assert_array_equal(a['quad']['nodes_nums'][-1], np.array([3, 4, 5, 6]))
    
    # Write dataset 2412
    uff_write = pyuff.UFF('./data/tmp.uff')
    uff_write._write_set(a,'overwrite')

    # Read dataset 2412 in written file
    uff_ascii = pyuff.UFF('./data/tmp.uff')
    b = uff_ascii.read_sets(0)

    # Test
    np.testing.assert_array_equal(a[41], b[41])
    np.testing.assert_array_equal(a[44], b[44])

def test_prepare_2412():
    dict_2412 = pyuff.prepare_2412(return_full_dict=True)

    x = sorted(list(dict_2412.keys()))
    y = sorted(['beam_aftend_cross',
                'beam_foreend_cross',
                'beam_orientation',
                'color',
                'element_nums',
                'fe_descriptor',
                'mat_table',
                'nodes_nums',
                'num_nodes',
                'phys_table',
                'type'])
    np.testing.assert_array_equal(x,y)

    #empty dictionary test
    x2=pyuff.prepare_2412()
    if 'type' not in x2.keys():
        raise Exception('Not correct keys')
    if x2['type'] != 2412:
        raise Exception('Not correct type')


if __name__ == '__main__':
    # test_read_2412()
    test_read_write_2412_mixed()

def test_read_write_2412_beams(tmp_path):
    # Linear beams (fe descriptor 21) have the beam record before the nodes
    a = pyuff.UFF('./data/simcenter_exported_result.uff').read_sets(2)
    path = tmp_path / 'beams.uff'
    pyuff.UFF(path).write_sets(a, 'add')
    b = pyuff.UFF(path).read_sets(0)

    for desc in (21, 91, 94):
        assert b[desc] == a[desc]
    assert b[21][0]['beam_orientation'] == a[21][0]['beam_orientation']


def test_read_write_2412_more_than_8_nodes(tmp_path):
    # 10-node tetrahedron (fe descriptor 118): node labels in lines of at most 8
    el = {'element_nums': 1, 'fe_descriptor': 118, 'phys_table': 1, 'mat_table': 1, 'color': 1,
          'num_nodes': 10, 'nodes_nums': list(range(1, 11))}
    quad = {'element_nums': 2, 'fe_descriptor': 44, 'phys_table': 1, 'mat_table': 1, 'color': 1,
            'num_nodes': 4, 'nodes_nums': [1, 2, 3, 4]}
    path = tmp_path / 'tet10.uff'
    pyuff.UFF(path).write_sets({'type': 2412, 118: [el], 44: [quad]}, 'add')

    lines = open(path).read().splitlines()
    assert lines[3] == '%10i' * 8 % tuple(range(1, 9))
    assert lines[4] == '%10i' * 2 % (9, 10)
    b = pyuff.UFF(path).read_sets(0)
    assert b[118] == [el]
    assert b[44] == [quad]


def test_read_2412_node_labels_in_one_line(tmp_path):
    # pyuff 2.6.1 and older wrote all node labels of an element in one line
    text = ('    -1\n  2412\n' + '%10i' * 6 % (1, 118, 1, 1, 1, 10) + '\n' + '%10i' * 10 % tuple(range(1, 11)) + '\n'
            + '%10i' * 6 % (2, 44, 1, 1, 1, 4) + '\n' + '%10i' * 4 % (1, 2, 3, 4) + '\n    -1\n')
    path = tmp_path / 'one_line.uff'
    with open(path, 'w') as fh:
        fh.write(text)
    b = pyuff.UFF(path).read_sets(0)
    assert b[118][0]['nodes_nums'] == list(range(1, 11))
    assert b[44][0]['nodes_nums'] == [1, 2, 3, 4]
