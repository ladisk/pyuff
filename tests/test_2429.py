import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff

def test_prepare_2429():
    mygroup = pyuff.prepare_group(
        1,
        "testGroup",
        [8, 8],
        [1, 2])
    dict_2429 = pyuff.prepare_2429([mygroup], return_full_dict=True)

    x = sorted(list(dict_2429.keys()))
    y = sorted(['type',
                'groups'])
    np.testing.assert_array_equal(x,y)

    v = sorted(list(dict_2429['groups'][0].keys()))
    w = sorted(['group_number',
                'group_name',
                'entity_type_code',
                'entity_tag',
                'active_constraint_set_no_for_group',
                'active_restraint_set_no_for_group',
                'active_load_set_no_for_group',
                'active_dof_set_no_for_group',
                'active_temperature_set_no_for_group',
                'active_contact_set_no_for_group'])
    np.testing.assert_array_equal(v,w)

    #empty dictionary test
    x2 = pyuff.prepare_2429([mygroup], return_full_dict=True)

    if 'type' not in x2.keys():
        raise Exception('Not correct keys')
    if x2['type'] != 2429:
        raise Exception('Not correct type')


def _group(number, n_entities):
    return pyuff.prepare_group(number, 'group %i' % number, [8] * n_entities, list(range(1, n_entities + 1)))


def test_write_read_2429_groups(tmp_path):
    # groups with 1 to 9 entities: full lines of 4 pairs and partial last lines
    groups = [_group(n, n) for n in range(1, 10)]
    path = tmp_path / 'groups.unv'
    pyuff.UFF(path).write_sets(pyuff.prepare_2429(groups), 'add')
    dset = pyuff.UFF(path).read_sets(0)

    assert len(dset['groups']) == 9
    for group, read in zip(groups, dset['groups']):
        assert read['group_number'] == group['group_number']
        assert read['group_name'] == group['group_name']
        assert read['number_of_entities_in_group'] == len(group['entity_tag'])
        np.testing.assert_array_equal(read['entity_type_code'], group['entity_type_code'])
        np.testing.assert_array_equal(read['entity_tag'], group['entity_tag'])

    # a data-set read from a file is written back unchanged
    path_2 = tmp_path / 'groups_2.unv'
    pyuff.UFF(path_2).write_sets(dset, 'add')
    assert open(path_2).read() == open(path).read()


def test_write_2429_single_group(tmp_path):
    # a single group at the top level, as written by pyuff 2.7.0 and older
    group = _group(1, 5)
    group['type'] = 2429
    path = tmp_path / 'group.unv'
    pyuff.UFF(path).write_sets(group, 'add')

    assert open(path).read().splitlines()[5] == '%10i%10i' % (8, 5)
    np.testing.assert_array_equal(pyuff.UFF(path).read_sets(0)['groups'][0]['entity_tag'], [1, 2, 3, 4, 5])
