import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff

def test_prepare_2467():
    mygroup = pyuff.datasets.dataset_2467.prepare_group(
        1,
        'testGroup',
        [8, 8],
        [1, 2],
        [8, 8],
        [1, 2])
    dict_2467 = pyuff.prepare_2467([mygroup], return_full_dict=True)

    x = sorted(list(dict_2467.keys()))
    y = sorted(['type',
                'groups'])
    np.testing.assert_array_equal(x,y)

    v = sorted(list(dict_2467['groups'][0].keys()))
    w = sorted(['group_number',
                'group_name',
                'entity_type_code',
                'entity_tag',
                'entity_node_leaf_id',
                'entity_component_id',
                'active_constraint_set_no_for_group',
                'active_restraint_set_no_for_group',
                'active_load_set_no_for_group',
                'active_dof_set_no_for_group',
                'active_temperature_set_no_for_group',
                'active_contact_set_no_for_group'])
    np.testing.assert_array_equal(v,w)

    #empty dictionary test
    x2 = pyuff.prepare_2467([mygroup], return_full_dict=True)

    if 'type' not in x2.keys():
        raise Exception('Not correct keys')
    if x2['type'] != 2467:
        raise Exception('Not correct type')
    
def test_write_2467():
    a = pyuff.prepare_2467(
            [
                {
                    'group_number': 1,
                    'group_name': 'One',
                    'entity_type_code': [8, 8, 8, 8, 8],
                    'entity_tag': [1, 2, 3, 4, 5],
                    'entity_node_leaf_id': [2, 2, 2, 2, 2],
                    'entity_component_id': [3, 3, 3, 3, 3],
                    'active_constraint_set_no_for_group': 5,
                    'active_restraint_set_no_for_group': 5,
                    'active_load_set_no_for_group': 5,
                    'active_dof_set_no_for_group': 5,
                    'active_temperature_set_no_for_group': 5,
                    'active_contact_set_no_for_group': 5,
                },
                {
                    'group_number': 2,
                    'group_name': 'Two',
                    'entity_type_code': [4, 4, 4, 4, 4],
                    'entity_tag': [6, 7, 8, 9, 10],
                    'entity_node_leaf_id': [2, 2, 2, 2, 2],
                    'entity_component_id': [3, 3, 3, 3, 3],
                    'active_constraint_set_no_for_group': 6,
                    'active_restraint_set_no_for_group': 6,
                    'active_load_set_no_for_group': 6,
                    'active_dof_set_no_for_group': 6,
                    'active_temperature_set_no_for_group': 6,
                    'active_contact_set_no_for_group': 6,
                }
            ],
            return_full_dict=True)

    save_to_file = './data/group_output.uff'
    if os.path.exists(save_to_file):
        os.remove(save_to_file)
    _ = pyuff.UFF(save_to_file)
    _.write_sets(a, 'add')

    with open(save_to_file, 'r') as f:
        written_data = f.read()
    comp_str = '''    -1
  2467
         1         5         5         5         5         5         5         5
One                                     
         8         1         2         3         8         2         2         3
         8         3         2         3         8         4         2         3
         8         5         2         3
         2         6         6         6         6         6         6         5
Two                                     
         4         6         2         3         4         7         2         3
         4         8         2         3         4         9         2         3
         4        10         2         3
    -1
'''
    assert written_data == comp_str

    if os.path.exists(save_to_file):
        os.remove(save_to_file)

def test_read_2467():
    uff_ascii = pyuff.UFF('./data/groups_test.uff')
    sets = uff_ascii.get_set_types()
    a = uff_ascii.read_sets(4)

    assert a['type'] == 2467
    assert len(a['groups']) == 3

    assert a['groups'][0]['number_of_entities_in_group'] == 4
    assert a['groups'][1]['number_of_entities_in_group'] == 4
    assert a['groups'][2]['number_of_entities_in_group'] == 136

    assert len(a['groups'][0]['entity_tag']) == 4
    assert len(a['groups'][1]['entity_tag']) == 4
    assert len(a['groups'][2]['entity_tag']) == 136

    assert a['groups'][0]['group_name'] == 'Left_Side'
    assert a['groups'][1]['group_name'] == 'Right_Side'
    assert a['groups'][2]['group_name'] == 'Surface'

    np.testing.assert_array_equal(a['groups'][0]['entity_tag'], np.array([110, 117, 122, 135]))
    np.testing.assert_array_equal(a['groups'][1]['entity_tag'], np.array([168, 190, 191, 192]))

def test_write_read_2467():
    groups = [
                    {
                        'group_number': 1,
                        'group_name': 'One',
                        'entity_type_code': [8, 8, 8, 8, 8],
                        'entity_tag': [1, 2, 3, 4, 5],
                        'entity_node_leaf_id': [2, 2, 2, 2, 2],
                        'entity_component_id': [3, 3, 3, 3, 3],
                        'active_constraint_set_no_for_group': 5,
                        'active_restraint_set_no_for_group': 5,
                        'active_load_set_no_for_group': 5,
                        'active_dof_set_no_for_group': 5,
                        'active_temperature_set_no_for_group': 5,
                        'active_contact_set_no_for_group': 5,
                    },
                    {
                        'group_number': 2,
                        'group_name': 'Two',
                        'entity_type_code': [4, 4, 4, 4, 4],
                        'entity_tag': [6, 7, 8, 9, 10],
                        'entity_node_leaf_id': [2, 2, 2, 2, 2],
                        'entity_component_id': [3, 3, 3, 3, 3],
                        'active_constraint_set_no_for_group': 6,
                        'active_restraint_set_no_for_group': 6,
                        'active_load_set_no_for_group': 6,
                        'active_dof_set_no_for_group': 6,
                        'active_temperature_set_no_for_group': 6,
                        'active_contact_set_no_for_group': 6,
                    }
            ]

    a = pyuff.prepare_2467(groups=groups, return_full_dict=True)
    save_to_file = './data/group_output.uff'
    if os.path.exists(save_to_file):
        os.remove(save_to_file)
    _ = pyuff.UFF(save_to_file)
    _.write_sets(a, 'add')

    uff_ascii = pyuff.UFF(save_to_file)
    groups_read = uff_ascii.read_sets(0)['groups']

    for group, group_read in zip(groups, groups_read):
        for key in group.keys():
            np.testing.assert_array_equal(group[key], group_read[key])

    if os.path.exists(save_to_file):
        os.remove(save_to_file)

if __name__ == '__main__':
    test_write_read_2467()