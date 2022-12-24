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
