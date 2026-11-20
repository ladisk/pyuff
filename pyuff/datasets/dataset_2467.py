import numpy as np
import math

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none, _fit_string

def get_structure_2467(raw=False):
    """(source: https://github.com/victorsndvg/FEconv/blob/master/source/unv/module_dataset_2467.f90)"""
    out = """
Universal Dataset Number: 2467

Name:   Permanent Groups
Record 1:        FORMAT(8I10)
                Field 1       -- group number
                Field 2       -- active constraint set no. for group
                Field 3       -- active restraint set no. for group
                Field 4       -- active load set no. for group
                Field 5       -- active dof set no. for group
                Field 6       -- active temperature set no. for group
                Field 7       -- active contact set no. for group
                Field 8       -- number of entities in group

Record 2:        FORMAT(20A2)
                Field 1       -- group name

Record 3-N:      FORMAT(8I10)
                Field 1       -- entity type code
                Field 2       -- entity tag
                Field 3       -- entity node leaf id.
                Field 4       -- entity component/ ham id.
                Field 5       -- entity type code
                Field 6       -- entity tag
                Field 7       -- entity node leaf id.
                Field 8       -- entity component/ ham id.

Repeat record 3 for all entities as defined by record 1, field 8.
Records 1 thru n are repeated for each group in the model.
Entity node leaf id. and the component/ ham id. are zero for all
entities except "reference point", "reference point series"
and "coordinate system".

Example:

    -1
2467
        11         0         0         0         0         0         0         3
Group_1
        8         1         0         0         8         2         0         0
        8         3         0         0
    -1
 """
    if raw:
        return out
    else:
        print(out)   
def _write2467(fh, dset):
    try:
        dict = {'active_constraint_set_no_for_group': 0,
                'active_restraint_set_no_for_group': 0,
                'active_load_set_no_for_group': 0,
                'active_dof_set_no_for_group': 0,
                'active_temperature_set_no_for_group': 0,
                'active_contact_set_no_for_group': 0}

        dset = _opt_fields(dset, dict)
        fh.write('%6i\n%6i\n' % (-1, 2467))

        for group in dset['groups']:
            # Record 1
            # Field 8 (last field) contains the number or entities in the group
            fh.write('%10i%10i%10i%10i%10i%10i%10i%10i\n' % (group['group_number'],
                                                             group['active_constraint_set_no_for_group'],
                                                             group['active_restraint_set_no_for_group'],
                                                             group['active_load_set_no_for_group'],
                                                             group['active_dof_set_no_for_group'],
                                                             group['active_temperature_set_no_for_group'],
                                                             group['active_contact_set_no_for_group'],
                                                             len(group['entity_type_code'])))
            # Record 2
            fh.write('%-40s\n' % (_fit_string(group['group_name'], 40, 'group_name', 2467)))

            # Record 3-N
            # Write the full lines (which have 4 pairs)
            end = len(group['entity_type_code'])
            for i in range(0, end, 2):
                if end-i > 1:
                    fh.write('%10i%10i%10i%10i%10i%10i%10i%10i\n' % (group['entity_type_code'][i],
                                                                     group['entity_tag'][i],
                                                                     group['entity_node_leaf_id'][i],
                                                                     group['entity_component_id'][i],
                                                                     group['entity_type_code'][i+1],
                                                                     group['entity_tag'][i+1],
                                                                     group['entity_node_leaf_id'][i+1],
                                                                     group['entity_component_id'][i+1]))
                else:
                    fh.write('%10i%10i%10i%10i\n' % (group['entity_type_code'][i],
                                                     group['entity_tag'][i],
                                                     group['entity_node_leaf_id'][i],
                                                     group['entity_component_id'][i]))
        fh.write('%6i\n' % -1)

    except:
        raise Exception('Error writing data-set #2467')

def _extract2467(block_data):
    '''Extract physical groups -- data-set 2467.'''
    dset = {'type': 2467, 'groups': []}
    split_data = block_data.splitlines(True)[2:]

    group_ids = []
    constraint_sets = []
    restraint_sets = []
    load_sets = []
    dof_sets = []
    temp_sets = []
    contact_sets = []
    num_entities = []
    group_names = []

    ent_types = []
    ent_tags = []
    ent_node_ids = []
    ent_comp_ids = []

    lineIndex = 0
    while lineIndex < len(split_data):
        group = {}
        group.update(
            _parse_header_line(split_data[lineIndex], 8, [10, 10, 10, 10, 10, 10, 10, 10], [2, 2, 2, 2, 2, 2, 2, 2],
                               ['group_number', 'active_constraint_set_no_for_group',
                                'active_restraint_set_no_for_group', 'active_load_set_no_for_group',
                                'active_dof_set_no_for_group', 'active_temperature_set_no_for_group',
                                'active_contact_set_no_for_group', 'number_of_entities_in_group']))
        group.update(_parse_header_line(split_data[lineIndex + 1], 1, [40], [1], ['group_name']))
        indexLastLineForGroup = math.ceil(group['number_of_entities_in_group'] / 2) + lineIndex + 2
        # split all lines and then each line in separate integers. Put this in a ndarray
        values = [[int(elem) for elem in line.split()] for line in split_data[lineIndex + 2: indexLastLineForGroup]]
        # flatten the list and put in ndarray
        values = np.array([item for sublist in values for item in sublist], dtype=int)
        group['entity_type_code'] = np.array(values[::4].copy(), dtype=int)
        group['entity_tag'] = np.array(values[1::4].copy(), dtype=int)
        group['entity_node_leaf_id'] = np.array(values[2::4].copy(), dtype=int)
        group['entity_component_id'] = np.array(values[3::4].copy(), dtype=int)
        dset['groups'].append(group)  # dset is a dictionary, but 'groups' is a list

        lineIndex = indexLastLineForGroup

    dset.update({'group_ids': group_ids, 'active_constraint_set_no_for_group': constraint_sets, 'active_restraint_set_no_for_group': restraint_sets, 'active_load_set_no_for_group': load_sets, 'active_dof_set_no_for_group': dof_sets, 'active_temperature_set_no_for_group': temp_sets, 'active_contact_set_no_for_group': contact_sets, 'num_of_entities_in_group': num_entities, 'group_name': group_names, 'entity_type_code': ent_types, 'entity_tag': ent_tags, 'entity_node_leaf_id': ent_node_ids, 'entity_component_id': ent_comp_ids})
    return dset


def prepare_group(
        group_number,
        group_name,
        entity_type_code,
        entity_tag,
        entity_node_leaf_id,
        entity_component_id,
        active_constraint_set_no_for_group=0,
        active_restraint_set_no_for_group=0,
        active_load_set_no_for_group=0,
        active_dof_set_no_for_group=0,
        active_temperature_set_no_for_group=0,
        active_contact_set_no_for_group=0,
        return_full_dict=False):
    """Prepare one group of data-set 2467 (permanent groups) for writing.

    A group is a named set of entities of the model, e.g. the nodes of the measurement points on
    one part of a structure or the elements of a substructure. Each entity is given by an entity
    type code, an entity tag (the label of the node, element, ...), an entity node leaf id and an
    entity component id; the four entity arrays have the same length, the number of entities,
    which is taken from the length of ``entity_type_code`` when writing. The groups are
    collected in a list and passed to :func:`pyuff.prepare_2467`.

    This function is not available as ``pyuff.prepare_group``, which prepares a group of
    data-set 2429 without the node leaf and component ids; import it from
    ``pyuff.datasets.dataset_2467``.

    Entity type codes are those of data-set 2429 (printed by
    ``pyuff.datasets.dataset_2429.get_structure_2429()``), e.g. 7 nodes, 8 finite elements.

    Parameters
    ----------
    group_number : int
        Group number, a non-negative Python ``int`` (record 1, field 1).
    group_name : str
        Group name, not empty, up to 40 characters (record 2, field 1).
    entity_type_code : array_like of int
        Entity type code of each entity, see above (records 3 to N, fields 1 and 5).
    entity_tag : array_like of int
        Entity tag (label) of each entity (records 3 to N, fields 2 and 6).
    entity_node_leaf_id : array_like of int
        Entity node leaf id of each entity; 0 for all entities except reference points,
        reference point series and coordinate systems (records 3 to N, fields 3 and 7).
    entity_component_id : array_like of int
        Entity component (ham) id of each entity; 0 for all entities except reference points,
        reference point series and coordinate systems (records 3 to N, fields 4 and 8).
    active_constraint_set_no_for_group : int, optional
        Number of the active constraint set of the group; 0 for none (record 1, field 2).
        Default: 0.
    active_restraint_set_no_for_group : int, optional
        Number of the active restraint set of the group; 0 for none (record 1, field 3).
        Default: 0.
    active_load_set_no_for_group : int, optional
        Number of the active load set of the group; 0 for none (record 1, field 4). Default: 0.
    active_dof_set_no_for_group : int, optional
        Number of the active DOF set of the group; 0 for none (record 1, field 5). Default: 0.
    active_temperature_set_no_for_group : int, optional
        Number of the active temperature set of the group; 0 for none (record 1, field 6).
        Default: 0.
    active_contact_set_no_for_group : int, optional
        Number of the active contact set of the group; 0 for none (record 1, field 7).
        Default: 0.
    return_full_dict : bool, optional
        If True, the dictionary has all keys of the data-set, with ``None`` for
        the values not given; if False (default), only the given values.

    Returns
    -------
    dict
        One group, to be put in the list of groups passed to :func:`pyuff.prepare_2467`.

    Examples
    --------
    >>> import pyuff
    >>> from pyuff.datasets.dataset_2467 import prepare_group
    >>> sensors = prepare_group(
    ...     group_number=1,
    ...     group_name='Sensors',
    ...     entity_type_code=[7, 7, 7],
    ...     entity_tag=[1, 2, 3],
    ...     entity_node_leaf_id=[0, 0, 0],
    ...     entity_component_id=[0, 0, 0])
    >>> groups = pyuff.prepare_2467([sensors])
    >>> pyuff.UFF('sensor_group_2467.uff').write_sets(groups, mode='overwrite')
    """

    if type(group_number) != int:
        raise TypeError('group_number must be integer')
    if type(group_name) != str:
        raise TypeError('group_name must be string')
    if np.array(entity_type_code).dtype != int:
        raise TypeError('entity_type_code must all be positive integers')
    if np.array(entity_tag).dtype != int:
        raise TypeError('entity_tag must all be positive integers')
    if np.array(entity_node_leaf_id).dtype != int:
        raise TypeError('entity_node_leaf_id must all be positive integers')
    if np.array(entity_component_id).dtype != int:
        raise TypeError('entity_component_id must all be positive integers')
    if type(active_constraint_set_no_for_group) != int:
        raise TypeError('active_constraint_set_no_for_group must be integer')
    if type(active_restraint_set_no_for_group) != int:
        raise TypeError('active_restraint_set_no_for_group must be integer')
    if type(active_load_set_no_for_group) != int:
        raise TypeError('active_load_set_no_for_group must be integer')
    if type(active_dof_set_no_for_group) != int:
        raise TypeError('active_dof_set_no_for_group must be integer')
    if type(active_temperature_set_no_for_group) != int:
        raise TypeError('active_temperature_set_no_for_group must be integer')
    if type(active_contact_set_no_for_group) != int:
        raise TypeError('active_contact_set_no_for_group must be integer')

    if group_number < 0:
        raise ValueError('group_number needs to be a positive integer')
    if group_name == '':
        raise ValueError('group_name needs to be a non emtpy string')
    if active_constraint_set_no_for_group < 0:
        raise ValueError('active_constraint_set_no_for_group needs to be a positive integer')
    if active_restraint_set_no_for_group < 0:
        raise ValueError('active_restraint_set_no_for_group needs to be a positive integer')
    if active_load_set_no_for_group < 0:
        raise ValueError('active_load_set_no_for_group needs to be a positive integer')
    if active_dof_set_no_for_group < 0:
        raise ValueError('active_dof_set_no_for_group needs to be a positive integer')
    if active_temperature_set_no_for_group < 0:
        raise ValueError('active_temperature_set_no_for_group needs to be a positive integer')
    if active_contact_set_no_for_group < 0:
        raise ValueError('active_contact_set_no_for_group needs to be a positive integer')

    group = {
        'group_number': group_number,
        'group_name': group_name,
        'entity_type_code': entity_type_code,
        'entity_tag': entity_tag,
        'entity_node_leaf_id': entity_node_leaf_id,
        'entity_component_id': entity_component_id,
        'active_constraint_set_no_for_group': active_constraint_set_no_for_group,
        'active_restraint_set_no_for_group': active_restraint_set_no_for_group,
        'active_load_set_no_for_group': active_load_set_no_for_group,
        'active_dof_set_no_for_group': active_dof_set_no_for_group,
        'active_temperature_set_no_for_group': active_dof_set_no_for_group,
        'active_contact_set_no_for_group': active_dof_set_no_for_group,
    }

    if return_full_dict is False:
        group = check_dict_for_none(group)

    return group

def prepare_2467(
        groups,
        return_full_dict = False):
    """Prepare data-set 2467 (permanent groups) for writing.

    The data-set holds named groups of entities of the model, e.g. the nodes of the measurement
    points on one part of a structure or the elements of a substructure. It is the newer form of
    data-set 2429: each entity has, besides the entity type code and the entity tag (the label of
    the node, element, ...), an entity node leaf id and an entity component id; they are written
    2 entities per line. Entity type codes are those of data-set 2429, e.g. 7 nodes, 8 finite
    elements.

    Each group is a dictionary, prepared with
    :func:`pyuff.datasets.dataset_2467.prepare_group` or returned by
    :meth:`pyuff.UFF.read_sets`, with the keys ``'group_number'``, ``'group_name'``,
    ``'entity_type_code'``, ``'entity_tag'``, ``'entity_node_leaf_id'``,
    ``'entity_component_id'`` and the six ``'active_..._set_no_for_group'`` keys. All keys are
    required when writing; groups from ``pyuff.prepare_group`` (data-set 2429) cannot be written
    as data-set 2467.

    Parameters
    ----------
    groups : list of dict
        The permanent groups, also a single group must be in a list (records 1 to N, repeated
        for each group).
    return_full_dict : bool, optional
        If True, the dictionary has all keys of the data-set, with ``None`` for
        the values not given; if False (default), only the given values.

    Returns
    -------
    dict
        The data-set, to be written with :meth:`pyuff.UFF.write_sets`.

    Examples
    --------
    A group of three nodes:

    >>> import pyuff
    >>> from pyuff.datasets.dataset_2467 import prepare_group
    >>> sensors = prepare_group(
    ...     group_number=1, group_name='Sensors',
    ...     entity_type_code=[7, 7, 7], entity_tag=[1, 2, 3],
    ...     entity_node_leaf_id=[0, 0, 0], entity_component_id=[0, 0, 0])
    >>> groups = pyuff.prepare_2467([sensors])
    >>> pyuff.UFF('groups_2467.uff').write_sets(groups, mode='overwrite')
    """
    # **Test prepare_2467**
    #>>> save_to_file = 'test_pyuff'
    #>>> myGroup1 = pyuff.prepare_group(
    #>>>     group_number = 1,
    #>>>     group_name = 'myGroup',
    #>>>     entity_type_code = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]),
    #>>>     entity_tag = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]),
    #>>>     active_constraint_set_no_for_group = 0)
    #>>> dataset = pyuff.prepare_2467(
    #>>>     groups = [myGroup1])
    #>>> if save_to_file:
    #>>>     if os.path.exists(save_to_file):
    #>>>         os.remove(save_to_file)
    #>>>     uffwrite = pyuff.UFF(save_to_file)
    #>>>     uffwrite.write_sets(dataset, mode='add')
    #>>> dataset

    if type(groups) != list:
         raise TypeError('groups must be in a list, also a single group')
    for item in groups:
        prepare_group(
            item['group_number'],
            item['group_name'],
            item['entity_type_code'],
            item['entity_tag'],
            item['active_constraint_set_no_for_group'],
            item['active_restraint_set_no_for_group'],
            item['active_load_set_no_for_group'],
            item['active_dof_set_no_for_group'],
            item['active_temperature_set_no_for_group'],
            item['active_contact_set_no_for_group'])

    dataset={
        'type': 2467,
        'groups': groups,
        }

    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)

    return dataset
