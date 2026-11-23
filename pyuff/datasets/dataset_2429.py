import numpy as np
import math

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none, _fit_string

def get_structure_2429(raw=False):
    """(source: https://www.ceas3.uc.edu/sdrluff/"""
    out = """
Universal Dataset Number: 2429

Name:   Permanent Groups
-----------------------------------------------------------------------

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
                 Field 3       -- entity type code
                 Field 4       -- entity tag
                 Field 5       -- entity type code
                 Field 6       -- entity tag
                 Field 7       -- entity type code
                 Field 8       -- entity tag

Repeat record 3 for all entities as defined by record 1, field 8.
Records 1 thru n are repeated for each group in the model.

          Permanent group entity type codes

    Entity Type Code        Entity Description

           2                data surface thickness
           3                force on point
           4                force on edge
           5                traction on face
           6                pressure on face
           7                nodes
           8                finite elements
           9                dof sets, dof entities
          10                constraint sets, coupled dofs
          11                constraint sets, mpc equations
          12                restraint sets, nodal displacements
          13                restraint sets, nodal temperatures
          14                load sets, nodal forces
          15                load sets, nodal temperatures
          16                load sets, nodal heat sources/sinks
          17                load sets, face pressures
          18                load sets, edge pressures
          19                load sets, face heat fluxes
          20                load sets, edge heat fluxes
          21                load sets, face heat convections
          22                load sets, edge heat convections
          23                load sets, face heat radiations
          24                load sets, edge heat radiations
          25                load sets, element heat generations
          26                load sets, beam temperatures
          27                trace lines
          28                beam force
          29                beam distributed load
          30                data surface
          31                data curve
          32                displacement on point (restraint)
          33                displacement on edge (restraint)
          34                displacement on surface (restraint)
          35                temperature on point (restraint)
          36                temperature on edge (restraint) 
          37                temperature on face (restraint)
          38                temperature on point (temperature)
          39                temperature on edge (temperature)
          40                temperature on face (temperature)
          41                heat source on point
          42                heat flux on edge
          43                convection on edge
          44                radiation on edge
          45                heat flux on face
          46                convection on face
          47                radiation on face
          48                geometry contact region
          49                fe contact region
          50                contact pair
          51                kinematic dof on point
          52                kinematic dof on edge
          53                kinematic dof on face
          54-61             geometric grouping entities

-----------------------------------------------------------------------
"""

    if raw:
        return out
    else:
        print(out)   
def _write2429(fh, dset):
    try:
        fh.write('%6i\n%6i\n' % (-1, 2429))
        # The groups are in dset['groups'], as returned by the reader and by
        # prepare_2429; a single group at the top level is accepted as well
        groups = dset['groups'] if 'groups' in dset else [dset]
        for group in groups:
            _write_group(fh, group)
        fh.write('%6i\n' % -1)

    except:
        raise Exception('Error writing data-set #2429')


def _write_group(fh, group):
    """Writes records 1 to N of one permanent group."""
    group = _opt_fields(group, {'active_constraint_set_no_for_group': 0,
                                'active_restraint_set_no_for_group': 0,
                                'active_load_set_no_for_group': 0,
                                'active_dof_set_no_for_group': 0,
                                'active_temperature_set_no_for_group': 0,
                                'active_contact_set_no_for_group': 0})
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
    fh.write('%-40s\n' % (_fit_string(group['group_name'], 40, 'group_name', 2429)))

    # Record 3-N: pairs of entity type code and tag, at most 4 pairs in a line
    if len(group['entity_tag']) != len(group['entity_type_code']):
        # record 1 gives the number of entities; the pairs must match it
        raise ValueError('entity_tag and entity_type_code of group %s differ in length'
                         % group['group_number'])
    pairs = [value for pair in zip(group['entity_type_code'], group['entity_tag']) for value in pair]
    for i in range(0, len(pairs), 8):
        fh.write('%10i' * len(pairs[i:i + 8]) % tuple(pairs[i:i + 8]) + '\n')


def _extract2429(block_data):
    """Extract  permanent groups data - data-set 2429."""
    dset = {'type': 2429, 'groups': []}
    try:
        split_data = block_data.splitlines()
        
        # there can be multiple groups in the data-set, therefore need to loop.
        lineIndex = 2
        # a group has at least records 1 and 2 (a group without entities has no more)
        while lineIndex + 1 < len(split_data) and split_data[lineIndex].strip():
            group = {}
            group.update(_parse_header_line(split_data[lineIndex], 8, [10, 10, 10, 10, 10, 10, 10, 10], [2, 2, 2, 2, 2, 2, 2, 2],
                                            ['group_number', 'active_constraint_set_no_for_group', 'active_restraint_set_no_for_group', 'active_load_set_no_for_group',
                                                'active_dof_set_no_for_group', 'active_temperature_set_no_for_group', 'active_contact_set_no_for_group', 'number_of_entities_in_group']))
            group.update(_parse_header_line(split_data[lineIndex + 1], 1, [40], [1], ['group_name']))
            # Record 1 field 8 (last field) contains the number or entities in the group
            # max 4 items per line, last line in group doesn't necessarily contain 4 elements
            indexLastLineForGroup = math.ceil(group['number_of_entities_in_group'] / 4) + lineIndex + 2
            # split all lines and then each line in separate integers. Put this in a ndarray
            values = [[int(elem) for elem in line.split()] for line in split_data[lineIndex + 2: indexLastLineForGroup]]
            # flatten the list and put in ndarray
            values = np.array([item for sublist in values for item in sublist], dtype=int)
            group['entity_type_code'] = np.array(values[::2].copy(), dtype=int)
            group['entity_tag'] = np.array(values[1::2].copy(), dtype=int)
            dset['groups'].append(group) # dset is a dictionary, but 'groups' is a list

            lineIndex = indexLastLineForGroup

    except:
        raise Exception('Error reading data-set #2429')
    return dset


def _default_0(value):
    """An optional active set number: None (missing) is the default 0, as when writing."""
    return 0 if value is None else value


def prepare_group(
        group_number,
        group_name,
        entity_type_code,
        entity_tag,
        active_constraint_set_no_for_group = 0,
        active_restraint_set_no_for_group = 0,
        active_load_set_no_for_group = 0,
        active_dof_set_no_for_group = 0,
        active_temperature_set_no_for_group = 0,
        active_contact_set_no_for_group = 0,
        return_full_dict = False):
    """Prepare one group of data-set 2429 (permanent groups) for writing.

    A group is a named set of entities of the model, e.g. the nodes of the measurement points on
    one part of a structure or the elements of a substructure. Each entity is given by an entity
    type code and an entity tag (the label of the node, element, ...); the number of entities is
    taken from the length of ``entity_type_code`` when writing. The groups are collected in a
    list and passed to :func:`pyuff.prepare_2429`.

    Entity type codes (from the specification; the full list is printed by
    ``pyuff.datasets.dataset_2429.get_structure_2429()``): 7 nodes, 8 finite elements,
    9 DOF sets, 12 restraint sets (nodal displacements), 14 load sets (nodal forces),
    27 trace lines, 30 data surfaces, 31 data curves, 54-61 geometric grouping entities.

    Parameters
    ----------
    group_number : int
        Group number, a non-negative Python ``int`` (record 1, field 1).
    group_name : str
        Group name, not empty, up to 40 characters (record 2, field 1).
    entity_type_code : array_like of int
        Entity type code of each entity, see above (records 3 to N, fields 1, 3, 5 and 7).
    entity_tag : array_like of int
        Entity tag (label) of each entity, same length as ``entity_type_code`` (records 3 to N,
        fields 2, 4, 6 and 8).
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
        One group, to be put in the list of groups passed to :func:`pyuff.prepare_2429`.

    Examples
    --------
    >>> import pyuff
    >>> sensors = pyuff.prepare_group(
    ...     group_number=1,
    ...     group_name='Sensors',
    ...     entity_type_code=[7, 7, 7],
    ...     entity_tag=[1, 2, 3])
    >>> groups = pyuff.prepare_2429([sensors])
    >>> pyuff.UFF('sensor_group.uff').write_sets(groups, mode='overwrite')
    """

    if type(group_number) != int:
         raise TypeError('group_number must be integer')
    if type(group_name) != str:
         raise TypeError('group_name must be string')
    if np.array(entity_type_code).dtype != int:
        raise TypeError('entity_type_code must all be positive integers')
    if np.array(entity_tag).dtype != int:
        raise TypeError('entity_tag must all be positive integers')
    if np.size(entity_tag) != np.size(entity_type_code):
        raise ValueError('entity_tag must have the same length as entity_type_code')
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

    group={
        'group_number': group_number,
        'group_name': group_name,
        'entity_type_code': entity_type_code,
        'entity_tag': entity_tag,
        'active_constraint_set_no_for_group': active_constraint_set_no_for_group,
        'active_restraint_set_no_for_group':  active_restraint_set_no_for_group,
        'active_load_set_no_for_group': active_load_set_no_for_group,
        'active_dof_set_no_for_group': active_dof_set_no_for_group,
        'active_temperature_set_no_for_group': active_temperature_set_no_for_group,
        'active_contact_set_no_for_group': active_contact_set_no_for_group,
        }
    
    if return_full_dict is False:
        group = check_dict_for_none(group)

    return group


def prepare_2429(
        groups,
        return_full_dict = False):
    """Prepare data-set 2429 (permanent groups) for writing.

    The data-set holds named groups of entities of the model, e.g. the nodes of the measurement
    points on one part of a structure or the elements of a substructure. Each group is a
    dictionary, prepared with :func:`pyuff.prepare_group` or returned by
    :meth:`pyuff.UFF.read_sets`, with the keys ``'group_number'``, ``'group_name'``,
    ``'entity_type_code'``, ``'entity_tag'`` and the six ``'active_..._set_no_for_group'`` keys
    (see :func:`pyuff.prepare_group`); the active set numbers are optional (default 0). The entities of a group are written as pairs of entity
    type code and entity tag (the label of the node, element, ...), 4 pairs per line.

    Entity type codes (from the specification; the full list is printed by
    ``pyuff.datasets.dataset_2429.get_structure_2429()``): 7 nodes, 8 finite elements,
    9 DOF sets, 12 restraint sets (nodal displacements), 14 load sets (nodal forces),
    27 trace lines, 30 data surfaces, 31 data curves, 54-61 geometric grouping entities.

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
    A group of nodes and a group of elements:

    >>> import pyuff
    >>> sensors = pyuff.prepare_group(
    ...     group_number=1, group_name='Sensors', entity_type_code=[7, 7, 7], entity_tag=[1, 2, 3])
    >>> plate = pyuff.prepare_group(
    ...     group_number=2, group_name='Plate', entity_type_code=[8, 8], entity_tag=[2, 3])
    >>> groups = pyuff.prepare_2429([sensors, plate])
    >>> pyuff.UFF('groups.uff').write_sets(groups, mode='overwrite')
    """
    # **Test prepare_2429**
    #>>> save_to_file = 'test_pyuff'
    #>>> myGroup1 = pyuff.prepare_group(
    #>>>     group_number = 1,
    #>>>     group_name = 'myGroup',
    #>>>     entity_type_code = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]),
    #>>>     entity_tag = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]),
    #>>>     active_constraint_set_no_for_group = 0)
    #>>> dataset = pyuff.prepare_2429(
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
        # check the group; the active set numbers are optional (default 0)
        prepare_group(
            group_number=item['group_number'],
            group_name=item['group_name'],
            entity_type_code=item['entity_type_code'],
            entity_tag=item['entity_tag'],
            active_constraint_set_no_for_group=_default_0(item.get('active_constraint_set_no_for_group')),
            active_restraint_set_no_for_group=_default_0(item.get('active_restraint_set_no_for_group')),
            active_load_set_no_for_group=_default_0(item.get('active_load_set_no_for_group')),
            active_dof_set_no_for_group=_default_0(item.get('active_dof_set_no_for_group')),
            active_temperature_set_no_for_group=_default_0(item.get('active_temperature_set_no_for_group')),
            active_contact_set_no_for_group=_default_0(item.get('active_contact_set_no_for_group')))

    dataset={
        'type': 2429,
        'groups': groups,
        }
    
    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)

    return dataset
