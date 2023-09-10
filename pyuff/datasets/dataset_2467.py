"""
Dataset 2467 (source: https://github.com/victorsndvg/FEconv/blob/master/source/unv/module_dataset_2467.f90)

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
import numpy as np

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none

def _write2467(fh, dset):
    try:

        fh.write('%6i\n%6i%74s\n' % (-1, 2467, ' '))

        for i in range(len(dset['group_ids'])):
            fh.write('%10i%10i%10i%10i%10i%10i%10i%10i\n' % (dset['group_ids'][i], dset['constraint_sets'][i], dset['restraint_sets'][i], dset['load_sets'][i], dset['dof_sets'][i], dset['temp_sets'][i], dset['contact_sets'][i], dset['num_entities'][i]))
            fh.write('%-80s\n' % (dset['group_names'][i]))

            ii = 0
            while ii < dset['num_entities'][i]:
                if dset['num_entities'][i] - ii > 1:
                    fh.write('%10i%10i%10i%10i%10i%10i%10i%10i\n' % (dset['ent_types'][i][ii], dset['ent_tags'][i][ii], dset['ent_node_ids'][i][ii], dset['ent_comp_ids'][i][ii], dset['ent_types'][i][ii+1], dset['ent_tags'][i][ii+1], dset['ent_node_ids'][i][ii+1], dset['ent_comp_ids'][i][ii+1]))
                    ii += 2
                else:
                    fh.write('%10i%10i%10i%10i\n' % (dset['ent_types'][i][ii], dset['ent_tags'][i][ii], dset['ent_node_ids'][i][ii], dset['ent_comp_ids'][i][ii]))
                    ii += 1
        fh.write('%6i\n' % -1)
    except:
        raise Exception('Error writing data-set #2467')

def _extract2467(block_data):
    '''Extract physical groups -- data-set 2467.'''
    dset = {'type': 2467}
    split_data = block_data.splitlines(True)
    split_data = [a.split() for a in split_data][2:]

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

    i = 0
    index = 0
    while i < len(split_data):
        group_ids.append(int(split_data[i][0]))
        constraint_sets.append(int(split_data[i][1]))
        restraint_sets.append(int(split_data[i][2]))
        load_sets.append(int(split_data[i][3]))
        dof_sets.append(int(split_data[i][4]))
        temp_sets.append(int(split_data[i][5]))
        contact_sets.append(int(split_data[i][6]))
        left_entities = int(split_data[i][7])
        num_entities.append(left_entities)

        group_names.append(str(split_data[i+1][0]))

        ent_types.append([])
        ent_tags.append([])
        ent_node_ids.append([])
        ent_comp_ids.append([])

        i += 2
        while left_entities:
            line = split_data[i]
            if len(line) == 8:
                ent_types[index].append(int(line[0]))
                ent_tags[index].append(int(line[1]))
                ent_node_ids[index].append(int(line[2]))
                ent_comp_ids[index].append(int(line[3]))
                ent_types[index].append(int(line[4]))
                ent_tags[index].append(int(line[5]))
                ent_node_ids[index].append(int(line[6]))
                ent_comp_ids[index].append(int(line[7]))
                left_entities -= 2
            elif len(line) == 4:
                ent_types[index].append(int(line[0]))
                ent_tags[index].append(int(line[1]))
                ent_node_ids[index].append(int(line[2]))
                ent_comp_ids[index].append(int(line[3]))
                left_entities -= 1
            else:
                raise Exception("R3 of dataset 2467 needs to contain 4 or 8 values")
            i += 1
        index += 1

    dset.update({'group_ids': group_ids, 'constraint_sets': constraint_sets, 'restraint_sets': restraint_sets, 'load_sets': load_sets, 'dof_sets': dof_sets, 'temp_sets': temp_sets, 'contact_sets': contact_sets, 'num_entities': num_entities, 'group_names': group_names, 'ent_types': ent_types, 'ent_tags': ent_tags, 'ent_node_ids': ent_node_ids, 'ent_comp_ids': ent_comp_ids})
    return dset


def prepare_2467(
        group_ids=None,
        constraint_sets=None,
        restraint_sets=None,
        load_sets=None,
        dof_sets=None,
        temp_sets=None,
        contact_sets=None,
        num_entities=None,
        group_names=None,
        ent_types=None,
        ent_tags=None,
        ent_node_ids=None,
        ent_comp_ids=None,
        return_full_dict=False):
    """Name: Coordinate Systems

    R-Record, F-Field

    :param group_id: R1 F1, group ID
    :param constraint_sets: R1 F2, active group constraint set no.
    :param restraint_sets: R1 F3, active group restraint set no.
    :param load_sets: R1 F4, active group load set no.
    :param dof_sets: R1 F5, active group dof set no.
    :param temp_sets: R1 F6, active group temperature set no.
    :param contact_sets: R1 F7, active group contact set no.
    :param num_entities: R1 F8, number of entities in group
    :param group_names: R2 F1, group Name
    :param ent_types: R3 F1(F5), entity type code
    :param ent_tags: R3 F2(F6), entity tag
    :param ent_node_ids: R3 F3(F7), entity type code
    :param ent_comp_ids: R3 F4(F8), entity component / ham id
    :param return_full_dict: If True full dict with all keys is returned, else only specified arguments are included

    Records 1-2 are repeated for each node in the model, followed by a specified number (R1F8) of Record 3's.
    """
    # **Test prepare_2467**
    #>>> save_to_file = 'test_pyuff'
    #>>> dataset = pyuff.prepare_2467(
    #>>>    Part_UID = 1,
    #>>>    Part_Name = 'None',
    #>>>    CS_sys_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    #>>>    CS_types = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #>>>    CS_colors = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    #>>>    CS_names = ['CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6', 'CS7', 'CS8', 'CS9', 'CS10'],
    #>>>    CS_matrices = [np.array([[-0.44807362, 0., 0.89399666], [-0., 1., 0.], [-0.89399666, -0., -0.44807362]]),
    #>>>                    np.array([[-0.44807362,  0.,  0.89399666], [-0.,  1.,  0.], [-0.89399666, -0., -0.44807362]]),
    #>>>                    np.array([[-0.44807362,  0.,  0.89399666], [-0.,  1.,  0.], [-0.89399666, -0., -0.44807362]]),
    #>>>                    np.array([[-0.44807362,  0.,  0.89399666], [-0.,  1., 0.], [-0.89399666, -0., -0.44807362]]),
    #>>>                    np.array([[-0.44807362,  0.,  0.89399666], [-0., 1., 0.], [-0.89399666, -0., -0.44807362]]),
    #>>>                    np.array([[-0.44807362,  0.,  0.89399666], [-0., 1., 0.], [-0.89399666, -0., -0.44807362]]),
    #>>>                    np.array([[-0.44807362,  0.,  0.89399666], [-0., 1., 0.], [-0.89399666, -0., -0.44807362]]),
    #>>>                    np.array([[-0.44807362,  0.,  0.89399666], [-0., 1., 0.], [-0.89399666, -0., -0.44807362]]),
    #>>>                    np.array([[-0.44807362,  0.,  0.89399666], [-0., 1., 0.], [-0.89399666, -0., -0.44807362]]),
    #>>>                    np.array([[-0.44807362,  0.,  0.89399666], [-0., 1., 0.], [-0.89399666, -0., -0.44807362]])])
    #>>>if save_to_file:
    #>>>    if os.path.exists(save_to_file):
    #>>>        os.remove(save_to_file)
    #>>>    uffwrite = pyuff.UFF(save_to_file)
    #>>>    uffwrite.write_sets(dataset, mode='add')
    #>>>dataset

    if np.array(group_ids).dtype != int and group_ids != None:
        raise TypeError('group_ids must be integer')
    if np.array(constraint_sets).dtype != int and constraint_sets != None:
        raise ValueError('constraint_sets must be integer')
    if np.array(restraint_sets).dtype != int and restraint_sets != None:
        raise TypeError('restraint_sets must be integer')
    if np.array(load_sets).dtype != int and load_sets != None:
        raise ValueError('load_sets must be integer')
    if np.array(dof_sets).dtype != int and dof_sets != None:
        raise TypeError('dof_sets must be integer')
    if np.array(temp_sets).dtype != int and temp_sets != None:
        raise ValueError('temp_sets must be integer')
    if np.array(contact_sets).dtype != int and contact_sets != None:
        raise TypeError('contact_sets must be integer')
    if np.array(num_entities).dtype != int and num_entities != None:
        raise ValueError('num_entities must be integer')
    if np.array(group_names).dtype != str and group_names != None:
        raise TypeError('group_names must be str')
    if type(group_names) == np.ndarray or type(group_names) == list:
        for i in group_names:
            if type(i) != np.str_:
                raise TypeError('group_names datatype must be str')
    if np.array(ent_types).dtype != int and ent_types != None:
        raise ValueError('ent_types must be integer')
    if np.array(ent_tags).dtype != int and ent_tags != None:
        raise ValueError('ent_tags must be integer')
    if np.array(ent_node_ids).dtype != int and ent_node_ids != None:
        raise ValueError('ent_node_ids must be integer')
    if np.array(ent_comp_ids).dtype != int and ent_comp_ids != None:
        raise ValueError('ent_comp_ids must be integer')

    dataset={
        'type': 2467,
        'group_ids': group_ids,
        'constraint_sets': constraint_sets,
        'restraint_sets': restraint_sets,
        'load_sets': load_sets,
        'dof_sets': dof_sets,
        'temp_sets': temp_sets,
        'contact_sets': contact_sets,
        'num_entities': num_entities,
        'group_names': group_names,
        'ent_types': ent_types,
        'ent_tags': ent_tags,
        'ent_node_ids': ent_node_ids,
        'ent_comp_ids': ent_comp_ids,
        }

    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)

    return dataset

