import os

import numpy as np
import pytest

import pyuff


def _set_field(dset, field, value):
    if field == 'groups':
        dset['groups'][0]['group_name'] = value
    elif isinstance(dset[field], (list, np.ndarray)):
        dset[field] = [value] * len(dset[field])
    else:
        dset[field] = value


def _get_field(dset, field):
    if field == 'groups':
        return dset['groups'][0]['group_name']
    elif isinstance(dset[field], (list, np.ndarray)):
        return dset[field][0]
    return dset[field]


# file with the data set, data-set number, field, field width
CASES = [
    ('./data/Sample_UFF58_ascii.uff', 58, 'id1', 80),
    ('./data/Sample_UFF58_ascii.uff', 58, 'rsp_ent_name', 10),
    ('./data/Sample_UFF58_ascii.uff', 58, 'ref_ent_name', 10),
    ('./data/Sample_UFF58_ascii.uff', 58, 'ordinate_axis_units_lab', 20),
    ('./data/Sample_UFF58b_bin.uff', 58, 'abscissa_axis_lab', 20),
    ('./data/uff55_translation.uff', 55, 'id1', 80),
    ('./data/TestLab 161 164 18 15 82.uff', 18, 'cs_name', 40),
    ('./data/TestLab 161 164 18 15 82.uff', 82, 'id', 80),
    ('./data/beam.uff', 151, 'model_name', 80),
    ('./data/beam.uff', 151, 'date_db_created', 10),
    ('./data/beam.uff', 164, 'units_description', 20),
    ('./data/1858.uff', 1858, 'response_direction', 4),
    ('./data/NX simulation output.uff', 2400, 'entity_name', 80),
    ('./data/heat_engine_housing.uff', 2414, 'analysis_dataset_name', 80),
    ('./data/heat_engine_housing.uff', 2414, 'id1', 80),
    ('./data/beam.uff', 2420, 'Part_Name', 80),
    ('./data/beam.uff', 2420, 'CS_names', 80),
    ('./data/groups_test.uff', 2467, 'groups', 40),
]


@pytest.mark.parametrize('file, set_type, field, width', CASES)
def test_write_long_string(file, set_type, field, width):
    """Text longer than its field is truncated to the field width, with a warning."""
    save_to_file = './data/temp_long_string.uff'
    if os.path.exists(save_to_file):
        os.remove(save_to_file)
    uff = pyuff.UFF(file)
    n = list(uff.get_set_types()).index(set_type)
    dset = uff.read_sets(n)
    _set_field(dset, field, 'ABCDEFGHIJ' * 10)

    name = 'group_name' if field == 'groups' else field
    with pytest.warns(UserWarning, match='Data-set %i: %s is longer than %i characters' % (set_type, name, width)) as record:
        pyuff.UFF(save_to_file).write_sets(dset, mode='add')
    assert record[0].filename == __file__
    with open(save_to_file, 'rb') as f:
        lines = f.read().split(b'\n')
    if set_type == 58:
        lines = lines[:13]  # binary data may follow the header records
    b = pyuff.UFF(save_to_file).read_sets(0)
    os.remove(save_to_file)

    assert max(len(line.rstrip(b'\r')) for line in lines) <= 80
    assert _get_field(b, field) == ('ABCDEFGHIJ' * 10)[:width]


def test_write_long_entity_name_58():
    """A long entity name no longer shifts the node and direction of record 6."""
    save_to_file = './data/temp_long_string.uff'
    if os.path.exists(save_to_file):
        os.remove(save_to_file)
    dataset = pyuff.prepare_58(binary=0, func_type=4, rsp_ent_name='Accelerometer', rsp_node=12, rsp_dir=3,
                               ref_ent_name='Hammer', ref_node=1, ref_dir=3,
                               data=np.arange(5.), x=np.arange(5.), abscissa_spacing=1,
                               abscissa_spec_data_type=18, ordinate_spec_data_type=12,
                               orddenom_spec_data_type=13)
    with pytest.warns(UserWarning, match='rsp_ent_name'):
        pyuff.UFF(save_to_file).write_sets(dataset, mode='add')
    b = pyuff.UFF(save_to_file).read_sets(0)
    os.remove(save_to_file)

    assert b['rsp_ent_name'] == 'Accelerome'
    assert (b['rsp_node'], b['rsp_dir'], b['ref_node'], b['ref_dir']) == (12, 3, 1, 3)


def test_write_long_group_name_2429():
    save_to_file = './data/temp_long_string.uff'
    if os.path.exists(save_to_file):
        os.remove(save_to_file)
    dset = pyuff.prepare_group(group_number=1, group_name='ABCDEFGHIJ' * 10,
                               entity_type_code=np.array([8, 8, 8, 8]), entity_tag=np.array([1, 2, 3, 4]),
                               return_full_dict=True)
    dset['type'] = 2429
    with pytest.warns(UserWarning, match='Data-set 2429: group_name is longer than 40 characters'):
        pyuff.UFF(save_to_file).write_sets(dset, mode='add')
    with open(save_to_file) as f:
        lines = f.read().splitlines()
    os.remove(save_to_file)

    assert lines[3] == ('ABCDEFGHIJ' * 10)[:40]
