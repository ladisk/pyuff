import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff


def test_prepare_2400():
    dataset = pyuff.prepare_2400(
        model_uid=13,
        entity_type=7,
        entity_subtype=-1,
        version_number=0,
        entity_name='Fem1',
        part_number='',
        status_mask=[0] * 32,
        date='05-MAY-25',
        time='18:05:29',
        idm_version_id=0,
        idm_item_id=0,
        primary_parent_uid=0,
        optimization_switches=0)

    assert dataset['type'] == 2400
    assert dataset['model_uid'] == 13
    assert dataset['entity_type'] == 7
    assert dataset['entity_subtype'] == -1
    assert dataset['entity_name'] == 'Fem1'
    assert dataset['status_mask'] == [0] * 32

    # empty dict test
    x = pyuff.prepare_2400()
    assert 'type' in x
    assert x['type'] == 2400


def test_read_2400():
    uff_file = pyuff.UFF('./data/NX simulation output.uff')
    sets = uff_file.read_sets()
    dset = [s for s in sets if s['type'] == 2400][0]

    assert dset['model_uid'] == 13
    assert dset['entity_type'] == 7
    assert dset['entity_subtype'] == -1
    assert dset['version_number'] == 0
    assert dset['entity_name'].strip() == 'Fem1'
    assert len(dset['status_mask']) == 32
    assert all(v == 0 for v in dset['status_mask'])
    assert dset['optimization_switches'] == 0


def test_write_read_2400():
    save_to_file = './data/temp2400.uff'
    if os.path.exists(save_to_file):
        os.remove(save_to_file)

    dataset = pyuff.prepare_2400(
        model_uid=42,
        entity_type=3,
        entity_subtype=-1,
        version_number=1,
        entity_name='TestModel',
        part_number='PART-001',
        status_mask=[0, 1, 0] + [0] * 29,
        date='27-FEB-26',
        time='12:00:00',
        idm_version_id=10,
        idm_item_id=20,
        primary_parent_uid=30,
        optimization_switches=3)

    uffwrite = pyuff.UFF(save_to_file)
    uffwrite._write_set(dataset, 'add')

    uff_read = pyuff.UFF(save_to_file)
    b = uff_read.read_sets(0)

    if os.path.exists(save_to_file):
        os.remove(save_to_file)

    assert b['type'] == 2400
    assert b['model_uid'] == 42
    assert b['entity_type'] == 3
    assert b['entity_subtype'] == -1
    assert b['version_number'] == 1
    assert b['entity_name'].strip() == 'TestModel'
    assert b['part_number'].strip() == 'PART-001'
    assert b['status_mask'] == [0, 1, 0] + [0] * 29
    assert b['date'].strip() == '27-FEB-26'
    assert b['time'].strip() == '12:00:00'
    assert b['idm_version_id'] == 10
    assert b['idm_item_id'] == 20
    assert b['primary_parent_uid'] == 30
    assert b['optimization_switches'] == 3


if __name__ == '__main__':
    test_prepare_2400()
    test_read_2400()
    test_write_read_2400()
