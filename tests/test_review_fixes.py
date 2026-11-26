"""Regression tests for the issues found by code review (one test per issue)."""
import numpy as np
import pytest

import pyuff
from pyuff.tools import convert_dataset_2412_to_82


def _group_2429(number, name, codes, tags, **kw):
    return pyuff.prepare_group(group_number=number, group_name=name, entity_type_code=codes,
                               entity_tag=tags, **kw)


def test_2429_entity_lengths_must_match(tmp_path):
    # 1: a header with 5 entities but 4 tags corrupted the following groups
    with pytest.raises(ValueError):
        _group_2429(1, 'g1', [7] * 5, [1, 2, 3, 4])
    bad = {'group_number': 1, 'group_name': 'g1', 'entity_type_code': [7] * 5, 'entity_tag': [1, 2, 3, 4]}
    with pytest.raises(Exception):
        pyuff.UFF(tmp_path / 'g.uff').write_sets({'type': 2429, 'groups': [bad]}, 'overwrite')


def test_convert_2412_to_82_several_datasets():
    # 3: the second data-set 2412 with quads raised AttributeError
    # 8: the identification line must not be blank
    sets = [{'type': 2412, 'quad': {'nodes_nums': [[1, 2, 3, 4]]}},
            {'type': 15},
            {'type': 2412, 'quad': {'nodes_nums': [[5, 6, 7, 8], [9, 10, 11, 12]]}}]
    traces = convert_dataset_2412_to_82(sets)
    assert len(traces) == 2
    np.testing.assert_array_equal(traces[0]['nodes'], [1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 0])
    assert len(traces[1]['nodes']) == 24 and traces[1]['nodes'][:3].tolist() == [5, 6, 0]
    assert traces[1]['trace_num'] == 2
    assert all(t['id'].strip() for t in traces)


def test_151_date_db_written_change_is_written(tmp_path):
    # 4: after the first write, a changed date_db_written was ignored
    h = pyuff.prepare_151(model_name='m', description='d', db_app='a', program='p',
                          date_db_written='01-Jan-20', time_db_written='10:00:00')
    fn = tmp_path / 'h.uff'
    pyuff.UFF(fn).write_sets(h, 'overwrite')
    h['date_db_written'] = '02-Feb-21'
    pyuff.UFF(fn).write_sets(h, 'overwrite')
    assert pyuff.UFF(fn).read_sets(0)['date_file_written'] == '02-Feb-21'


def test_read_error_message(tmp_path):
    # 5: the error message of a failed read raised TypeError (str + int)
    fn = tmp_path / 'f.uff'
    pyuff.UFF(fn).write_sets(pyuff.prepare_82(trace_num=1, color=1, id='t', nodes=[1, 2, 0]), 'overwrite')
    uff = pyuff.UFF(fn)
    uff._dataset_ind[0, 0] = -10  # seek to a negative offset fails
    with pytest.raises(Exception, match='Error reading data-set #: 0'):
        uff._read_set(0)


def test_2429_last_group_without_entities(tmp_path):
    # 6: a last group with no entities was not read
    groups = [_group_2429(1, 'g1', np.array([7, 7]), np.array([1, 2])),
              _group_2429(2, 'empty', np.array([], dtype=int), np.array([], dtype=int))]
    fn = tmp_path / 'g.uff'
    pyuff.UFF(fn).write_sets(pyuff.prepare_2429(groups), 'overwrite')
    read = pyuff.UFF(fn).read_sets(0)
    assert [g['group_name'] for g in read['groups']] == ['g1', 'empty']
    assert len(read['groups'][1]['entity_tag']) == 0


def test_82_count_matches_written_values(tmp_path):
    # 7: record 1 gave 9 nodes, but 10 values were written
    nodes = [1, 2, 3, 4, 5, 6, 7, 8, 0, 0]
    fn = tmp_path / 't.uff'
    pyuff.UFF(fn).write_sets(pyuff.prepare_82(trace_num=1, color=1, id='t', nodes=nodes), 'overwrite')
    with open(fn) as f:
        lines = f.read().splitlines()
    n_nodes = int(lines[2][10:20])
    n_values = len(' '.join(lines[4:-1]).split())
    assert n_nodes == n_values == 9
    np.testing.assert_array_equal(pyuff.UFF(fn).read_sets(0)['nodes'], nodes[:9])


def test_82_padding_zeros_are_not_nodes():
    # 7 (cont.): zeros filling the last line in files of other programs are not nodes
    read = pyuff.UFF('./data/TestLab 161 164 18 15 82.uff').read_sets(4)
    assert len(read['nodes']) == read['n_nodes']


@pytest.mark.parametrize('dataset', [2429, 2467])
def test_active_set_none_is_default(dataset):
    # 9: an active set number given as None (as with return_full_dict) failed prepare_*
    if dataset == 2429:
        g = pyuff.prepare_group(1, 'g', [7], [1], return_full_dict=True)
        g['active_dof_set_no_for_group'] = None
        assert pyuff.prepare_2429([g])['type'] == 2429
    else:
        g = pyuff.datasets.dataset_2467.prepare_group(1, 'g', [7], [1], [0], [0], return_full_dict=True)
        g['active_dof_set_no_for_group'] = None
        assert pyuff.prepare_2467([g])['type'] == 2467


def test_2467_round_trip_without_empty_top_level_keys(tmp_path):
    # 10: the reader added 13 always-empty top-level lists
    g = pyuff.datasets.dataset_2467.prepare_group(1, 'g', [8, 8, 8], [1, 2, 3], [0, 0, 0], [0, 0, 0])
    fn = tmp_path / 'p.uff'
    pyuff.UFF(fn).write_sets(pyuff.prepare_2467([g]), 'overwrite')
    read = pyuff.UFF(fn).read_sets(0)
    assert set(read) == {'type', 'groups'}
    np.testing.assert_array_equal(read['groups'][0]['entity_tag'], [1, 2, 3])
    with pytest.raises(ValueError):
        pyuff.datasets.dataset_2467.prepare_group(1, 'g', [8, 8], [1], [0, 0], [0, 0])


def test_prepare_58_single_precision_data(tmp_path):
    # 11: float32/complex64 data raised ValueError ('data != None' on an array)
    data = (np.arange(10) + 1j).astype('complex64')
    dset = pyuff.prepare_58(func_type=4, rsp_node=1, rsp_dir=1, ref_node=1, ref_dir=1, data=data,
                            x=np.arange(10.), abscissa_spacing=1, ord_data_type=5)
    fn = tmp_path / 's.uff'
    pyuff.UFF(fn).write_sets(dset, 'overwrite', force_double=False)
    read = pyuff.UFF(fn).read_sets(0)
    assert read['ord_data_type'] == 5
    np.testing.assert_allclose(read['data'], data)
    with pytest.raises(TypeError):
        pyuff.prepare_58(data=np.arange(3))
