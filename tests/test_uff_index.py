import numpy as np

import pyuff


def _prepare_frf(node, binary=0):
    dset = pyuff.prepare_58(binary=binary, func_type=4, rsp_node=node, rsp_dir=3, ref_node=1, ref_dir=3,
                            data=np.arange(8) * (1 + 1j), x=np.arange(8.), abscissa_spacing=1,
                            id1='FRF %i' % node, rsp_ent_name='pt', ref_ent_name='pt', return_full_dict=True)
    return {key: (0 if value is None else value) for key, value in dset.items()}


def test_write_sets_indexes_the_file_once(tmp_path, monkeypatch):
    uff = pyuff.UFF(tmp_path / 'frf.uff')
    n_refresh = []
    original_refresh = pyuff.UFF.refresh

    def counting_refresh(self):
        n_refresh.append(1)
        return original_refresh(self)

    monkeypatch.setattr(pyuff.UFF, 'refresh', counting_refresh)
    uff.write_sets([_prepare_frf(node) for node in range(1, 6)], 'add')

    assert len(n_refresh) == 1
    assert uff.get_status()
    assert uff.get_n_sets() == 5
    np.testing.assert_array_equal(uff.get_set_types(), [58] * 5)


def test_read_after_writing_single_sets(tmp_path):
    uff = pyuff.UFF(tmp_path / 'frf.uff')
    for node in range(1, 4):
        uff._write_set(_prepare_frf(node), 'add')

    sets = uff.read_sets()
    assert [dset['rsp_node'] for dset in sets] == [1, 2, 3]
    assert uff.read_sets(2)['rsp_node'] == 3

