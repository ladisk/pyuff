import numpy as np
import pytest

import pyuff


def _58(**kwargs):
    return pyuff.prepare_58(func_type=4, rsp_node=1, rsp_dir=1, ref_node=1, ref_dir=1,
                            data=np.arange(10) * (1 + 1j), x=np.arange(10.), id1='id1',
                            rsp_ent_name='n', ref_ent_name='n', abscissa_spacing=1,
                            abscissa_spec_data_type=18, ordinate_spec_data_type=12,
                            **{'orddenom_spec_data_type': 13, **kwargs})


def _58_without_orddenom(**kwargs):
    # orddenom_spec_data_type is optional and defaults to 0, as the other specific data types
    return _58(orddenom_spec_data_type=None, **kwargs)


def _55(**kwargs):
    return pyuff.prepare_55(analysis_type=2, data_ch=2, spec_data_type=8, node_nums=np.arange(1, 4),
                            load_case=1, mode_n=1, freq=10., r1=np.ones(3), r2=np.zeros(3), r3=np.ones(3),
                            **kwargs)


def _15(**kwargs):
    return pyuff.prepare_15(node_nums=np.arange(1, 4), x=np.zeros(3), y=np.ones(3), z=np.arange(3.), **kwargs)


def _82(**kwargs):
    return pyuff.prepare_82(trace_num=1, n_nodes=3, nodes=np.array([1, 2, 3]), **kwargs)


def _164(**kwargs):
    return pyuff.prepare_164(units_code=1, length=1., force=1., temp=1., temp_offset=273.15, **kwargs)


@pytest.mark.parametrize('prepare', [_58, _58_without_orddenom, _55, _15, _82, _164])
def test_write_full_dict(tmp_path, prepare):
    # the optional fields of the full dictionary are None and get their default values
    full = prepare(return_full_dict=True)
    assert any(value is None for value in full.values())
    pyuff.UFF(tmp_path / 'full.uff').write_sets(full, 'add')
    pyuff.UFF(tmp_path / 'part.uff').write_sets(prepare(), 'add')

    assert open(tmp_path / 'full.uff', 'rb').read() == open(tmp_path / 'part.uff', 'rb').read()


def test_write_full_dict_58b(tmp_path):
    # single precision: ord_data_type is required
    for name, dset in (('full', _58(binary=1, ord_data_type=5, return_full_dict=True)),
                       ('part', _58(binary=1, ord_data_type=5))):
        pyuff.UFF(tmp_path / (name + '.uff')).write_sets(dset, 'add', force_double=False)
    assert open(tmp_path / 'full.uff', 'rb').read() == open(tmp_path / 'part.uff', 'rb').read()
