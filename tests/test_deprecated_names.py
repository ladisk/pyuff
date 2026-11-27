"""The names used before pyuff 2.8.0 still work, with a DeprecationWarning, until pyuff 3.0."""
import numpy as np
import pytest

import pyuff


def _equal(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return np.array_equal(a, b)


@pytest.mark.parametrize('old, new', [
    ('get_supported_sets', 'get_supported_datasets'),
    ('get_n_sets', 'get_n_datasets'),
    ('get_set_types', 'get_dataset_types'),
    ('get_set_formats', 'get_dataset_formats'),
    ('read_sets', 'read_datasets'),
])
def test_old_method_warns_and_gives_the_same(old, new):
    uff = pyuff.UFF('./data/beam.uff')
    with pytest.warns(DeprecationWarning, match='UFF.%s is deprecated.*use UFF.%s' % (old, new)):
        result = getattr(uff, old)()
    assert _equal(result, getattr(uff, new)())


def test_read_sets_with_setn():
    uff = pyuff.UFF('./data/beam.uff')
    with pytest.warns(DeprecationWarning):
        result = uff.read_sets(setn=[4, 5], header_only=True)
    assert _equal(result, uff.read_datasets(index=[4, 5], header_only=True))
    with pytest.warns(DeprecationWarning):
        result = uff._read_set(4)
    assert _equal(result, uff._read_dataset(4))


def test_write_sets(tmp_path):
    dsets = pyuff.UFF('./data/beam.uff').read_datasets()
    old, new = tmp_path / 'old.uff', tmp_path / 'new.uff'
    with pytest.warns(DeprecationWarning, match='UFF.write_sets'):
        pyuff.UFF(old).write_sets(dsets=dsets, mode='overwrite')
    pyuff.UFF(new).write_datasets(dsets, mode='overwrite')
    with pytest.warns(DeprecationWarning, match='UFF._write_set'):
        pyuff.UFF(old)._write_set(dsets[0], 'add')
    pyuff.UFF(new)._write_dataset(dsets[0], 'add')
    assert old.read_bytes() == new.read_bytes()


@pytest.mark.parametrize('old, new', [
    ('_block_ind', '_dataset_ind'),
    ('_n_sets', '_n_datasets'),
    ('_set_types', '_dataset_types'),
    ('_set_formats', '_dataset_formats'),
])
def test_old_attribute_warns_and_is_the_same(old, new):
    uff = pyuff.UFF('./data/beam.uff')
    with pytest.warns(DeprecationWarning, match='UFF.%s is deprecated' % old):
        value = getattr(uff, old)
    assert value is getattr(uff, new)
    with pytest.warns(DeprecationWarning, match='UFF.%s is deprecated' % old):
        setattr(uff, old, 'x')
    assert getattr(uff, new) == 'x'


def test_warning_points_to_the_caller():
    uff = pyuff.UFF('./data/beam.uff')
    with pytest.warns(DeprecationWarning) as record:
        uff.get_n_sets()
        uff._n_sets
    assert all(w.filename == __file__ for w in record)
