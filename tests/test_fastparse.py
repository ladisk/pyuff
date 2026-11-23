"""The optional compiled parser (pyuff._fastparse) gives the values of float()."""
import random
import struct

import numpy as np
import pytest

from pyuff import _vector_text

fastparse = pytest.importorskip('pyuff._fastparse', reason='optional module, build: python setup_fastparse.py')


def _parse(strings, width):
    data = b''.join(s.encode().rjust(width) for s in strings)
    lines = np.frombuffer(data, dtype=np.uint8).reshape(-1, width)
    return fastparse.parse_fields(lines, np.array([0], np.intp), np.array([width], np.intp))[:, 0]


def test_same_as_float():
    rng = np.random.default_rng(0)
    values = rng.integers(-2 ** 63, 2 ** 63 - 1, 20000, dtype=np.int64).view(np.float64)
    values = values[np.isfinite(values)]
    strings = ['%.5e' % v for v in values] + ['%.11e' % v for v in values] + ['%.17g' % v for v in values]
    strings += ['-0.0', '0', '+.5', '5.', '1e22', '1e23', '9007199254740993', '12345678901234567890',
                '4.9e-324', '1e-400', '1e400', '-1e400', '1.7976931348623159e308']
    got = _parse(strings, 26)
    expected = [float(s) for s in strings]
    assert all(struct.pack('d', a) == struct.pack('d', b) for a, b in zip(got, expected))


@pytest.mark.parametrize('field', ['', '1e', '1.2.3', '+-1', '1 2', '.', 'e5'])
def test_not_a_number_raises(field):
    with pytest.raises(ValueError):
        _parse([field], 13)


def test_random_fields_as_float():
    rnd = random.Random(1)
    alphabet = ' 0123456789+-.eE'
    for _ in range(20000):
        s = ''.join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 13)))
        try:
            expected = float(s)
        except ValueError:
            expected = None
        try:
            got = _parse([s], 13)[0]
        except ValueError:
            got = None
        assert (got is None and expected is None) or struct.pack('d', got) == struct.pack('d', expected), s


def test_parse_fixed_width_with_and_without_compiled_parser(monkeypatch):
    rng = np.random.default_rng(2)
    values = rng.standard_normal(4000) * 10.0 ** rng.integers(-20, 20, 4000)
    body = ''.join(''.join('%20.11e' % v for v in values[i:i + 4]) + '\n' for i in range(0, 4000, 4)).encode()
    layout = lambda length: [(20 * i, 20) for i in range(length // 20)]
    compiled, _ = _vector_text.parse_fixed_width(body, layout)
    monkeypatch.setattr(_vector_text, '_fastparse', None)
    with_numpy, _ = _vector_text.parse_fixed_width(body, layout)
    np.testing.assert_array_equal(compiled, with_numpy)
    np.testing.assert_array_equal(compiled, [float('%20.11e' % v) for v in values])
