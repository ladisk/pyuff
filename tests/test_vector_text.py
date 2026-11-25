"""pyuff._vector_text gives the text of Python's % operator and the values of float()."""
import numpy as np
import pytest

from pyuff._vector_text import format_rows, parse_fixed_width


def _floats(rng, n):
    x = np.concatenate([
        rng.standard_normal(n),
        rng.standard_normal(n) * 10.0 ** rng.integers(-120, 120, n),
        np.round(rng.standard_normal(n) * 1e6) / 10.0 ** rng.integers(0, 9, n),   # decimal ties
        rng.integers(-2 ** 63, 2 ** 63 - 1, n, dtype=np.int64).view(np.float64),  # any bits
        10.0 ** rng.integers(-30, 30, n) * (1 + rng.choice([-1, 1], n) * rng.choice([0, 1e-17, 5e-7, 5e-12], n)),
    ])
    x[rng.random(x.size) < 0.02] = -0.0
    x[rng.random(x.size) < 0.01] = np.inf
    x[rng.random(x.size) < 0.01] = np.nan
    return x


@pytest.mark.parametrize('fmt', ['%13.5e', '%20.11e', '%13.5E', '%25.16e', '%10.3e'])
def test_format_e_as_python(fmt):
    rng = np.random.default_rng(0)
    x = _floats(rng, 2000)
    x = x[rng.permutation(x.size)][:x.size // 2 * 2].reshape(-1, 2)
    expected = ''.join((fmt + fmt + '\n') % (a, b) for a, b in x.tolist())
    assert format_rows([(fmt, x[:, 0]), (fmt, x[:, 1]), '\n']) == expected


def test_format_integers_and_overflow_as_python():
    rng = np.random.default_rng(1)
    ints = rng.integers(-10 ** 9, 10 ** 9, 5000)
    ints[::97] = 10 ** 12   # does not fit the field: every row is formatted by Python
    x = rng.standard_normal(5000)
    expected = ''.join('%10i%13.5e\n' % (i, v) for i, v in zip(ints.tolist(), x.tolist()))
    assert format_rows([('%10i', ints), ('%13.5e', x), '\n']) == expected


def test_parse_fixed_width_as_float():
    rng = np.random.default_rng(2)
    x = _floats(rng, 1000)
    x = x[np.isfinite(x)][:4000]
    body = ''.join(''.join('%20.11e' % v for v in x[i:i + 4]) + '\n' for i in range(0, x.size, 4)).encode()
    values, rest = parse_fixed_width(body, lambda length: [(20 * i, 20) for i in range(length // 20)])
    n = values.size
    assert np.array_equal(values, [float('%20.11e' % v) for v in x[:n]])
    assert [float(f) for f in rest.split()] == [float('%20.11e' % v) for v in x[n:]]


def test_parse_fixed_width_rejects_other_text():
    body = (('%20.11e' % 1.0) * 4 + '\n').encode() * 1000
    layout = lambda length: [(20 * i, 20) for i in range(length // 20)]
    assert parse_fixed_width(body.replace(b'e+00', b'x+00', 1), layout) is None
    assert parse_fixed_width(body[:-1] + b'\r', layout) is None
