"""Vectorized formatting and parsing of the fixed-width numeric records of universal files.

The records of a universal file are fixed-width fields such as ``%13.5e``, ``%20.11e`` and
``%10i``. Formatting them value by value with Python's ``%`` operator dominates the time of
writing large data-sets; this module formats whole columns at once with NumPy and gives
**exactly the same text** as the ``%`` operator:

* ``%W.Pe`` (up to 15 significant digits): the absolute value is scaled by a power of ten
  to an integer of P+1 digits and rounded (the approach of fixed-precision printing, see
  R. Cox, "Floating-Point Printing and Parsing Can Be Simple And Fast", 2026). The scaled
  value is computed with at most a few floating-point roundings; where this error could
  change the correctly rounded result (the fraction is close to 1/2, e.g. exact ties, or
  the value is close to a power of ten), and for non-finite values, the value is formatted
  by Python. The digits are written two at a time from a table.
* ``%Wi`` / ``%Wd`` and ``%W.0f``: integer digits from the same table.

Other formats (e.g. ``%25.16e``, 17 significant digits) are formatted by Python column by
column. If a value does not fit its field width, all rows are formatted by Python, so the
result is always that of ``fmt % row`` for every row.

Parsing: :func:`parse_fixed_width` converts fixed-width fields of equal-length lines to
float with NumPy's bytes-to-float conversion, with the same values (and rejecting the same
fields) as Python's ``float()``.
"""
import re

import numpy as np

__all__ = ['format_rows', 'parse_fixed_width']

# '00', '01', ..., '99' as 2-byte (uint16) values
_DIGIT_PAIRS = np.frombuffer(b''.join(b'%02d' % i for i in range(100)), dtype=np.uint16)
_POW10 = 10.0 ** np.arange(-30, 31)   # _POW10[k + 30] == 10.0**k
_FORMAT = re.compile(r'^%(\d+)(?:\.(\d+))?([eEidf])$')
_SPACE, _MINUS, _PLUS, _DOT = ord(' '), ord('-'), ord('+'), ord('.')


_MIN_VECTOR_VALUES = 1000   # below this, Python is faster (measured)
_MIN_VECTOR_BYTES = 16384


class _Overflow(Exception):
    """A value does not fit the width of its field."""


def _digits(m, n_digits):
    """ASCII digits (n, n_digits) of the non-negative integers m (float64, < 10**n_digits <= 1e15)."""
    n = m.shape[0]
    n_groups = -(-n_digits // 6)
    out = np.empty((n, 3 * n_groups), dtype=np.uint16)
    rest = m
    for g in range(n_groups - 1, -1, -1):
        # 6 digits at a time; the products are exact enough for floor (see module doc)
        high = np.floor(rest * 1e-6)
        low = (rest - high * 1e6).astype(np.uint32)
        rest = high
        a, r = np.divmod(low, np.uint32(10000))
        b, c = np.divmod(r, np.uint32(100))
        out[:, 3 * g] = _DIGIT_PAIRS[a]
        out[:, 3 * g + 1] = _DIGIT_PAIRS[b]
        out[:, 3 * g + 2] = _DIGIT_PAIRS[c]
    return out.view(np.uint8).reshape(n, 6 * n_groups)[:, 6 * n_groups - n_digits:]


def _scale(ax, k):
    """ax * 10**k, with one or two roundings."""
    k1 = np.clip(k, -30, 30)
    s = ax * _POW10[k1 + 30]
    rest = k - k1
    if rest.any():
        s = s * _POW10[np.clip(rest, -30, 30) + 30]
    return s


def _format_e(x, width, prec, upper):
    """'%{width}.{prec}e' % v for each v of the float64 array x, as uint8 array (n, width)."""
    n = x.shape[0]
    n_sig = prec + 1
    length = 1 + n_sig + (1 if prec else 0) + 4   # sign, digits, '.', 'e+XX'
    if length > width:
        raise _Overflow
    ax = np.abs(x)
    finite = np.isfinite(x)
    nonzero = finite & (ax > 0)
    low_bound, high_bound = 10.0 ** prec, 10.0 ** n_sig
    with np.errstate(all='ignore'):
        e = np.floor(np.log10(np.where(nonzero, ax, 1.0))).astype(np.int64)
        s = _scale(ax, prec - e)
        # log10 can be off by one near powers of ten
        too_low = nonzero & (s < low_bound)
        too_high = nonzero & (s >= high_bound)
        if too_low.any() or too_high.any():
            e[too_low] -= 1
            e[too_high] += 1
            s = _scale(ax, prec - e)
        whole = np.floor(s)
        frac = s - whole
    m = whole + (frac > 0.5)
    carry = m >= high_bound   # e.g. 9.999996 -> 1.00000e+01
    m[carry] = low_bound
    e[carry] += 1
    m[~nonzero] = 0.0
    e[~nonzero] = 0
    # error of s: a few roundings of a value below 10**n_sig
    tol = 16.0 * high_bound * 2.0 ** -53
    slow = ~finite | (nonzero & ((np.abs(frac - 0.5) < tol) | (s < low_bound + tol) |
                                 (s >= high_bound - tol))) | (e <= -100) | (e >= 100)
    out = np.full((n, width), _SPACE, dtype=np.uint8)
    pos = width - length
    out[:, pos] = np.where(np.signbit(x), _MINUS, _SPACE)
    digits = _digits(m, n_sig)
    out[:, pos + 1] = digits[:, 0]
    pos += 2
    if prec:
        out[:, pos] = _DOT
        out[:, pos + 1:pos + 1 + prec] = digits[:, 1:]
        pos += 1 + prec
    out[:, pos] = ord('E') if upper else ord('e')
    out[:, pos + 1] = np.where(e < 0, _MINUS, _PLUS)
    out[:, pos + 2:pos + 4] = _DIGIT_PAIRS[np.minimum(np.abs(e), 99)].view(np.uint8).reshape(n, 2)
    idx = np.flatnonzero(slow)
    if idx.size:
        fmt = '%' + str(width) + '.' + str(prec) + ('E' if upper else 'e')
        text = ''.join([fmt % v for v in x[idx].tolist()]).encode('ascii')
        if len(text) != idx.size * width:
            raise _Overflow
        out[idx] = np.frombuffer(text, dtype=np.uint8).reshape(-1, width)
    return out


def _format_integer(v, negative, width):
    """Right-aligned integers |v| (float64, integral) with '-' where negative, as (n, width)."""
    n = v.shape[0]
    if width > 15 or (v >= 10.0 ** (width - 1)).any():
        # (a value with width digits fits only without a sign; rare, left to Python)
        raise _Overflow
    n_digits = 1 + np.searchsorted(10.0 ** np.arange(1, width), v, side='right')
    out = np.full((n, width), _SPACE, dtype=np.uint8)
    digits = _digits(v, width - 1)
    col = np.arange(width - 1)
    show = col[None, :] >= (width - 1 - n_digits)[:, None]
    out[:, 1:] = np.where(show, digits, _SPACE)
    sign_col = width - 1 - n_digits
    rows = np.flatnonzero(negative)
    out[rows, sign_col[rows]] = _MINUS
    return out


def _format_column(fmt, column, n):
    """uint8 array (n, width) of fmt % value for the first n values of column."""
    match = _FORMAT.match(fmt)
    if match is None:
        raise _Overflow
    width = int(match.group(1))
    prec = match.group(2)
    kind = match.group(3)
    values = np.asarray(column)[:n] if not isinstance(column, np.ndarray) else column[:n]
    if values.ndim == 1 and values.dtype.kind in 'biuf':
        if kind in 'eE' and prec is not None and int(prec) <= 14:
            return _format_e(values.astype(np.float64), width, int(prec), kind == 'E')
        if kind in 'id' and prec is None:
            if values.dtype.kind == 'f':
                if not np.isfinite(values).all():
                    raise _Overflow
                values = np.trunc(values)   # '%i' % 1.7 == '1'
            values = values.astype(np.float64)
            if (np.abs(values) >= 2.0 ** 53).any():
                raise _Overflow
            return _format_integer(np.abs(values), values < 0, width)
        if kind == 'f' and prec == '0':
            if values.dtype.kind == 'f':
                if not np.isfinite(values).all():
                    raise _Overflow
                rounded = np.rint(values)   # round half to even, as '%.0f'
                negative = np.signbit(rounded)   # '%.0f' % -0.4 == '-0'
            else:
                rounded = values.astype(np.float64)
                negative = rounded < 0
            if (np.abs(rounded) >= 2.0 ** 53).any():
                raise _Overflow
            return _format_integer(np.abs(rounded), negative, width)
    # any other format or value type: Python, a whole column at once
    text = ((fmt * n) % tuple(values.tolist() if isinstance(values, np.ndarray) else values)).encode('ascii')
    if len(text) != n * width:
        raise _Overflow
    return np.frombuffer(text, dtype=np.uint8).reshape(n, width)


def format_rows(items, n=None):
    """Formats rows of fixed-width fields; the same text as ``fmt % row`` for each row.

    :param items: list of ``(fmt, column)`` pairs, e.g. ``('%13.5e', x)``, and literal
        strings, e.g. ``'\\n'``. ``fmt`` is a single conversion: ``%W.Pe``, ``%W.PE``,
        ``%Wi``, ``%Wd``, ``%W.0f`` are vectorized, other conversions are formatted by Python.
    :param n: number of rows; default (and at most) the length of the shortest column,
        as with ``zip``.
    :return: str
    """
    columns = [item[1] for item in items if not isinstance(item, str)]
    length = min(len(c) for c in columns) if columns else 0
    n = length if n is None else min(n, length)
    if n == 0:
        return ''
    if n * len(columns) < _MIN_VECTOR_VALUES:
        # few values: NumPy's overhead exceeds the gain; one % operation for all rows
        fmt = ''.join(item if isinstance(item, str) else item[0] for item in items)
        lists = [c[:n].tolist() if isinstance(c, np.ndarray) else list(c)[:n] for c in columns]
        return (fmt * n) % tuple([value for row in zip(*lists) for value in row])
    try:
        blocks = []
        for item in items:
            if isinstance(item, str):
                blocks.append(np.frombuffer(item.encode('ascii'), dtype=np.uint8)[None, :].repeat(n, axis=0))
            else:
                blocks.append(_format_column(item[0], item[1], n))
        return np.concatenate(blocks, axis=1).tobytes().decode('ascii')
    except (_Overflow, ValueError, TypeError, OverflowError):
        # exactly as before: Python formats row by row (and raises its own errors)
        fmt = ''.join(item if isinstance(item, str) else item[0] for item in items)
        return ''.join([fmt % row for row in zip(*[list(c)[:n] for c in columns])])


# bytes that can occur in fixed-width float fields and line ends
_FLOAT_BYTES = np.zeros(256, dtype=bool)
_FLOAT_BYTES[np.frombuffer(b'0123456789+-.eE \r\n', dtype=np.uint8)] = True


def parse_fixed_width(body, fields):
    """Parses lines of fixed-width float fields at once, or returns None.

    :param body: bytes, the lines (each ending with LF or CR LF).
    :param fields: function of the line length L (with the line end) returning the list of
        (offset, width) of the fields of a line, in order.
    :return: (values of all full-length lines, in order (float64), text of the remaining
        lines (str)), or None if the lines cannot be parsed at once (they have characters
        other than those of numbers, lines of different length, CR without LF, or a field
        that is not a number); the caller then parses them by other means.

    The lines up to the last one of the most common length are parsed; the rest (usually
    the shorter last line) is returned as text, to be parsed by the caller.
    """
    if len(body) < _MIN_VECTOR_BYTES:
        return None   # few lines: NumPy's overhead exceeds the gain
    data = np.frombuffer(body, dtype=np.uint8)
    if data[-1] != 10 or not _FLOAT_BYTES[data].all():
        return None
    line_ends = np.flatnonzero(data == 10)
    n_cr = np.count_nonzero(data == 13)
    if n_cr and (n_cr != line_ends.size or not (data[line_ends - 1] == 13).all()):
        return None
    lengths = np.diff(line_ends, prepend=-1)
    line_length = int(lengths[0])
    # the full-length lines are the first ones; others (if any) only at the end
    n_full = int(np.argmax(lengths != line_length)) if (lengths != line_length).any() else lengths.size
    if (lengths[n_full:] == line_length).any():
        return None
    layout = fields(line_length)
    if not layout or n_full == 0:
        return None
    lines = data[:n_full * line_length].reshape(n_full, line_length)
    if any(offset + width > line_length for offset, width in layout):
        return None
    try:
        values = np.empty((n_full, len(layout)), dtype=np.float64)
        for j, (offset, width) in enumerate(layout):
            column = np.ascontiguousarray(lines[:, offset:offset + width])
            values[:, j] = column.view('S%d' % width).ravel().astype(np.float64)
    except ValueError:
        return None
    rest = body[n_full * line_length:].decode('ascii')
    return values.ravel(), rest
