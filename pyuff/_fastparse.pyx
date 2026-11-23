# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Optional compiled parser of fixed-width float fields (Clinger's fast path).

Each field is parsed in C: optional blanks, sign, digits with an optional decimal point,
an optional exponent (e/E, sign, digits) and blanks. When the decimal significand m has
at most 19 digits, m <= 2**53 and the decimal exponent k satisfies |k| <= 22, the value
m * 10**k (or m / 10**-k) is computed with a single correctly rounded floating-point
operation on exact operands, so it is the correctly rounded value of the decimal number
(W. D. Clinger, "How to read floating point numbers accurately", PLDI 1990; used by
D. Lemire, "Number Parsing at a Gigabyte per Second", 2021). Any other field is given
to CPython's own parser (PyOS_string_to_double, as float()), which also raises
ValueError for a field that is not a number. The result is the same as float() of the
field (after stripping blanks).

Build (optional; pyuff works without it): python setup_fastparse.py build_ext --inplace
"""
import numpy as np
cimport numpy as cnp
from cpython.ref cimport PyObject
from libc.stdint cimport uint64_t, int64_t

cnp.import_array()

cdef extern from "Python.h":
    double PyOS_string_to_double(const char *s, char **endptr, PyObject *overflow_exception) except? -1.0

cdef double[23] POW10 = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11,
                         1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22]
cdef uint64_t MAX_EXACT = 9007199254740992  # 2**53


cdef inline bint _fast(const unsigned char *p, Py_ssize_t n, double *out) noexcept nogil:
    """Clinger's fast path; returns False if the field must be parsed by CPython."""
    cdef Py_ssize_t i = 0
    cdef bint negative = False
    cdef uint64_t m = 0
    cdef int n_digits = 0, frac_digits = 0, exp_value = 0, exp_digits = 0
    cdef bint exp_negative = False, seen_dot = False
    cdef int64_t k
    cdef unsigned char c
    while i < n and p[i] == 32:
        i += 1
    if i < n and (p[i] == 43 or p[i] == 45):      # '+' '-'
        negative = p[i] == 45
        i += 1
    while i < n:
        c = p[i]
        if 48 <= c <= 57:
            if n_digits >= 19:
                return False
            m = m * 10 + (c - 48)
            n_digits += 1
            if seen_dot:
                frac_digits += 1
        elif c == 46 and not seen_dot:            # '.'
            seen_dot = True
        else:
            break
        i += 1
    if n_digits == 0:
        return False
    if i < n and (p[i] == 101 or p[i] == 69):     # 'e' 'E'
        i += 1
        if i < n and (p[i] == 43 or p[i] == 45):
            exp_negative = p[i] == 45
            i += 1
        while i < n and 48 <= p[i] <= 57:
            if exp_digits < 4:
                exp_value = exp_value * 10 + (p[i] - 48)
            exp_digits += 1
            i += 1
        if exp_digits == 0 or exp_digits >= 4:
            return False
    while i < n and p[i] == 32:
        i += 1
    if i != n or m > MAX_EXACT:
        return False
    k = (-exp_value if exp_negative else exp_value) - frac_digits
    if k < -22 or k > 22:
        return False
    if k >= 0:
        out[0] = <double>m * POW10[k]
    else:
        out[0] = <double>m / POW10[-k]
    if negative:
        out[0] = -out[0]
    return True


cdef double _slow(const unsigned char *p, Py_ssize_t n) except? -1.0:
    """CPython's parser (as float()) for the field without blanks around it."""
    cdef char buf[128]
    cdef Py_ssize_t a = 0, b = n
    while a < b and p[a] == 32:
        a += 1
    while b > a and p[b - 1] == 32:
        b -= 1
    if b - a >= 127:
        return float(bytes(p[a:b]))
    for j in range(b - a):
        buf[j] = <char>p[a + j]
    buf[b - a] = 0
    return PyOS_string_to_double(buf, NULL, NULL)


def parse_fields(const unsigned char[:, ::1] lines, Py_ssize_t[::1] offsets, Py_ssize_t[::1] widths):
    """Values (n_lines, n_fields) of the fixed-width fields of each line.

    :param lines: uint8 array (n_lines, line_length), C-contiguous.
    :param offsets, widths: of each field (intp arrays).
    Raises ValueError for a field that is not a number (as float()).
    """
    cdef Py_ssize_t n_lines = lines.shape[0], n_fields = offsets.shape[0]
    cdef Py_ssize_t i, j
    cdef cnp.ndarray[cnp.float64_t, ndim=2] out = np.empty((n_lines, n_fields), dtype=np.float64)
    cdef double[:, ::1] o = out
    cdef double v
    cdef const unsigned char *row
    for j in range(n_fields):
        if offsets[j] < 0 or offsets[j] + widths[j] > lines.shape[1]:
            raise ValueError('field outside the line')
    for i in range(n_lines):
        row = &lines[i, 0]
        for j in range(n_fields):
            if not _fast(row + offsets[j], widths[j], &v):
                v = _slow(row + offsets[j], widths[j])
            o[i, j] = v
    return out
