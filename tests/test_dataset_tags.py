"""The '    -1' tags that start and end the data-sets."""
import pytest

import pyuff
from pyuff.pyuff import _find_dataset_tags


def _tags(lines):
    return _find_dataset_tags(''.join(line + '\n' for line in lines).encode())


def _offsets(lines, indices):
    return [sum(len(line) + 1 for line in lines[:i]) for i in indices]


HEADER_164 = ['    -1', '   164',
              '         1  SI: Meter (newton)           2',
              '    1.00000000000000000E+00    1.00000000000000000E+00    1.00000000000000000E+00',
              '    2.73149999999999977E+02',
              '    -1']


def test_record_with_a_single_field_of_minus_one():
    # a record with one I10 field of -1 looks like a tag shifted to the right
    lines = ['    -1', '    82', '         1         9         1', 'NONE',
             '         1         2         3         4         5         6         7         8',
             '        -1', '    -1']
    assert _tags(lines) == _offsets(lines, [0, 6])


def test_record_ending_with_minus_one():
    # a record that ends with -1 (as in data-set 1101 of ebox.unv, an I-DEAS example in
    # github.com/usnistgov/engineering-design-models); the data-set has more lines
    lines = ['    -1', '  1101', '           4     1     1          -1', 'Main', '    -1'] + HEADER_164
    assert _tags(lines) == _offsets(lines, [0, 4, 5, 10])


def test_last_record_ending_with_minus_one():
    # the record ending with -1 is followed by the end tag and the next start tag
    lines = ['    -1', '  1101', '           4     1     1          -1', '    -1'] + HEADER_164
    assert _tags(lines) == _offsets(lines, [0, 3, 4, 9])


@pytest.mark.parametrize('line_end', ['\n', '\r\n', '\r'])
def test_line_ends_and_80_column_tags(line_end):
    lines = ['    -1' + ' ' * 74, '   164', '    -1' + ' ' * 74, '    -1', '    82', '        -1', '    -1']
    text = ''.join(line + line_end for line in lines).encode()
    starts = [sum(len(line) + len(line_end) for line in lines[:i]) for i in [0, 2, 3, 6]]
    assert _find_dataset_tags(text) == starts


def test_block_ind_is_a_deprecated_alias():
    uff = pyuff.UFF('./data/beam.uff')
    with pytest.warns(FutureWarning, match='_dataset_ind'):
        block_ind = uff._block_ind
    assert block_ind is uff._dataset_ind
    with pytest.warns(FutureWarning, match='_dataset_ind'):
        uff._block_ind = block_ind[:2]
    assert len(uff._dataset_ind) == 2
