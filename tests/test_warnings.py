import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

import pyuff


def test_import_keeps_caller_warning_filters():
    # Issue #113: importing pyuff must not override filters set by the caller.
    code = (
        "import warnings\n"
        "warnings.simplefilter('error')\n"
        "warnings.filterwarnings('ignore', message='sentinel')\n"
        "import pyuff\n"
        "warnings.warn('sentinel')  # must stay ignored\n"
        "try:\n"
        "    warnings.warn('other')\n"
        "except UserWarning:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('caller filters were overridden by pyuff')\n"
    )
    subprocess.run([sys.executable, '-c', code], check=True)


def test_fileName_deprecation_warning():
    with pytest.warns(FutureWarning, match='fileName') as record:
        pyuff.UFF(fileName='./data/beam.uff')
    assert record[0].filename == __file__


def test_write_nan_inf_warns():
    # NaN and infinity are replaced when written (NaN by 0), with a warning.
    save_to_file = './data/temp_nan.uff'
    if os.path.exists(save_to_file):
        os.remove(save_to_file)
    x = np.arange(5.)
    data = np.array([1., np.nan, 3., np.inf, 5.]) * (1 + 1j)
    dataset = pyuff.prepare_58(binary=0, func_type=4, rsp_node=1, rsp_dir=1, ref_node=1, ref_dir=1,
                               data=data, x=x, abscissa_spacing=1, abscissa_spec_data_type=18,
                               ordinate_spec_data_type=12, orddenom_spec_data_type=13)
    with pytest.warns(UserWarning, match='Data-set 58: 2 of 5 data values are NaN or infinite') as record:
        pyuff.UFF(save_to_file).write_sets(dataset, mode='add')
    assert record[0].filename == __file__
    b = pyuff.UFF(save_to_file).read_sets(0)
    os.remove(save_to_file)

    np.testing.assert_array_equal(b['data'][[0, 1, 2, 4]], np.array([1., 0., 3., 5.]) * (1 + 1j))
