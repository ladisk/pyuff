import subprocess
import sys
import warnings

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
