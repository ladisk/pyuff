import numpy as np
import time
import os

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none
from .. import pyuff

def _write151(fh, dset):
    """Writes dset data - data-set 151 - to an open file fh."""
    try:
        ds = time.strftime('%d-%b-%y', time.localtime())
        ts = time.strftime('%H:%M:%S', time.localtime())
        # handle optional fields
        dset = _opt_fields(dset, {'version_db1': '0',
                                        'version_db2': '0',
                                        'file_type': '0',
                                        'date_db_created': ds,
                                        'time_db_created': ts,
                                        'date_db_saved': ds,
                                        'time_db_saved': ts,
                                        'date_file_written': ds,
                                        'time_file_written': ts})
        # write strings to the file
        fh.write('%6i\n%6i%74s\n' % (-1, 151, ' '))
        fh.write('%-80s\n' % dset['model_name'])
        fh.write('%-80s\n' % dset['description'])
        fh.write('%-80s\n' % dset['db_app'])
        fh.write('%-10s%-10s%10s%10s%10s\n' % (dset['date_db_created'],
                                                dset['time_db_created'], dset['version_db1'], dset['version_db2'],
                                                dset['file_type']))
        fh.write('%-10s%-10s\n' % (dset['date_db_saved'], dset['time_db_saved']))
        fh.write('%-80s\n' % dset['program'])
        fh.write('%-10s%-10s\n' % (dset['date_file_written'], dset['time_file_written']))
        fh.write('%6i\n' % -1)
    except KeyError as msg:
        raise Exception('The required key \'' + msg.args[0] + '\' not present when writing data-set #151')
    except:
        raise Exception('Error writing data-set #151')


def _extract151(blockData):
    """Extract dset data - data-set 151."""
    dset = {'type': 151}
    try:
        splitData = blockData.splitlines(True)
        dset.update(_parse_header_line(splitData[2], 1, [80], [1], ['model_name']))
        dset.update(_parse_header_line(splitData[3], 1, [80], [1], ['description']))
        dset.update(_parse_header_line(splitData[4], 1, [80], [1], ['db_app']))
        dset.update(_parse_header_line(splitData[5], 2, [10, 10, 10, 10, 10], [1, 1, 2, 2, 2],
                                            ['date_db_created', 'time_db_created', 'version_db1', 'version_db2',
                                                'file_type']))
        dset.update(_parse_header_line(splitData[6], 1, [10, 10], [1, 1], ['date_db_saved', 'time_db_saved']))
        dset.update(_parse_header_line(splitData[7], 1, [80], [1], ['program']))
        dset.update(
            _parse_header_line(splitData[8], 1, [10, 10], [1, 1], ['date_file_written', 'time_file_written']))
    except:
        raise Exception('Error reading data-set #151')
    return dset


def dict_151(
    model_name=None,
    description=None,
    db_app=None,
    date_db_created=None,
    time_db_created=None,
    version_db1=None,
    version_db2=None,
    file_type=None,
    date_db_saved=None,
    time_db_saved=None,
    program=None,
    date_file_written=None,
    time_file_written=None,
    return_full_dict=False):

    """Name: Header

    R-Record, F-Field

    :param model_name: R1 F1, Model file name
    :param description: R2 F1, Model file description
    :param db_app: R3 F1, Name of the application that created database
    :param date_db_created: R4 F1, Date database created
    :param time_db_created: R4 F2, Time database created
    :param version_db1: R4 F3, Version string 1 of the database
    :param version_db2: R4 F4, Version string 2 of the database
    :param file_type: R4 F5, File type
    :param date_db_saved: R5 F1, Date database last saved
    :param time_db_saved: R5 F2, Time database last saved
    :param program: R6 F1, Program which created universal file
    :param date_file_written: R7 F1, Date universal file was written
    :param time_file_written: R7 F2 Time universal file was written

    :param return_full_dict: If True full dict with all keys is returned, else only specified arguments are included
    
    """

    dataset={'type': 151,
            'model_name': model_name,
            'description': description,
            'db_app': db_app,
            'date_db_created': date_db_created,
            'time_db_created': time_db_created,
            'version_db1': version_db1,
            'version_db2': version_db2,
            'file_type': file_type,
            'date_db_saved': date_db_saved,
            'time_db_saved': time_db_saved,
            'program': program,
            'date_file_written': date_file_written,
            'time_file_written': time_file_written,
            }
    

    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)


    return dataset


def prepare_test_151(save_to_file=''):
    dataset = {'type': 151,  # Header
               'model_name': 'Model file name',  # 80A1, model file name
               'description': 'Model file description',  # 80A1, model file description
               'db_app': 'Program which created DB',  # 80A1, program which created DB
               'date_db_created': '27-Jan-16',  # 10A1, date database created (DD-MMM-YY)
               'time_db_created': '14:38:15',  # 10A1, time database created (HH:MM:SS)
               'version_db1': 1,  # I10, Version from database
               'version_db2': 2,  # I10, Subversion from database
               'file_type': 0,  # I10, File type (0  Universal, 1 Archive, 2 Other)
               'date_db_saved': '28-Jan-16',  # 10A1, date database saved (DD-MMM-YY)
               'time_db_saved': '14:38:16',  # 10A1, time database saved (HH:MM:SS)
               'program': 'OpenModal',  # 80A1, program which created DB
               'date_db_written': '29-Jan-16',  # 10A1, date database written (DD-MMM-YY)
               'time_db_written': '14:38:17',  # 10A1, time database written (HH:MM:SS)
               }

    dataset_out = dataset.copy()

    if save_to_file:
        if os.path.exists(save_to_file):
            os.remove(save_to_file)
        uffwrite = pyuff.UFF(save_to_file)
        uffwrite._write_set(dataset, 'add')

    return dataset_out

