import numpy as np
import time
import os

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none, _fit_string

def get_structure_151(raw=False):
    """(source: https://www.ceas3.uc.edu/sdrluff/"""
    out = """
Universal Dataset Number: 151

Name:   Header
-----------------------------------------------------------------------

Record 1:       FORMAT(80A1)
                Field 1      -- model file name
Record 2:       FORMAT(80A1)
                Field 1      -- model file description
Record 3:       FORMAT(80A1)
                Field 1      -- program which created DB
Record 4:       FORMAT(10A1,10A1,3I10)
                Field 1      -- date database created (DD-MMM-YY)
                Field 2      -- time database created (HH:MM:SS)
                Field 3      -- Version from database
                Field 4      -- Version from database
                Field 5      -- File type
                                =0  Universal
                                =1  Archive
                                =2  Other
Record 5:       FORMAT(10A1,10A1)
                Field 1      -- date database last saved (DD-MMM-YY)
                Field 2      -- time database last saved (HH:MM:SS)
Record 6:       FORMAT(80A1)
                Field 1      -- program which created universal file
Record 7:       FORMAT(10A1,10A1)
                Field 1      -- date universal file written (DD-MMM-YY)
                Field 2      -- time universal file written (HH:MM:SS)

-----------------------------------------------------------------
"""

    if raw:
        return out
    else:
        print(out)   

def _write151(fh, dset):
    """Writes dset data - data-set 151 - to an open file fh."""
    try:
        ds = time.strftime('%d-%b-%y', time.localtime())
        ts = time.strftime('%H:%M:%S', time.localtime())
        # the date and time the file was written are read as date_file_written and
        # time_file_written; prepare_151 gives them as date_db_written and time_db_written,
        # which take precedence (they are not copied to the dictionary, so a later change
        # of date_db_written or time_db_written is written too)
        file_written = {}
        for key in ('date', 'time'):
            if dset.get(key + '_db_written') is not None:
                file_written[key] = dset[key + '_db_written']
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
        fh.write('%-80s\n' % _fit_string(dset['model_name'], 80, 'model_name', 151))
        fh.write('%-80s\n' % _fit_string(dset['description'], 80, 'description', 151))
        fh.write('%-80s\n' % _fit_string(dset['db_app'], 80, 'db_app', 151))
        fh.write('%-10s%-10s%10s%10s%10s\n' % (_fit_string(dset['date_db_created'], 10, 'date_db_created', 151),
                                                _fit_string(dset['time_db_created'], 10, 'time_db_created', 151),
                                                dset['version_db1'], dset['version_db2'],
                                                dset['file_type']))
        fh.write('%-10s%-10s\n' % (_fit_string(dset['date_db_saved'], 10, 'date_db_saved', 151),
                                     _fit_string(dset['time_db_saved'], 10, 'time_db_saved', 151)))
        fh.write('%-80s\n' % _fit_string(dset['program'], 80, 'program', 151))
        fh.write('%-10s%-10s\n' % (_fit_string(file_written.get('date', dset['date_file_written']), 10,
                                                 'date_file_written', 151),
                                     _fit_string(file_written.get('time', dset['time_file_written']), 10,
                                                 'time_file_written', 151)))
        fh.write('%6i\n' % -1)
    except KeyError as msg:
        raise Exception('The required key \'' + msg.args[0] + '\' not present when writing data-set #151')
    except:
        raise Exception('Error writing data-set #151')


def _extract151(dataset_data):
    """Extract dset data - data-set 151."""
    dset = {'type': 151}
    try:
        split_data = dataset_data.splitlines(True)
        dset.update(_parse_header_line(split_data[2], 1, [80], [1], ['model_name']))
        dset.update(_parse_header_line(split_data[3], 1, [80], [1], ['description']))
        dset.update(_parse_header_line(split_data[4], 1, [80], [1], ['db_app']))
        dset.update(_parse_header_line(split_data[5], 2, [10, 10, 10, 10, 10], [1, 1, 2, 2, 2],
                                            ['date_db_created', 'time_db_created', 'version_db1', 'version_db2',
                                                'file_type']))
        dset.update(_parse_header_line(split_data[6], 1, [10, 10], [1, 1], ['date_db_saved', 'time_db_saved']))
        dset.update(_parse_header_line(split_data[7], 1, [80], [1], ['program']))
        dset.update(
            _parse_header_line(split_data[8], 1, [10, 10], [1, 1], ['date_file_written', 'time_file_written']))
    except:
        raise Exception('Error reading data-set #151')
    return dset


def prepare_151(
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
        date_db_written=None,
        time_db_written=None,
        return_full_dict=False):

    """Prepare data-set 151 (header) for writing.

    The header describes the model and the programs that created the database and
    the universal file. It is usually the first data-set of a file.

    Parameters
    ----------
    model_name : str
        Model file name, up to 80 characters (record 1, field 1).
    description : str
        Model file description, up to 80 characters (record 2, field 1).
    db_app : str
        Name of the application that created the database, up to 80 characters
        (record 3, field 1).
    date_db_created : str, optional
        Date the database was created, e.g. ``'27-Jan-16'``, up to 10 characters
        (record 4, field 1). Default: the current date.
    time_db_created : str, optional
        Time the database was created, e.g. ``'14:38:15'``, up to 10 characters
        (record 4, field 2). Default: the current time.
    version_db1 : int, optional
        Version string 1 of the database (record 4, field 3). Default: 0.
    version_db2 : int, optional
        Version string 2 of the database (record 4, field 4). Default: 0.
    file_type : {0, 1, 2}, optional
        File type (record 4, field 5): 0 universal, 1 archive, 2 other.
        Default: 0.
    date_db_saved : str, optional
        Date the database was last saved (record 5, field 1). Default: the
        current date.
    time_db_saved : str, optional
        Time the database was last saved (record 5, field 2). Default: the
        current time.
    program : str
        Program that wrote the universal file, up to 80 characters (record 6,
        field 1).
    date_db_written : str, optional
        Date the universal file was written (record 7, field 1); it is read back as
        ``date_file_written``. Default: the current date.
    time_db_written : str, optional
        Time the universal file was written (record 7, field 2); it is read back as
        ``time_file_written``. Default: the current time.
    return_full_dict : bool, optional
        If True, the dictionary has all keys of the data-set, with ``None`` for
        the values not given; if False (default), only the given values.

    Returns
    -------
    dict
        The data-set, to be written with :meth:`pyuff.UFF.write_sets`.

    Examples
    --------
    >>> import pyuff
    >>> header = pyuff.prepare_151(
    ...     model_name='Beam',
    ...     description='Free-free steel beam',
    ...     db_app='pyuff',
    ...     program='pyuff')
    >>> pyuff.UFF('beam_model.uff').write_sets(header, mode='overwrite')
    """
    if type(model_name) != str and model_name != None:
        raise TypeError('model_name must be string')
    if type(description) != str and description != None:
        raise TypeError('description must be string')
    if type(db_app) != str and db_app != None:
        raise TypeError('db_app must be string')
    if type(date_db_created) != str and date_db_created != None:
        raise TypeError('date_db_created must be string')
    if type(time_db_created) != str and time_db_created != None:
        raise TypeError('time_db_created must be string')
    if np.array(version_db1).dtype != int and version_db1 != None:
        raise TypeError('version_db1 must be integer')
    if np.array(version_db2).dtype != int and version_db2 != None:
        raise TypeError('version_db2 must be integer')
    if file_type not in (0, 1, 2, None):
        raise ValueError('file type can be 0:Universal, 1:Archive, 2:Other')
    if type(date_db_saved) != str and date_db_saved != None:
        raise TypeError('date_db_saved must be string')
    if type(time_db_saved) != str and time_db_saved != None:
        raise TypeError('time_db_saved must be string')
    if type(program) != str and program != None:
        raise TypeError('program must be string')
    if type(date_db_written) != str and date_db_written != None:
        raise TypeError('date_db_written must be string')
    if type(time_db_written) != str and time_db_written != None:
        raise TypeError('time_db_written must be string')


    dataset={
        'type': 151,
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
        'date_db_written': date_db_written,
        'time_db_written': time_db_written,
        }
    

    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)


    return dataset
