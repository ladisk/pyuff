import numpy as np
import time

from ..tools import UFFException, _opt_fields, _parse_header_line


def _write151(fh, dset):
    # Writes dset data - data-set 151 - to an open file fh
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
        raise UFFException('The required key \'' + msg.args[0] + '\' not present when writing data-set #151')
    except:
        raise UFFException('Error writing data-set #151')


def _extract151(blockData):
    # Extract dset data - data-set 151.
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
        raise UFFException('Error reading data-set #151')
    return dset