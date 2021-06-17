import numpy as np
import struct
import sys

from ..tools import UFFException, _opt_fields, _parse_header_line

def _write58(fh, dset, mode='add', _fileName=None):
    # Writes function at nodal DOF - data-set 58 - to an open file fh.
    try:
        if not (dset['func_type'] in [1, 2, 3, 4, 6]):
            raise UFFException('Unsupported function type')
        # handle optional fields - only those that are not calculated
        # automatically
        dict = {'units_description': '',
                'id1': 'NONE',
                'id2': 'NONE',
                'id3': 'NONE',
                'id4': 'NONE',
                'id5': 'NONE',
                'func_id': 0,
                'ver_num': 0,
                'binary': 0,
                'load_case_id': 0,
                'rsp_ent_name': 'NONE',
                'ref_ent_name': 'NONE',
                'abscissa_axis_lab': 'NONE',
                'abscissa_axis_units_lab': 'NONE',
                'abscissa_len_unit_exp': 0,
                'abscissa_force_unit_exp': 0,
                'abscissa_temp_unit_exp': 0,
                'ordinate_len_unit_exp': 0,
                'ordinate_force_unit_exp': 0,
                'ordinate_temp_unit_exp': 0,
                'ordinate_axis_lab': 'NONE',
                'ordinate_axis_units_lab': 'NONE',
                'orddenom_len_unit_exp': 0,
                'orddenom_force_unit_exp': 0,
                'orddenom_temp_unit_exp': 0,
                'orddenom_axis_lab': 'NONE',
                'orddenom_axis_units_lab': 'NONE',
                'z_axis_len_unit_exp': 0,
                'z_axis_force_unit_exp': 0,
                'z_axis_temp_unit_exp': 0,
                'z_axis_axis_lab': 'NONE',
                'z_axis_axis_units_lab': 'NONE',
                'z_axis_value': 0,
                'spec_data_type': 0,
                'abscissa_spec_data_type': 0,
                'ordinate_spec_data_type': 0,
                'z_axis_spec_data_type': 0,
                'version_num': 0,
                'abscissa_spacing': 0}
        dset = _opt_fields(dset, dict)
        # Write strings to the file - always in double precision => ord_data_type = 2
        # for real data and 6 for complex data
        numPts = len(dset['data'])
        isR = not np.iscomplexobj(dset['data'])
        if isR:
            # real data
            dset['ord_data_type'] = 4
            nBytes = numPts * 8
            if 'n_bytes' in dset.keys():
                dset['n_bytes'] = nBytes
            ordDataType = dset['ord_data_type']
        else:
            # complex data
            dset['ord_data_type'] = 6
            nBytes = numPts * 8
            ordDataType = 6

        isEven = bool(dset['abscissa_spacing'])  # handling even/uneven abscissa spacing manually

        # handling abscissa spacing automatically
        # isEven = len( set( [ dset['x'][ii]-dset['x'][ii-1] for ii in range(1,len(dset['x'])) ] ) ) == 1
        # decode utf to ascii
        for k, v in dset.items():
            if type(v) == str:
                dset[k] = v.encode("utf-8").decode('ascii','ignore')

        dset['abscissa_min'] = dset['x'][0]
        dx = dset['x'][1] - dset['x'][0]
        fh.write('%6i\n%6i' % (-1, 58))
        if dset['binary']:
            if sys.byteorder == 'little':
                bo = 1
            else:
                bo = 2
            fh.write('b%6i%6i%12i%12i%6i%6i%12i%12i\n' % (bo, 2, 11, nBytes, 0, 0, 0, 0))
        else:
            fh.write('%74s\n' % ' ')
        fh.write('%-80s\n' % dset['id1'])
        fh.write('%-80s\n' % dset['id2'])
        fh.write('%-80s\n' % dset['id3'])
        fh.write('%-80s\n' % dset['id4'])
        fh.write('%-80s\n' % dset['id5'])
        fh.write('%5i%10i%5i%10i %10s%10i%4i %10s%10i%4i\n' %
                    (dset['func_type'], dset['func_id'], dset['ver_num'], dset['load_case_id'],
                    dset['rsp_ent_name'], dset['rsp_node'], dset['rsp_dir'], dset['ref_ent_name'],
                    dset['ref_node'], dset['ref_dir']))
        fh.write('%10i%10i%10i%13.5e%13.5e%13.5e\n' % (ordDataType, numPts, isEven,
                                                        isEven * dset['abscissa_min'], isEven * dx,
                                                        dset['z_axis_value']))
        fh.write('%10i%5i%5i%5i %-20s %-20s\n' % (dset['abscissa_spec_data_type'],
                                                    dset['abscissa_len_unit_exp'], dset['abscissa_force_unit_exp'],
                                                    dset['abscissa_temp_unit_exp'], dset['abscissa_axis_lab'],
                                                    dset['abscissa_axis_units_lab']))
        fh.write('%10i%5i%5i%5i %-20s %-20s\n' % (dset['ordinate_spec_data_type'],
                                                    dset['ordinate_len_unit_exp'], dset['ordinate_force_unit_exp'],
                                                    dset['ordinate_temp_unit_exp'], dset['ordinate_axis_lab'],
                                                    dset['ordinate_axis_units_lab']))
        fh.write('%10i%5i%5i%5i %-20s %-20s\n' % (dset['orddenom_spec_data_type'],
                                                    dset['orddenom_len_unit_exp'], dset['orddenom_force_unit_exp'],
                                                    dset['orddenom_temp_unit_exp'], dset['orddenom_axis_lab'],
                                                    dset['orddenom_axis_units_lab']))
        fh.write('%10i%5i%5i%5i %-20s %-20s\n' % (dset['z_axis_spec_data_type'],
                                                    dset['z_axis_len_unit_exp'], dset['z_axis_force_unit_exp'],
                                                    dset['z_axis_temp_unit_exp'], dset['z_axis_axis_lab'],
                                                    dset['z_axis_axis_units_lab']))
        if isR:
            if isEven:
                data = dset['data'].copy()
            else:
                data = np.zeros(2 * numPts, 'd')
                data[0:-1:2] = dset['x']
                data[1::2] = dset['data']
        else:
            if isEven:
                data = np.zeros(2 * numPts, 'd')
                data[0:-1:2] = dset['data'].real
                data[1::2] = dset['data'].imag
            else:
                data = np.zeros(3 * numPts, 'd')
                data[0:-2:3] = dset['x']
                data[1:-1:3] = dset['data'].real
                data[2::3] = dset['data'].imag
        # always write data in double precision
        if dset['binary']:
            fh.close()
            if mode.lower() == 'overwrite':
                fh = open(_fileName, 'wb')
            elif mode.lower() == 'add':
                fh = open(_fileName, 'ab')
            # write data
            if bo == 1:
                [fh.write(struct.pack('<d', datai)) for datai in data]
            else:
                [fh.write(struct.pack('>d', datai)) for datai in data]
            fh.close()
            if mode.lower() == 'overwrite':
                fh = open(_fileName, 'wt')
            elif mode.lower() == 'add':
                fh = open(_fileName, 'at')
        else:
            n4Blocks = len(data) // 4
            remVals = len(data) % 4
            if isR:
                if isEven:
                    fh.write(n4Blocks * '%20.11e%20.11e%20.11e%20.11e\n' % tuple(data[:4 * n4Blocks]))
                    if remVals > 0:
                        fh.write((remVals * '%20.11e' + '\n') % tuple(data[4 * n4Blocks:]))
                else:
                    fh.write(n4Blocks * '%13.5e%20.11e%13.5e%20.11e\n' % tuple(data[:4 * n4Blocks]))
                    if remVals > 0:
                        fmt = ['%13.5e', '%20.11e', '%13.5e', '%20.11e']
                        fh.write((''.join(fmt[remVals]) + '\n') % tuple(data[4 * n4Blocks:]))
            else:
                if isEven:
                    fh.write(n4Blocks * '%20.11e%20.11e%20.11e%20.11e\n' % tuple(data[:4 * n4Blocks]))
                    if remVals > 0:
                        fh.write((remVals * '%20.11e' + '\n') % tuple(data[4 * n4Blocks:]))
                else:
                    n3Blocks = len(data) / 3
                    remVals = len(data) % 3
                    # TODO: It breaks here for long measurements. Implement exceptions.
                    # n3Blocks seems to be a natural number but of the wrong type. Convert for now,
                    # but make assertion to prevent werid things from happening.
                    if float(n3Blocks - int(n3Blocks)) != 0.0:
                        print('Warning: Something went wrong when savning the uff file.')
                    n3Blocks = int(n3Blocks)
                    fh.write(n3Blocks * '%13.5e%20.11e%20.11e\n' % tuple(data[:3 * n3Blocks]))
                    if remVals > 0:
                        fmt = ['%13.5e', '%20.11e', '%20.11e']
                        fh.write((''.join(fmt[remVals]) + '\n') % tuple(data[3 * n3Blocks:]))
        fh.write('%6i\n' % -1)
        del data
    except KeyError as msg:
        raise UFFException('The required key \'' + msg.args[0] + '\' not present when writing data-set #58')
    except:
        raise UFFException('Error writing data-set #58')


def _extract58(blockData):
    # Extract function at nodal DOF - data-set 58.
    dset = {'type': 58, 'binary': 0}
    try:
        binary = False
        split_header = b''.join(blockData.splitlines(True)[:13]).decode('utf-8',  errors='replace').splitlines(True)
        if len(split_header[1]) >= 7:
            if split_header[1][6].lower() == 'b':
                # Read some addititional fields from the header section
                binary = True
                dset['binary'] = 1
                dset.update(_parse_header_line(split_header[1], 6, [6, 1, 6, 6, 12, 12, 6, 6, 12, 12],
                                                    [-1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
                                                    ['', '', 'byte_ordering', 'fp_format', 'n_ascii_lines',
                                                        'n_bytes', '', '', '', '']))
        dset.update(_parse_header_line(split_header[2], 1, [80], [1], ['id1']))
        dset.update(_parse_header_line(split_header[3], 1, [80], [1], ['id2']))
        dset.update(_parse_header_line(split_header[4], 1, [80], [1], ['id3']))  # usually for the date
        dset.update(_parse_header_line(split_header[5], 1, [80], [1], ['id4']))
        dset.update(_parse_header_line(split_header[6], 1, [80], [1], ['id5']))
        dset.update(_parse_header_line(split_header[7], 1, [5, 10, 5, 10, 11, 10, 4, 11, 10, 4],
                                            [2, 2, 2, 2, 1, 2, 2, 1, 2, 2],
                                            ['func_type', 'func_id', 'ver_num', 'load_case_id', 'rsp_ent_name',
                                                'rsp_node', 'rsp_dir', 'ref_ent_name',
                                                'ref_node', 'ref_dir']))
        dset.update(_parse_header_line(split_header[8], 6, [10, 10, 10, 13, 13, 13], [2, 2, 2, 3, 3, 3],
                                            ['ord_data_type', 'num_pts', 'abscissa_spacing', 'abscissa_min',
                                                'abscissa_inc', 'z_axis_value']))
        dset.update(_parse_header_line(split_header[9], 4, [10, 5, 5, 5, 21, 21], [2, 2, 2, 2, 1, 1],
                                            ['abscissa_spec_data_type', 'abscissa_len_unit_exp',
                                                'abscissa_force_unit_exp', 'abscissa_temp_unit_exp',
                                                'abscissa_axis_lab', 'abscissa_axis_units_lab']))
        dset.update(_parse_header_line(split_header[10], 4, [10, 5, 5, 5, 21, 21], [2, 2, 2, 2, 1, 1],
                                            ['ordinate_spec_data_type', 'ordinate_len_unit_exp',
                                                'ordinate_force_unit_exp', 'ordinate_temp_unit_exp',
                                                'ordinate_axis_lab', 'ordinate_axis_units_lab']))
        dset.update(_parse_header_line(split_header[11], 4, [10, 5, 5, 5, 21, 21], [2, 2, 2, 2, 1, 1],
                                            ['orddenom_spec_data_type', 'orddenom_len_unit_exp',
                                                'orddenom_force_unit_exp', 'orddenom_temp_unit_exp',
                                                'orddenom_axis_lab', 'orddenom_axis_units_lab']))
        dset.update(_parse_header_line(split_header[12], 4, [10, 5, 5, 5, 21, 21], [2, 2, 2, 2, 1, 1],
                                            ['z_axis_spec_data_type', 'z_axis_len_unit_exp',
                                                'z_axis_force_unit_exp', 'z_axis_temp_unit_exp', 'z_axis_axis_lab',
                                                'z_axis_axis_units_lab']))
        # Body
        # splitData = ''.join(splitData[13:])
        if binary:
            split_data = b''.join(blockData.splitlines(True)[13:])
            if dset['byte_ordering'] == 1:
                bo = '<'
            else:
                bo = '>'
            if (dset['ord_data_type'] == 2) or (dset['ord_data_type'] == 5):
                # single precision - 4 bytes
                values = np.asarray(struct.unpack('%c%sf' % (bo, int(len(split_data) / 4)), split_data), 'd')
            else:
                # double precision - 8 bytes
                values = np.asarray(struct.unpack('%c%sd' % (bo, int(len(split_data) / 8)), split_data), 'd')
        else:
            values = []
            split_data = blockData.decode('utf-8', errors='replace').splitlines(True)[13:]
            if (dset['ord_data_type'] == 2) or (dset['ord_data_type'] == 5):
                for line in split_data[:-1]:  # '6E13.5'
                    values.extend([float(line[13 * i:13 * (i + 1)]) for i in range(len(line) // 13)])
                else:
                    line = split_data[-1]
                    values.extend([float(line[13 * i:13 * (i + 1)]) for i in range(len(line) // 13) if line[13 * i:13 * (i + 1)]!='             '])
            elif ((dset['ord_data_type'] == 4) or (dset['ord_data_type'] == 6)) and (dset['abscissa_spacing'] == 1):
                for line in split_data:  # '4E20.12'
                    values.extend([float(line[20 * i:20 * (i + 1)]) for i in range(len(line) // 20)])
            elif (dset['ord_data_type'] == 4) and (dset['abscissa_spacing'] == 0):
                for line in split_data:  # 2(E13.5,E20.12)
                    values.extend(
                        [float(line[13 * (i + j) + 20 * (i):13 * (i + 1) + 20 * (i + j)]) \
                            for i in range(len(line) // 33) for j in [0, 1]])
            elif (dset['ord_data_type'] == 6) and (dset['abscissa_spacing'] == 0):
                for line in split_data:  # 1E13.5,2E20.12
                    values.extend([float(line[0:13]), float(line[13:33]), float(line[33:53])])
            else:
                raise UFFException('Error reading data-set #58b; not proper data case.')

            values = np.asarray(values)
            # values = np.asarray([float(str) for str in splitData],'d')
        if (dset['ord_data_type'] == 2) or (dset['ord_data_type'] == 4):
            # Non-complex ordinate data
            if (dset['abscissa_spacing'] == 0):
                # Uneven abscissa
                dset['x'] = values[:-1:2].copy()
                dset['data'] = values[1::2].copy()
            else:
                # Even abscissa
                nVal = len(values)
                minVal = dset['abscissa_min']
                d = dset['abscissa_inc']
                dset['x'] = np.arange(minVal, minVal + nVal * d, d)
                dset['data'] = values.copy()
        elif (dset['ord_data_type'] == 5) or (dset['ord_data_type'] == 6):
            # Complex ordinate data
            if (dset['abscissa_spacing'] == 0):
                # Uneven abscissa
                dset['x'] = values[:-2:3].copy()
                dset['data'] = values[1:-1:3] + 1.j * values[2::3]
            else:
                # Even abscissa
                nVal = len(values) / 2
                minVal = dset['abscissa_min']
                d = dset['abscissa_inc']
                dset['x'] = np.arange(minVal, minVal + nVal * d, d)
                dset['data'] = values[0:-1:2] + 1.j * values[1::2]
        del values
    except:
        raise UFFException('Error reading data-set #58b')
    return dset


