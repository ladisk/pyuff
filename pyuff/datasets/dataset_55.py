import numpy as np

from ..tools import UFFException, _opt_fields, _parse_header_line

def _write55(fh, dset):
    # Writes data at nodes - data-set 55 - to an open file fh. Currently:
    #   - only normal mode (2)
    #   - complex eigenvalue first order (displacement) (3)
    #   - frequency response and (5)
    #   - complex eigenvalue second order (velocity) (7)
    # analyses are supported.
    try:
        # Handle general optional fields
        dset = _opt_fields(dset,
                                {'units_description': ' ',
                                    'id1': 'NONE',
                                    'id2': 'NONE',
                                    'id3': 'NONE',
                                    'id4': 'NONE',
                                    'id5': 'NONE',
                                    'model_type': 1})
        # ... and some data-type specific optional fields
        if dset['analysis_type'] == 2:
            # normal modes
            dset = _opt_fields(dset,
                                    {'modal_m': 0,
                                        'modal_damp_vis': 0,
                                        'modal_damp_his': 0})
        elif dset['analysis_type'] in (3, 7):
            # complex modes
            dset = _opt_fields(dset,
                                    {'modal_b': 0.0 + 0.0j,
                                        'modal_a': 0.0 + 0.0j})
            if not np.iscomplexobj(dset['modal_a']):
                dset['modal_a'] = dset['modal_a'] + 0.j
            if not np.iscomplexobj(dset['modal_b']):
                dset['modal_b'] = dset['modal_b'] + 0.j
        elif dset['analysis_type'] == 5:
            # frequency response
            pass
        else:
            # unsupported analysis type
            raise UFFException('Error writing data-set #55: unsupported analysis type')
        # Some additional checking
        dataType = 2
        #             if dset.has_key('r4') and dset.has_key('r5') and dset.has_key('r6'):
        if ('r4' in dset) and ('r5' in dset) and ('r6' in dset):
            nDataPerNode = 6
        else:
            nDataPerNode = 3
        if np.iscomplexobj(dset['r1']):
            dataType = 5
        else:
            dataType = 2
        # Write strings to the file
        fh.write('%6i\n%6i%74s\n' % (-1, 55, ' '))
        fh.write('%-80s\n' % dset['id1'])
        fh.write('%-80s\n' % dset['id2'])
        fh.write('%-80s\n' % dset['id3'])
        fh.write('%-80s\n' % dset['id4'])
        fh.write('%-80s\n' % dset['id5'])
        fh.write('%10i%10i%10i%10i%10i%10i\n' %
                    (dset['model_type'], dset['analysis_type'], dset['data_ch'],
                    dset['spec_data_type'], dataType, nDataPerNode))
        if dset['analysis_type'] == 2:
            # Normal modes
            fh.write('%10i%10i%10i%10i\n' % (2, 4, dset['load_case'], dset['mode_n']))
            fh.write('%13.5e%13.5e%13.5e%13.5e\n' % (dset['freq'], dset['modal_m'],
                                                        dset['modal_damp_vis'], dset['modal_damp_his']))
        elif dset['analysis_type'] == 5:
            # Frequenc response
            fh.write('%10i%10i%10i%10i\n' % (2, 1, dset['load_case'], dset['freq_step_n']))
            fh.write('%13.5e\n' % dset['freq'])
        elif (dset['analysis_type'] == 3) or (dset['analysis_type'] == 7):
            # Complex modes
            fh.write('%10i%10i%10i%10i\n' % (2, 6, dset['load_case'], dset['mode_n']))
            fh.write('%13.5e%13.5e%13.5e%13.5e%13.5e%13.5e\n' % (
                dset['eig'].real, dset['eig'].imag, dset['modal_a'].real, dset['modal_a'].imag,
                dset['modal_b'].real, dset['modal_b'].imag))
        else:
            raise UFFException('Unsupported analysis type')
        n = len(dset['node_nums'])
        if dataType == 2:
            # Real data
            if nDataPerNode == 3:
                for k in range(0, n):
                    fh.write('%10i\n' % dset['node_nums'][k])
                    fh.write('%13.5e%13.5e%13.5e\n' % (dset['r1'][k], dset['r2'][k], dset['r3'][k]))
            else:
                for k in range(0, n):
                    fh.write('%10i\n' % dset['node_nums'][k])
                    fh.write('%13.5e%13.5e%13.5e%13.5e%13.5e%13.5e\n' %
                                (dset['r1'][k], dset['r2'][k], dset['r3'][k], dset['r4'][k], dset['r5'][k],
                                dset['r6'][k]))
        elif dataType == 5:
            # Complex data; n_data_per_node is assumed being 3
            for k in range(0, n):
                fh.write('%10i\n' % dset['node_nums'][k])
                fh.write('%13.5e%13.5e%13.5e%13.5e%13.5e%13.5e\n' %
                            (dset['r1'][k].real, dset['r1'][k].imag, dset['r2'][k].real, dset['r2'][k].imag,
                            dset['r3'][k].real, dset['r3'][k].imag))
        else:
            raise UFFException('Unsupported data type')
        fh.write('%6i\n' % -1)
    except KeyError as msg:
        raise UFFException('The required key \'' + msg.args[0] + '\' not present when writing data-set #55')
    except:
        raise UFFException('Error writing data-set #55')


def _extract55(blockData):
    # Extract data at nodes - data-set 55. Currently:
    #   - only normal mode (2)
    #   - complex eigenvalue first order (displacement) (3)
    #   - frequency response and (5)
    #   - complex eigenvalue second order (velocity) (7)
    # analyses are supported.
    dset = {'type': 55}
    try:
        splitData = blockData.splitlines(True)
        dset.update(_parse_header_line(splitData[2], 1, [80], [1], ['id1']))
        dset.update(_parse_header_line(splitData[3], 1, [80], [1], ['id2']))
        dset.update(_parse_header_line(splitData[4], 1, [80], [1], ['id3']))
        dset.update(_parse_header_line(splitData[5], 1, [80], [1], ['id4']))
        dset.update(_parse_header_line(splitData[6], 1, [80], [1], ['id5']))
        dset.update(_parse_header_line(splitData[7], 6, [10, 10, 10, 10, 10, 10], [2, 2, 2, 2, 2, 2],
                                            ['model_type', 'analysis_type', 'data_ch', 'spec_data_type',
                                                'data_type', 'n_data_per_node']))
        if dset['analysis_type'] == 2:
            # normal mode
            dset.update(_parse_header_line(splitData[8], 4, [10, 10, 10, 10, 10, 10, 10, 10],
                                                [-1, -1, 2, 2, -1, -1, -1, -1],
                                                ['', '', 'load_case', 'mode_n', '', '', '', '']))
            dset.update(_parse_header_line(splitData[9], 4, [13, 13, 13, 13, 13, 13], [3, 3, 3, 3, -1, -1],
                                                ['freq', 'modal_m', 'modal_damp_vis', 'modal_damp_his', '', '']))
        elif (dset['analysis_type'] == 3) or (dset['analysis_type'] == 7):
            # complex eigenvalue
            dset.update(_parse_header_line(splitData[8], 4, [10, 10, 10, 10, 10, 10, 10, 10],
                                                [-1, -1, 2, 2, -1, -1, -1, -1],
                                                ['', '', 'load_case', 'mode_n', '', '', '', '']))
            dset.update(_parse_header_line(splitData[9], 4, [13, 13, 13, 13, 13, 13], [3, 3, 3, 3, 3, 3],
                                                ['eig_r', 'eig_i', 'modal_a_r', 'modal_a_i', 'modal_b_r',
                                                    'modal_b_i']))
            dset.update({'modal_a': dset['modal_a_r'] + 1.j * dset['modal_a_i']})
            dset.update({'modal_b': dset['modal_b_r'] + 1.j * dset['modal_b_i']})
            dset.update({'eig': dset['eig_r'] + 1.j * dset['eig_i']})
            del dset['modal_a_r'], dset['modal_a_i'], dset['modal_b_r'], dset['modal_b_i']
            del dset['eig_r'], dset['eig_i']
        elif dset['analysis_type'] == 5:
            # frequency response
            dset.update(_parse_header_line(splitData[8], 4, [10, 10, 10, 10, 10, 10, 10, 10],
                                                [-1, -1, 2, 2, -1, -1, -1, -1],
                                                ['', '', 'load_case', 'freq_step_n', '', '', '', '']))
            dset.update(_parse_header_line(splitData[9], 1, [13, 13, 13, 13, 13, 13], [3, -1, -1, -1, -1, -1],
                                                ['freq', '', '', '', '', '']))
            # Body
        splitData = ''.join(splitData[10:])
        values = np.asarray([float(str) for str in splitData.split()], 'd')
        if dset['data_type'] == 2:
            # real data
            if dset['n_data_per_node'] == 3:
                dset['node_nums'] = values[:-3:4].copy()
                dset['r1'] = values[1:-2:4].copy()
                dset['r2'] = values[2:-1:4].copy()
                dset['r3'] = values[3::4].copy()
            else:
                dset['node_nums'] = values[:-6:7].copy()
                dset['r1'] = values[1:-5:7].copy()
                dset['r2'] = values[2:-4:7].copy()
                dset['r3'] = values[3:-3:7].copy()
                dset['r4'] = values[4:-2:7].copy()
                dset['r5'] = values[5:-1:7].copy()
                dset['r6'] = values[6::7].copy()
        elif dset['data_type'] == 5:
            # complex data
            if dset['n_data_per_node'] == 3:
                dset['node_nums'] = values[:-6:7].copy()
                dset['r1'] = values[1:-5:7] + 1.j * values[2:-4:7]
                dset['r2'] = values[3:-3:7] + 1.j * values[4:-2:7]
                dset['r3'] = values[5:-1:7] + 1.j * values[6::7]
            else:
                raise UFFException('Cannot handle 6 points per node and complex data when reading data-set #55')
        else:
            raise UFFException('Error reading data-set #55')
    except:
        raise UFFException('Error reading data-set #55')
    del values
    return dset