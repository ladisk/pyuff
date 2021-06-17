import numpy as np

from ..tools import UFFException, _opt_fields, _parse_header_line


def _write82(fh, dset):
    # Writes line data - data-set 82 - to an open file fh
    try:
        # handle optional fields
        dset = _opt_fields(dset, {'id': 'NONE',
                                        'color': 0})
        # write strings to the file
        # removed jul 2017: unique_nodes = set(dset['nodes'])
        # removed jul 2017:if 0 in unique_nodes: unique_nodes.remove(0)
        # number of changes of node need to
        # nNodes = len(dset['nodes'])
        nNodes = np.sum((dset['nodes'][1:] - dset['nodes'][:-1]) != 0) + 1
        fh.write('%6i\n%6i%74s\n' % (-1, 82, ' '))
        fh.write('%10i%10i%10i\n' % (dset['trace_num'], nNodes, dset['color']))
        fh.write('%-80s\n' % dset['id'])
        sl = 0
        n8Blocks = nNodes // 8
        remLines = nNodes % 8
        if n8Blocks:
            for ii in range(0, n8Blocks):
                #                 fh.write( string.join(['%10i'%lineN for lineN in dset['lines'][sl:sl+8]],'')+'\n' )
                fh.write(''.join(['%10i' % lineN for lineN in dset['nodes'][sl:sl + 8]]) + '\n')
                sl += 8
        if remLines > 0:
            fh.write(''.join(['%10i' % lineN for lineN in dset['nodes'][sl:]]) + '\n')
        #                 fh.write( string.join(['%10i'%lineN for lineN in dset['lines'][sl:]],'')+'\n' )
        fh.write('%6i\n' % -1)
    except KeyError as msg:
        raise UFFException('The required key \'' + msg.args[0] + '\' not present when writing data-set #82')
    except:
        raise UFFException('Error writing data-set #82')


def _extract82(blockData):
    # Extract line data - data-set 82.
    dset = {'type': 82}
    try:
        splitData = blockData.splitlines(True)
        dset.update(
            _parse_header_line(splitData[2], 3, [10, 10, 10], [2, 2, 2], ['trace_num', 'n_nodes', 'color']))
        dset.update(_parse_header_line(splitData[3], 1, [80], [1], ['id']))
        splitData = ''.join(splitData[4:])
        splitData = splitData.split()
        dset['nodes'] = np.asarray([float(str) for str in splitData])
    except:
        raise UFFException('Error reading data-set #82')
    return dset