import numpy as np

from ..tools import UFFException, _opt_fields, _parse_header_line

def _write2411(fh, dset):
    try:
        dict = {'export_cs_number': 0,
                'cs_color': 8}

        dset = _opt_fields(dset, dict)
        fh.write('%6i\n%6i%74s\n' % (-1, 2411, ' '))

        for node in range(dset['grid_global'].shape[0]):
            fh.write('%10i%10i%10i%10i\n' % (dset['grid_global'][node, 0], dset['export_cs_number'],
                                                dset['grid_global'][node, 0], dset['cs_color']))

            fh.write('%25.16e%25.16e%25.16e\n' % tuple(dset['grid_global'][node, 1:]))

        fh.write('%6i\n' % -1)

    except:
        raise UFFException('Error writing data-set #2411')


def _extract2411(blockData):
    # Extract coordinate data - data-set 15.
    dset = {'type': 15}
    try:
        # Body
        splitData = blockData.splitlines(True)  # Keep the line breaks!
        splitData = ''.join(splitData[2:])  # ..as they are again needed
        splitData = splitData.split()
        values = np.asarray([float(str) for str in splitData], 'd')
        dset['node_nums'] = values[::7].copy()
        dset['def_cs'] = values[1::7].copy()
        dset['disp_cs'] = values[2::7].copy()
        dset['color'] = values[3::7].copy()
        dset['x'] = values[4::7].copy()
        dset['y'] = values[5::7].copy()
        dset['z'] = values[6::7].copy()
    except:
        raise UFFException('Error reading data-set #15')
    return dset