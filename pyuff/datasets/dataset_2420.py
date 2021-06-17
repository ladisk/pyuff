import numpy as np

from ..tools import UFFException, _opt_fields, _parse_header_line

# TODO: Big deal - the output dictionary when reading this set
#    is different than the dictionary that is expected (keys) when
#    writing this same set. This is not OK!
def _write2420(fh, dset):
    try:
        dict = {'part_UID': 1,
                'part_name': 'None',
                'cs_type': 0,
                'cs_color': 8}
        dset = _opt_fields(dset, dict)

        fh.write('%6i\n%6i%74s\n' % (-1, 2420, ' '))
        fh.write('%10i\n' % (dset['part_UID']))
        fh.write('%-80s\n' % (dset['part_name']))

        for node in range(len(dset['nodes'])):
            fh.write('%10i%10i%10i\n' % (dset['nodes'][node], dset['cs_type'], dset['cs_color']))
            fh.write('CS%i\n' % dset['nodes'][node])
            fh.write('%25.16e%25.16e%25.16e\n' % tuple(dset['local_cs'][node * 4, :]))
            fh.write('%25.16e%25.16e%25.16e\n' % tuple(dset['local_cs'][node * 4 + 1, :]))
            fh.write('%25.16e%25.16e%25.16e\n' % tuple(dset['local_cs'][node * 4 + 2, :]))
            fh.write('%25.16e%25.16e%25.16e\n' % tuple(dset['local_cs'][node * 4 + 3, :]))
        fh.write('%6i\n' % -1)
    except:
        raise UFFException('Error writing data-set #2420')

def _extract2420(blockData):
    '''Extract local CS/transforms -- data-set 2420.'''
    dset = {'type': 2420}
    #        try:
    splitData = blockData.splitlines(True)

    # -- Get Record 1
    dset['Part_UID'] = float(splitData[2])

    # -- Get Record 2
    dset['Part_Name'] = splitData[3].rstrip()

    # -- Get Record 3
    rec_3 = list(map(int, ''.join(splitData[4::6]).split()))
    dset['CS_sys_labels'] = rec_3[::3]
    dset['CS_types'] = rec_3[1::3]
    dset['CS_colors'] = rec_3[2::3]

    # -- Get Record 4
    dset['CS_names'] = list(map(str.rstrip, splitData[5::6]))

    # !! The following part should be made smoother
    # -- Get Record 5
    row1 = list(map(float, ''.join(splitData[6::6]).split()))
    row2 = list(map(float, ''.join(splitData[7::6]).split()))
    row3 = list(map(float, ''.join(splitData[8::6]).split()))
    # !! Row 4 left out for now - usually zeros ...
    #            row4 = map(float, splitData[7::6].split())
    dset['CS_matrices'] = [np.vstack((row1[i:(i + 3)], row2[i:(i + 3)], row3[i:(i + 3)])) \
                            for i in np.arange(0, len(row1), 3)]
    #        except:
    #            raise UFFException('Error reading data-set #2420')
    return dset