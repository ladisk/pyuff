import numpy as np

from ..tools import _opt_fields, _parse_header_line

def _extract18(block_data):
    '''Extract local CS definitions -- data-set 18.'''
    dset = {'type': 18}
    try:
        split_data = block_data.splitlines()

        # -- Get Record 1
        rec_1 = np.array(list(map(float, ''.join(split_data[2::4]).split())))

        dset['cs_num'] = rec_1[::5]
        # removed - clutter
        # dset['cs_type'] = rec_1[1::5]
        dset['ref_cs_num'] = rec_1[2::5]
        # left out here are the parameters color and definition type

        # -- Get Record 2
        # removed because clutter
        # dset['cs_name'] = split_data[3::4]

        # -- Get Record 31 and 32
        # ... these are the origins of cs defined in ref
        #             rec_31 = np.array(list(map(float, ''.join(split_data[4::4]).split())))
        line_data = ''.join(split_data[4::4])
        rec_31 = [float(line_data[i * 13:(i + 1) * 13]) for i in range(int(len(line_data) / 13))]
        dset['ref_o'] = np.vstack((np.array(rec_31[::6]),
                                    np.array(rec_31[1::6]),
                                    np.array(rec_31[2::6]))).transpose()

        # ... these are points on the x axis of cs defined in ref
        dset['x_point'] = np.vstack((np.array(rec_31[3::6]),
                                        np.array(rec_31[4::6]),
                                        np.array(rec_31[5::6]))).transpose()

        # ... these are the points on the xz plane
        line_data = ''.join(split_data[5::4])
        rec_32 = [float(line_data[i * 13:(i + 1) * 13]) for i in range(int(len(line_data) / 13))]
        #             rec_32 = np.array(list(map(float, ''.join(split_data[5::4]).split())))
        dset['xz_point'] = np.vstack((np.array(rec_32[::3]),
                                        np.array(rec_32[1::3]),
                                        np.array(rec_32[2::3]))).transpose()
    except:
        raise Exception('Error reading data-set #18')
    return dset


