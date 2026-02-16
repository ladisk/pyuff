import numpy as np

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none

def get_structure_1858(raw=False):
    """(source: https://www.ceas3.uc.edu/sdrluff/"""
    out = """
Universal Dataset Number: 1858

Name:   Dataset 58 qualifiers
----------------------------------------------------------------------------
 
Record 1:     FORMAT(6I12)
              Field 1       - Set record number
              Field 2       - Octave format
                              0 - not in octave format (default)
                              1 - octave
                              3 - one third octave
                              n - 1/n octave
              Field 3       - Measurement run number
              Fields 4-6    - Not used (fill with zeros)

Record 2:     FORMAT(12I6)
              Field 1       - Weighting Type
                              0 - No weighting or Unknown (default)
                              1 - A weighting
                              2 - B weighting
                              3 - C weighting
                              4 - D weighting (not yet implemented)
              Field 2       - Window Type
                              0 - No window or unknown (default) 
                              1 - Hanning Narrow
                              2 - Hanning Broad
                              3 - Flattop
                              4 - Exponential
                              5 - Impact
                              6 - Impact and Exponential 
              Field 3       - Amplitude units
                              0 - unknown (default)
                              1 - Half-peak scale
                              2 - Peak scale
                              3 - RMS
              Field 4       - Normalization Method
                              0 - unknown (default)
                              1 - Units squared
                              2 - Units squared per Hz (PSD)
                              3 - Units squared seconds per Hz (ESD)
              Field 5       - Abscissa Data Type Qualifier
                              0 - Translation
                              1 - Rotation
                              2 - Translation Squared
                              3 - Rotation Squared
              Field 6       - Ordinate Numerator Data Type Qualifier
                              0 - Translation
                              1 - Rotation
                              2 - Translation Squared
                              3 - Rotation Squared
              Field 7       - Ordinate Denominator Data Type Qualifier
                              0 - Translation
                              1 - Rotation
                              2 - Translation Squared
                              3 - Rotation Squared
              Field 8       - Z-axis Data Type Qualifier
                              0 - Translation
                              1 - Rotation
                              2 - Translation Squared
                              3 - Rotation Squared

              Field 9       - Sampling Type
                              0 - Dynamic
                              1 - Static
                              2 - RPM from Tach
                              3 - Frequency from tach
              Fields 10-12  - not used (fill with zeros)
         
Record 3:     FORMAT  (1P5E15.7)
              Field 1       - Z RPM value
              Field 2       - Z Time value
              Field 3       - Z Order value
              Field 4       - Number of samples
              Field 5       - not used (fill with zero)
      
Record 4:     FORMAT  (1P5E15.7)
              Field 1       - User value 1
              Field 2       - User value 2
              Field 3       - User value 3
              Field 4       - User value 4
              Field 5       - Exponential window damping factor

Record 5:     FORMAT  (1P5E15.7)
              Fields 1-5    - not used (fill with zeros)

Record 6:     FORMAT  (2A2,2X,2A2)
              Field 1       - Response direction
              Field 2       - Reference direction
 
Record 7:     FORMAT  (40A2)
              Field 1       - not used

----------------------------------------------------------------------------
"""

    if raw:
        return out
    else:
        print(out)   

def _write1858(fh, dset):
    try:
        dict = {}

        dset = _opt_fields(dset, dict)
        fh.write('%6i\n%6i%74s\n' % (-1, 1858, ' '))

        fh.write('%10i%10i%10i%10i%10i%10i\n' % (
            dset['record_num'],
            dset['octave_format'],
            dset['measurement_run'],
            0,
            0,
            0,
        ))

        fh.write('%6i%6i%6i%6i%6i%6i%6i%6i%6i%6i%6i%6i\n' % (
            dset['weighting_type'],
            dset['window_type'],
            dset['amplitude_units'],
            dset['normalization_method'],
            dset['abscissa_data_type_qualifier'],
            dset['ordinate_numerator_data_type_qualifier'],
            dset['ordinate_denominator_data_type_qualifier'],
            dset['z_axis_data_type_qualifier'],
            dset['sampling_type'],
            0,
            0,
            0,
        ))

        fh.write('%15.7e%15.7e%15.7e%15.7e%15.7e\n' % (
            dset['z_rpm_value'],
            dset['z_time_value'],
            dset['z_order_value'],
            dset['num_of_samples'],
            0,
        ))

        fh.write('%15.7e%15.7e%15.7e%15.7e%15.7e\n' % (
            dset['user_value_1'],
            dset['user_value_2'],
            dset['user_value_3'],
            dset['user_value_4'],
            dset['exponential_window_damping_factor'],
        ))

        fh.write('%15.7e%15.7e%15.7e%15.7e%15.7e\n' % (0, 0, 0, 0, 0))

        fh.write('%-4s%2s%-4s\n' % (
            dset['response_direction'],
            '',
            dset['reference_direction'],
        ))

        fh.write('%-80s\n' % 'NONE')
        fh.write('%6i\n' % -1)

    except:
        raise Exception('Error writing data-set #1858')


def _extract1858(block_data):
    """Extract coordinate data - data-set 1858."""
    dset = {'type': 1858}
    try:
        split_header = block_data.splitlines(True)[:8]

        dset.update(_parse_header_line(
            split_header[2],
            6,
            [12, 12, 12, 12, 12, 12],
            [2, 2, 2, -1, -1, -1],
            [
                'record_num',
                'octave_format',
                'measurement_run',
                '',
                '',
                '',
            ]
        ))
        dset.update(_parse_header_line(
            split_header[3],
            12,
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, -1, -1, -1],
            [
                'weighting_type',
                'window_type',
                'amplitude_units',
                'normalization_method',
                'abscissa_data_type_qualifier',
                'ordinate_numerator_data_type_qualifier',
                'ordinate_denominator_data_type_qualifier',
                'z_axis_data_type_qualifier',
                'sampling_type',
                '',
                '',
                '',
            ]
        ))
        dset.update(_parse_header_line(
            split_header[4],
            5,
            [15, 15, 15, 15, 15],
            [3, 3, 3, 3, -1],
            [
                'z_rpm_value',
                'z_time_value',
                'z_order_value',
                'num_of_samples',
                '',
            ]
        ))
        dset.update(_parse_header_line(
            split_header[5],
            5,
            [15, 15, 15, 15, 15],
            [3, 3, 3, 3, 3],
            [
                'user_value_1',
                'user_value_2',
                'user_value_3',
                'user_value_4',
                'exponential_window_damping_factor',
            ]
        ))
        # record 5 is not used
        dset.update(_parse_header_line(
            split_header[7],
            3,
            [4, 2, 4],
            [1, -1, 1],
            [
                'response_direction',
                '',
                'reference_direction',
            ]
        ))
        # record 7 is not used
    except:
        raise Exception('Error reading data-set #1858')
    return dset


def prepare_1858(
        record_num=None,
        octave_format=None,
        measurement_run=None,
        weighting_type=None,
        window_type=None,
        amplitude_units=None,
        normalization_method=None,
        abscissa_data_type_qualifier=None,
        ordinate_numerator_data_type_qualifier=None,
        ordinate_denominator_data_type_qualifier=None,
        z_axis_data_type_qualifier=None,
        sampling_type=None,
        z_rpm_value=None,
        z_time_value=None,
        z_order_value=None,
        num_of_samples=None,
        user_value_1=None,
        user_value_2=None,
        user_value_3=None,
        user_value_4=None,
        exponential_window_damping_factor=None,
        response_direction=None,
        reference_direction=None,
        return_full_dict=False):
    """Name: Dataset 58 qualifiers

    R-Record, F-Field
    
    :param record_num: R1 F1, Record number
    :param octave_format: R1 F2, Octave format
    :param measurement_run: R1 F3, Measurement run number
    :param weighting_type: R2 F1, Weighting Type
    :param window_type: R2 F2, Window Type
    :param amplitude_units: R2 F3, Amplitude units
    :param normalization_method: R2 F4, Normalization Method
    :param abscissa_data_type_qualifier: R2 F5, Abscissa Data Type Qualifier
    :param ordinate_numerator_data_type_qualifier: R2 F6, Ordinate Numerator Data Type Qualifier
    :param ordinate_denominator_data_type_qualifier: R2 F7, Ordinate Denominator Data Type Qualifier
    :param z_axis_data_type_qualifier: R2 F8, Z-axis Data Type Qualifier
    :param sampling_type: R2 F9, Sampling Type
    :param z_rpm_value: R3 F1, Z RPM value
    :param z_time_value: R3 F2, Z Time value
    :param z_order_value: R3 F3, Z Order value
    :param num_of_samples: R3 F4, Number of samples
    :param user_value_1: R4 F1, User value 1
    :param user_value_2: R4 F2, User value 2
    :param user_value_3: R4 F3, User value 3
    :param user_value_4: R4 F4, User value 4
    :param exponential_window_damping_factor: R F,
    :param response_direction: R6 F1, Response direction
    :param reference_direction: R6 F2, Reference direction
    :param return_full_dict: If True full dict with all keys is returned, else only specified arguments are included
    """

    if np.array(record_num).dtype != int and record_num is not None:
        raise TypeError('record_num must be int')
    if np.array(octave_format).dtype != int and octave_format is not None:
        raise TypeError('octave_format must be int')
    if np.array(measurement_run).dtype != int and measurement_run is not None:
        raise TypeError('measurement_run must be int')
    if np.array(weighting_type).dtype != int and weighting_type is not None:
        raise TypeError('weighting_type must be int')
    if np.array(window_type).dtype != int and window_type is not None:
        raise TypeError('window_type must be int')
    if np.array(amplitude_units).dtype != int and amplitude_units is not None:
        raise TypeError('amplitude_units must be int')
    if np.array(normalization_method).dtype != int and normalization_method is not None:
        raise TypeError('normalization_method must be int')
    if np.array(abscissa_data_type_qualifier).dtype != int and abscissa_data_type_qualifier is not None:
        raise TypeError('abscissa_data_type_qualifier must be int')
    if (np.array(ordinate_numerator_data_type_qualifier).dtype != int and
            ordinate_numerator_data_type_qualifier is not None):
        raise TypeError('ordinate_numerator_data_type_qualifier must be int')
    if (np.array(ordinate_denominator_data_type_qualifier).dtype != int and
            ordinate_denominator_data_type_qualifier is not None):
        raise TypeError('ordinate_denominator_data_type_qualifier must be int')
    if np.array(z_axis_data_type_qualifier).dtype != int and z_axis_data_type_qualifier is not None:
        raise TypeError('z_axis_data_type_qualifier must be int')
    if np.array(sampling_type).dtype != int and sampling_type is not None:
        raise TypeError('sampling_type must be int')
    if np.array(z_rpm_value).dtype != float and z_rpm_value is not None:
        raise TypeError('z_rpm_value must be float')
    if np.array(z_time_value).dtype != float and z_time_value is not None:
        raise TypeError('z_time_value must be float')
    if np.array(z_order_value).dtype != float and z_order_value is not None:
        raise TypeError('z_order_value must be float')
    if np.array(num_of_samples).dtype != float and num_of_samples is not None:
        raise TypeError('num_of_samples must be float')
    if np.array(user_value_1).dtype != float and user_value_1 is not None:
        raise TypeError('user_value_1 must be float')
    if np.array(user_value_2).dtype != float and user_value_2 is not None:
        raise TypeError('user_value_2 must be float')
    if np.array(user_value_3).dtype != float and user_value_3 is not None:
        raise TypeError('user_value_3 must be float')
    if np.array(user_value_4).dtype != float and user_value_4 is not None:
        raise TypeError('user_value_4 must be float')
    if np.array(exponential_window_damping_factor).dtype != float and exponential_window_damping_factor is not None:
        raise TypeError('exponential_window_damping_factor must be float')
    if not isinstance(response_direction, str) and response_direction is not None:
        raise TypeError('response_direction must be str')
    if not isinstance(reference_direction, str) and reference_direction is not None:
        raise TypeError('reference_direction must be str')

    dataset = {
        'type': 1858,
        'record_num': record_num,
        'octave_format': octave_format,
        'measurement_run': measurement_run,
        'weighting_type': weighting_type,
        'window_type': window_type,
        'amplitude_units': amplitude_units,
        'normalization_method': normalization_method,
        'abscissa_data_type_qualifier': abscissa_data_type_qualifier,
        'ordinate_numerator_data_type_qualifier': ordinate_numerator_data_type_qualifier,
        'ordinate_denominator_data_type_qualifier': ordinate_denominator_data_type_qualifier,
        'z_axis_data_type_qualifier': z_axis_data_type_qualifier,
        'sampling_type': sampling_type,
        'z_rpm_value': z_rpm_value,
        'z_time_value': z_time_value,
        'z_order_value': z_order_value,
        'num_of_samples': num_of_samples,
        'user_value_1': user_value_1,
        'user_value_2': user_value_2,
        'user_value_3': user_value_3,
        'user_value_4': user_value_4,
        'exponential_window_damping_factor': exponential_window_damping_factor,
        'response_direction': response_direction,
        'reference_direction': reference_direction,
    }
    
    if not return_full_dict:
        dataset = check_dict_for_none(dataset)

    return dataset
