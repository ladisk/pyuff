import numpy as np

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none, _fit_string

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

        fh.write('%12i%12i%12i%12i%12i%12i\n' % (
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
            _fit_string(dset['response_direction'], 4, 'response_direction', 1858),
            '',
            _fit_string(dset['reference_direction'], 4, 'reference_direction', 1858),
        ))

        fh.write('%-80s\n' % 'NONE')
        fh.write('%6i\n' % -1)

    except:
        raise Exception('Error writing data-set #1858')


def _extract1858(dataset_data):
    """Extract coordinate data - data-set 1858."""
    dset = {'type': 1858}
    try:
        split_header = dataset_data.splitlines(True)[:8]

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
    """Prepare data-set 1858 (data-set 58 qualifiers) for writing.

    Data-set 1858 adds information to the function of the data-set 58 that follows it:
    how the function was acquired and processed, e.g. the window, the frequency
    weighting, the amplitude scaling and normalization of a spectrum, octave bands and
    the number of samples. It is written before the data-set 58 it qualifies and is
    exported by some test data acquisition systems.

    The writer has no default values: all parameters are required for writing (give 0
    for the unused or unknown ones). The integer fields must be Python or NumPy
    integers and the real fields floats; an ``int`` given for a real field raises
    ``TypeError``.

    Parameters
    ----------
    record_num : int
        Set record number (record 1, field 1).
    octave_format : int
        Octave format (record 1, field 2): 0 not in octave format, 1 octave, 3 one-third
        octave, n 1/n octave.
    measurement_run : int
        Measurement run number (record 1, field 3).
    weighting_type : {0, 1, 2, 3, 4}
        Frequency weighting (record 2, field 1): 0 no weighting or unknown, 1 A
        weighting, 2 B weighting, 3 C weighting, 4 D weighting.
    window_type : {0, 1, 2, 3, 4, 5, 6}
        Window applied to the time data (record 2, field 2): 0 no window or unknown,
        1 Hanning narrow, 2 Hanning broad, 3 flattop, 4 exponential, 5 impact, 6 impact
        and exponential.
    amplitude_units : {0, 1, 2, 3}
        Amplitude scaling of the spectrum (record 2, field 3): 0 unknown, 1 half-peak,
        2 peak, 3 RMS.
    normalization_method : {0, 1, 2, 3}
        Normalization of the spectrum (record 2, field 4): 0 unknown, 1 units squared,
        2 units squared per Hz (PSD), 3 units squared seconds per Hz (ESD).
    abscissa_data_type_qualifier : {0, 1, 2, 3}
        Abscissa data type qualifier (record 2, field 5): 0 translation, 1 rotation,
        2 translation squared, 3 rotation squared.
    ordinate_numerator_data_type_qualifier : {0, 1, 2, 3}
        Ordinate numerator data type qualifier, codes as for
        ``abscissa_data_type_qualifier`` (record 2, field 6).
    ordinate_denominator_data_type_qualifier : {0, 1, 2, 3}
        Ordinate denominator data type qualifier, codes as for
        ``abscissa_data_type_qualifier`` (record 2, field 7).
    z_axis_data_type_qualifier : {0, 1, 2, 3}
        Z-axis data type qualifier, codes as for ``abscissa_data_type_qualifier``
        (record 2, field 8).
    sampling_type : {0, 1, 2, 3}
        Sampling type (record 2, field 9): 0 dynamic, 1 static, 2 rpm from tachometer,
        3 frequency from tachometer.
    z_rpm_value : float
        Z-axis rotational speed value, in rpm (record 3, field 1).
    z_time_value : float
        Z-axis time value (record 3, field 2).
    z_order_value : float
        Z-axis order value (record 3, field 3).
    num_of_samples : float
        Number of samples (record 3, field 4).
    user_value_1 : float
        User value 1 (record 4, field 1).
    user_value_2 : float
        User value 2 (record 4, field 2).
    user_value_3 : float
        User value 3 (record 4, field 3).
    user_value_4 : float
        User value 4 (record 4, field 4).
    exponential_window_damping_factor : float
        Damping factor of the exponential window (record 4, field 5).
    response_direction : str
        Response direction as text, e.g. ``'X+'``, or ``'NONE'``, up to 4 characters
        (record 6, field 1).
    reference_direction : str
        Reference direction as text, e.g. ``'X+'``, or ``'NONE'``, up to 4 characters
        (record 6, field 2).
    return_full_dict : bool, optional
        If True, the dictionary has all keys of the data-set, with ``None`` for
        the values not given; if False (default), only the given values.

    Returns
    -------
    dict
        The data-set, to be written with :meth:`pyuff.UFF.write_datasets`.

    Examples
    --------
    Qualifiers of an impact-test FRF (impact and exponential window), followed by the
    FRF itself:

    >>> import numpy as np
    >>> import pyuff
    >>> qualifiers = pyuff.prepare_1858(
    ...     record_num=1,
    ...     octave_format=0,
    ...     measurement_run=1,
    ...     weighting_type=0,
    ...     window_type=6,
    ...     amplitude_units=0,
    ...     normalization_method=0,
    ...     abscissa_data_type_qualifier=0,
    ...     ordinate_numerator_data_type_qualifier=0,
    ...     ordinate_denominator_data_type_qualifier=0,
    ...     z_axis_data_type_qualifier=0,
    ...     sampling_type=0,
    ...     z_rpm_value=0.0,
    ...     z_time_value=0.0,
    ...     z_order_value=0.0,
    ...     num_of_samples=4096.0,
    ...     user_value_1=0.0,
    ...     user_value_2=0.0,
    ...     user_value_3=0.0,
    ...     user_value_4=0.0,
    ...     exponential_window_damping_factor=0.05,
    ...     response_direction='Z+',
    ...     reference_direction='Z+')
    >>> freq = np.arange(0, 501, 1.0)
    >>> frf = pyuff.prepare_58(
    ...     func_type=4, rsp_node=2, rsp_dir=3, ref_node=1, ref_dir=3,
    ...     abscissa_spacing=1, abscissa_spec_data_type=18,
    ...     ordinate_spec_data_type=12, orddenom_spec_data_type=13,
    ...     data=np.ones(len(freq), dtype=complex), x=freq)
    >>> pyuff.UFF('frf_1858.uff').write_datasets([qualifiers, frf], mode='overwrite')
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
