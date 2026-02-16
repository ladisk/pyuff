import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff

def test_1858():
    uff_file = pyuff.UFF('./data/1858.uff')
    uff_file.read_sets()

def test_prepare_1858():
    dict_1858 = pyuff.prepare_1858(return_full_dict=True)

    x = sorted(list(dict_1858.keys()))
    y = sorted([
        'record_num',
        'octave_format',
        'measurement_run',
        'weighting_type',
        'window_type',
        'amplitude_units',
        'normalization_method',
        'abscissa_data_type_qualifier',
        'ordinate_numerator_data_type_qualifier',
        'ordinate_denominator_data_type_qualifier',
        'z_axis_data_type_qualifier',
        'sampling_type',
        'z_rpm_value',
        'z_time_value',
        'z_order_value',
        'num_of_samples',
        'user_value_1',
        'user_value_2',
        'user_value_3',
        'user_value_4',
        'exponential_window_damping_factor',
        'response_direction',
        'reference_direction',
        'type',
    ])
    np.testing.assert_array_equal(x,y)

    #empty dictionary test
    x2=pyuff.prepare_1858()
    if 'type' not in x2.keys():
        raise Exception('Not correct keys')
    if x2['type'] != 1858:
        raise Exception('Not correct type')

def test_write_1858():
    # Read dataset 1858 in test file
    uff_ascii = pyuff.UFF('./data/1858.uff')
    a = uff_ascii.read_sets(0)
    b = uff_ascii.read_sets(1)
    # Test
    np.testing.assert_array_equal(a['measurement_run'], 1)
    np.testing.assert_array_equal(a['window_type'], 4)
    np.testing.assert_array_almost_equal(a['exponential_window_damping_factor'], 0.052706007)
    np.testing.assert_string_equal(a['response_direction'], 'X+')
    np.testing.assert_string_equal(a['reference_direction'], 'X+')

    np.testing.assert_array_equal(b['measurement_run'], 1)
    np.testing.assert_array_equal(b['octave_format'], 3)
    np.testing.assert_string_equal(b['response_direction'], 'NONE')
    np.testing.assert_string_equal(b['reference_direction'], 'NONE')

    # Write dataset 1858
    uff_write = pyuff.UFF('./data/tmp.uff')
    uff_write.write_sets([a, b], 'overwrite')

    # Read dataset 1858 in written file
    uff_ascii = pyuff.UFF('./data/tmp.uff')
    c = uff_ascii.read_sets(0)
    # Test
    np.testing.assert_array_equal(a['measurement_run'], c['measurement_run'])
    np.testing.assert_array_equal(a['window_type'], c['window_type'])
    np.testing.assert_array_almost_equal(
        a['exponential_window_damping_factor'], c['exponential_window_damping_factor']
    )
    np.testing.assert_string_equal(a['response_direction'], c['response_direction'])
    np.testing.assert_string_equal(a['reference_direction'], c['reference_direction'])

    d = uff_ascii.read_sets(1)
    np.testing.assert_array_equal(b['measurement_run'], d['measurement_run'])
    np.testing.assert_array_equal(b['octave_format'], d['octave_format'])
    np.testing.assert_string_equal(b['response_direction'], d['response_direction'])
    np.testing.assert_string_equal(b['reference_direction'], d['reference_direction'])

if __name__ == '__main__':
    test_write_1858()
