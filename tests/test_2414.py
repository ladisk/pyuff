# -*- coding: utf-8 -*-
import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff

def test_read_2414():
    uff_ascii = pyuff.UFF('./data/DS2414_disp_file.uff')
    a = uff_ascii.read_sets(3)
    np.testing.assert_array_equal(a['frequency'], np.array([100]))
    np.testing.assert_array_equal(a['x'][3], np.array([3.33652e-09-1j*9.17913e-13]))

def test_write_2414():
    # Read dataset 2414 in test file
    uff_ascii = pyuff.UFF('./data/DS2414_disp_file.uff')
    a = uff_ascii.read_sets(3)
    # Test
    np.testing.assert_array_equal(a['frequency'], np.array([100]))
    np.testing.assert_array_equal(a['x'][3], np.array([3.33652e-09-1j*9.17913e-13]))
    
    # Write dataset 2414
    uff_write = pyuff.UFF('./data/tmp.uff')
    uff_write._write_set(a,'overwrite')

    # Read dataset 2414 in written file
    uff_ascii = pyuff.UFF('./data/tmp.uff')
    b = uff_ascii.read_sets(0)
    # Test
    np.testing.assert_array_equal(a['frequency'], b['frequency'])
    np.testing.assert_array_equal(a['z'], b['z'])

def test_prepare_2414():
    dict_2414 = pyuff.prepare_2414(return_full_dict=True)

    x = sorted(list(dict_2414.keys()))
    y = sorted(['type',
                'analysis_dataset_label',
                'analysis_dataset_name',
                'dataset_location',
                'id1',
                'id2',
                'id3',
                'id4',
                'id5',
                'model_type',
                'analysis_type',
                'data_characteristic',
                'result_type',
                'data_type',
                'number_of_data_values_for_the_data_component',
                'design_set_id',
                'iteration_number',
                'solution_set_id',
                'boundary_condition',
                'load_set',
                'mode_number',
                'time_step_number',
                'frequency_number',
                'creation_option',
                'number_retained',
                'time',
                'frequency',
                'eigenvalue',
                'modal_mass',
                'viscous_damping',
                'hysteretic_damping',
                'real_part_eigenvalue',
                'imaginary_part_eigenvalue',
                'real_part_of_modal_A_or_modal_mass,',
                'imaginary_part_of_modal_A_or_modal_mass',
                'real_part_of_modal_B_or_modal_mass',
                'imaginary_part_of_modal_B_or_modal_mass',
                'd',
                'node_nums',
                'x',
                'y',
                'z'])

    np.testing.assert_array_equal(x,y)

    #empty dictionary test
    x2=pyuff.prepare_2414()
    if 'type' not in x2.keys():
        raise Exception('Not correct keys')
    if x2['type'] != 2414:
        raise Exception('Not correct type')

if __name__ == '__main__':
    #test_read_2412()
    test_write_2414()