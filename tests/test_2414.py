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


def test_read_2414_general():
    # Reading a universal file as exported by Siemens Simcenter3D (tested in release 2023)
    # set 3 is a nodal displacement result.
    uff_ascii = pyuff.UFF('./data/simcenter_exported_result.uff')
    a = uff_ascii.read_sets(3)
    # verify first 10 node numbers
    np.testing.assert_array_equal(a['node_nums'][0:10], np.array(range(1,11)))
    # verify last 10 node numbers
    np.testing.assert_array_equal(a['node_nums'][-10:], np.array(range(11610,11620)))
    # verify displacement for node with number 3
    np.testing.assert_array_equal(a['data_at_node'][2], np.array([-4.29674E-05, -9.05452E-06, -9.67413E-04]))

    # Reading a universal file generated to imported by Simcenter3D (tested in release 2023)
    # set 0 is an elemental result (element thickness).
    uff_ascii = pyuff.UFF('./data/Simcenter_nxopen_thickness.uff')
    a = uff_ascii.read_sets(0)
    # verify first 10 element numbers
    np.testing.assert_array_equal(a['element_nums'][0:10], np.array(range(1,11)))
    # verify last 10 element numbers
    np.testing.assert_array_equal(a['element_nums'][-10:], np.array(range(12072,12082)))
    # verify thickness for element with number 3
    np.testing.assert_array_equal(a['NDVAL'][2], np.array([1]))
    np.testing.assert_array_equal(a['data_at_element'][2], np.array([18]))

    # set 1 is a data at nodes on elements result (element thickness at nodes on elements).
    a = uff_ascii.read_sets(1)
    # verify first 10 element numbers
    np.testing.assert_array_equal(a['element_nums'][0:10], np.array(range(1,11)))
    # verify last 10 element numbers
    np.testing.assert_array_equal(a['element_nums'][-10:], np.array(range(12072,12082)))
    # verify thickness for element with number 3
    np.testing.assert_array_equal(a['IEXP'][2], np.array([2]))
    np.testing.assert_array_equal(a['number_of_nodes'][2], np.array([4]))
    np.testing.assert_array_equal(a['number_of_values_per_node'][2], np.array([1]))
    # note that here a list of array is returned, because of how record 15 is created
    np.testing.assert_array_equal(a['data_at_nodes_on_element'][2], np.array([[18]]))


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
    if os.path.exists(uff_ascii._filename):
        os.remove(uff_ascii._filename)
    # Test
    np.testing.assert_array_equal(a['frequency'], b['frequency'])
    np.testing.assert_array_equal(a['z'], b['z'])


def test_write_2414_general():
    # Reading a universal file generated to imported by Simcenter3D (tested in release 2023)
    # set 1 is a data at nodes on elements result (element thickness at nodes on elements).
    uff_ascii = pyuff.UFF('./data/Simcenter_nxopen_thickness.uff')
    a = uff_ascii.read_sets(1)

    # Write dataset 2414
    uff_write = pyuff.UFF('./data/tmp.uff')
    uff_write._write_set(a,'overwrite')

    # Read dataset 2414 in written file
    uff_ascii = pyuff.UFF('./data/tmp.uff')
    b = uff_ascii.read_sets(0) # only wrote the 1 dataset, so now index 0
    if os.path.exists(uff_ascii._filename):
       os.remove(uff_ascii._filename)

    # Test
    np.testing.assert_array_equal(a['element_nums'], b['element_nums'])
    np.testing.assert_array_equal(a['IEXP'], b['IEXP'])
    np.testing.assert_array_equal(a['number_of_nodes'], b['number_of_nodes'])
    np.testing.assert_array_equal(a['number_of_values_per_node'], b['number_of_values_per_node'])
    np.testing.assert_array_equal(a['data_at_nodes_on_element'], b['data_at_nodes_on_element'])


def test_write_2414_data_at_node():
    # Reading a universal file generated from unknown FEM-software
    uff_ascii = pyuff.UFF('./data/heat_engine_housing.uff')
    original_data_sets = uff_ascii.read_sets()
    # model_info, unit_info, node_info, element_info, temperature_of_nodes

    # Write dataset 2414
    uff_write = pyuff.UFF('./data/tmp.uff')
    uff_write.write_sets(original_data_sets, 'overwrite')

    # Read dataset 2414 in written file
    uff_ascii = pyuff.UFF('./data/tmp.uff')
    data_sets_after_write = uff_ascii.read_sets() # only wrote the 1 dataset, so now index 0
    if os.path.exists(uff_ascii._filename):
       os.remove(uff_ascii._filename)

    temp_orig = original_data_sets[4]
    temp_write = data_sets_after_write[4]
    # Test
    np.testing.assert_array_equal(temp_orig['node_nums'], temp_write['node_nums'])
    np.testing.assert_array_equal(temp_orig['data_at_node'], temp_write['data_at_node'])


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
                'real_part_of_modal_A_or_modal_mass',
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
    test_write_2414()
    test_read_2414_general()
    test_write_2414_general()
    test_write_2414_data_at_node()
