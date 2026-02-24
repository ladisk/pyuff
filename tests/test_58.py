import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff

def test_read_write_read_given_data():
    test_read_write_read_given_data_base('./data/sample_dataset58_psd.uff', header_only=True)
    test_read_write_read_given_data_base('./data/sample_dataset58_psd.uff', force_double=False)
    test_read_write_read_given_data_base('./data/time-history-not-all-columns-filled.uff')
    test_read_write_read_given_data_base('./data/Artemis export - data and dof 05_14102016_105117.uff')
    test_read_write_read_given_data_base('./data/BK_4_channels.uff')
    test_read_write_read_given_data_base('./data/Sample_UFF58_ascii.uff')
    test_read_write_read_given_data_base('./data/binary8byte.uff')
    data_at_the_end = np.array([-5.48363E-004,-7.51019E-004,-6.07967E-004,-0.00103712])
    test_read_write_read_given_data_base('./data/no_spacing2_UFF58_ascii.uff',data_at_the_end)
    test_read_write_read_given_data_base('./data/sample_dataset58_psd.uff')

def test_read_write_read_given_data_base(file='', data_at_the_end=None, force_double=True, header_only=False):
    if file=='':
        return
    #read from file
    uff_read = pyuff.UFF(file)

    a = uff_read.read_sets()
    if type(a)==list:
        a = [_ for _ in a if _['type']==58]
        a = a[0]

    #write to file
    save_to_file = './data/temp58.uff'
    if os.path.exists(save_to_file):
        os.remove(save_to_file)
    _ = pyuff.UFF(save_to_file)
    _.write_sets(a, 'add', force_double=force_double)

    #read back
    uff_read = pyuff.UFF(save_to_file)
    b = uff_read.read_sets(setn=0, header_only=header_only)

    if os.path.exists(save_to_file):
        os.remove(save_to_file)

    labels = [_ for _ in a.keys() if any(_[-len(w):]==w for w in ['_lab', '_name', '_description'])]
    string_keys = ['id1', 'id2', 'id3', 'id4', 'id5']
    exclude_keys = ['orddenom_spec_data_type', 'abscissa_spec_data_type',
                    'spec_data_type', 'units_description',
                    'version_num']

    string_keys = list(set(string_keys).union(set(labels)).difference(set(exclude_keys)))
    numeric_keys = list((set(a.keys()).difference(set(string_keys)).difference(set(exclude_keys))))

    #print(a['n_bytes'], b['n_bytes'])
    for k in numeric_keys:
        print('Testing: ', k)
        if header_only and k in ['data','x']:
            np.testing.assert_equal(b[k], None)
        else:
            np.testing.assert_array_almost_equal(a[k], b[k], decimal=3)
    for k in string_keys:
        print('Testing string: ', k, a[k])
        np.testing.assert_string_equal(a[k], b[k])


    if header_only:
        pass
    else:
        print('Testing data: ')
        np.testing.assert_array_almost_equal(a['data'], b['data'])

        if data_at_the_end is not None:
            print('Testing last data line: ')
            np.testing.assert_array_almost_equal(a['data'][-len(data_at_the_end):], data_at_the_end)



def test_write_read_58():
    save_to_file = './data/measurement.uff'

    if save_to_file:
        if os.path.exists(save_to_file):
            os.remove(save_to_file)
    uff_datasets = []
    binary = [0, 1, 0]  # ascii of binary
    frequency = np.arange(10)
    np.random.seed(0)
    for i, b in enumerate(binary):
        print('Adding point {}'.format(i + 1))
        response_node = 1
        response_direction = 1
        reference_node = i + 1
        reference_direction = 1
        
        # this is an artificial 'frf'
        acceleration_complex = np.random.normal(size=len(frequency)) + 1j * np.random.normal(size=len(frequency))
        name = 'TestCase'
        data=pyuff.prepare_58(
            binary=binary[i],
            func_type=4,
            rsp_node=response_node,
            rsp_dir=response_direction,
            ref_dir=reference_direction,
            ref_node=reference_node,
            data=acceleration_complex,
            x=frequency,
            id1='id1',
            rsp_ent_name=name,
            ref_ent_name=name,
            abscissa_spacing=1,
            abscissa_spec_data_type=18,
            ordinate_spec_data_type=12,
            orddenom_spec_data_type=13)
        
        uff_datasets.append(data.copy())
        if save_to_file:
            uffwrite = pyuff.UFF(save_to_file)
            uffwrite._write_set(data, 'add')
    
    uff_dataset_origin = uff_datasets
    uff_read = pyuff.UFF(save_to_file)
    uff_dataset_read = uff_read.read_sets()
    if os.path.exists(save_to_file):
        os.remove(save_to_file)

    string_keys = ['id1', 'rsp_ent_name', 'ref_ent_name']
    numeric_keys = list(set(uff_dataset_origin[0].keys()) - set(string_keys))

    for a, b in zip(uff_dataset_origin, uff_dataset_read):
        for k in numeric_keys:
            print('Testing: ', k)
            np.testing.assert_array_almost_equal(a[k], b[k])
        for k in string_keys:
            np.testing.assert_string_equal(a[k], b[k])

def test_read_58b_binary_vs_58_ascii():
    uff_ascii = pyuff.UFF('./data/Sample_UFF58_ascii.uff')
    a = uff_ascii.read_sets(0)
    uff_bin = pyuff.UFF('./data/Sample_UFF58b_bin.uff')
    b = uff_bin.read_sets(0)
    #print(uff_ascii.read_sets(0)['id1'])
    np.testing.assert_string_equal(a['id1'],b['id1'])
    np.testing.assert_array_equal(a['rsp_dir'],b['rsp_dir'])
    np.testing.assert_array_equal(a['x'],b['x'])
    np.testing.assert_array_almost_equal(a['data'],b['data'])

def test_non_ascii_in_header():
    uff_ascii = pyuff.UFF('./data/Artemis export - data and dof 05_14102016_105117.uff')
    a = uff_ascii.read_sets(0)
    np.testing.assert_string_equal(a['id1'], 'Channel 1 [20, -X]')
    np.testing.assert_almost_equal(a['x'][-1], 1.007999609375e+03)
    np.testing.assert_almost_equal(np.sum(a['data']), 172027.83668466809)
    np.testing.assert_string_equal(a['ordinate_axis_units_lab'][:3], 'm/s')
    num_pts = a['num_pts']
    np.testing.assert_equal(num_pts, len(a['x']))

def test_prepare_58():
    uff_datasets = []
    binary = [0, 1, 0]  # ascii of binary
    frequency = np.arange(10)
    np.random.seed(0)
    for i, b in enumerate(binary):
        print('Adding point {}'.format(i + 1))
        response_node = 1
        response_direction = 1
        reference_node = i + 1
        reference_direction = 1
        # this is an artificial 'frf'
        acceleration_complex = np.random.normal(size=len(frequency)) + 1j * np.random.normal(size=len(frequency))
        name = 'TestCase'
        data=pyuff.prepare_58(
            binary=binary[i],
            func_type=4,
            rsp_node=response_node,
            rsp_dir=response_direction,
            ref_dir=reference_direction,
            ref_node=reference_node,
            data=acceleration_complex,
            x=frequency,
            id1='id1',
            rsp_ent_name=name,
            ref_ent_name=name,
            abscissa_spacing=1,
            abscissa_spec_data_type=18,
            ordinate_spec_data_type=12,
            orddenom_spec_data_type=13)

        uff_datasets.append(data.copy())

    x = sorted(list(uff_datasets[0].keys()))
    y=sorted(['type',
            'binary',
            'id1',
            'func_type',
            'rsp_ent_name',
            'rsp_node',
            'rsp_dir',
            'ref_ent_name',
            'ref_node',
            'ref_dir',
            'abscissa_spacing',
            'abscissa_spec_data_type',
            'ordinate_spec_data_type',
            'orddenom_spec_data_type',
            'data',
            'x'])

    np.testing.assert_array_equal(x,y)

    #empty dictionary test
    x2=pyuff.prepare_58()
    if 'type' not in x2.keys():
        raise Exception('Not correct keys')
    if x2['type'] != 58:
        raise Exception('Not correct type')

def test_fix_58b():
    pyuff.fix_58b('./data/MPSTD#Set001_2024_10_08_10_27_07.uff')
    corrected_file = pyuff.UFF('./data/MPSTD#Set001_2024_10_08_10_27_07_fixed.uff')
    data_1 = corrected_file.read_sets(0)

    test_file = pyuff.UFF('./data/MPSTD#Set001_2024_10_08_10_27_07_fixed_test.uff')
    data_2 = test_file.read_sets(0)

    np.testing.assert_array_almost_equal(data_1['data'], data_2['data'])

    # remove the fixed file
    if os.path.exists('./data/MPSTD#Set001_2024_10_08_10_27_07_fixed.uff'):
        os.remove('./data/MPSTD#Set001_2024_10_08_10_27_07_fixed.uff')

def test_write_read_func_type_0():
    """
    Test that func_type=0 is handled correctly in _write58.
    """
    save_to_file = './data/test_func_type_0.uff'

    # Remove the file if it already exists
    if os.path.exists(save_to_file):
        os.remove(save_to_file)

    # Prepare a dataset with func_type=0
    test_data = np.array([1.0, 2.0, 3.0, 4.0])
    test_x = np.array([0.0, 1.0, 2.0, 3.0])
    dataset = pyuff.prepare_58(
        binary=0,
        func_type=0,  # General or unknown data
        rsp_node=0,
        rsp_dir=0,
        ref_node=0,
        ref_dir=0,
        data=test_data,
        x=test_x,
        id1='TestFuncType0',
        rsp_ent_name='TestEntity',
        ref_ent_name='TestEntity',
        abscissa_spacing=1,
        abscissa_spec_data_type=17,  # Time
        ordinate_spec_data_type=2,  # Real
        orddenom_spec_data_type =0
    )

    # Write the dataset to a UFF file
    uff_writer = pyuff.UFF(save_to_file)
    uff_writer.write_sets([dataset], mode='overwrite')

    # Read the dataset back
    uff_reader = pyuff.UFF(save_to_file)
    read_data = uff_reader.read_sets()

    # Clean up the file
    if os.path.exists(save_to_file):
        os.remove(save_to_file)

    # Validate the content
    assert read_data, "Expected one dataset to be read."
    # No need to index; use read_data directly
    read_dataset = read_data

    # Check basic metadata
    np.testing.assert_equal(read_dataset['func_type'], 0)
    np.testing.assert_string_equal(read_dataset['id1'], 'TestFuncType0')
    np.testing.assert_string_equal(read_dataset['rsp_ent_name'], 'TestEntity')
    np.testing.assert_string_equal(read_dataset['ref_ent_name'], 'TestEntity')

    # Validate data content
    np.testing.assert_array_equal(read_dataset['data'], test_data)
    np.testing.assert_array_equal(read_dataset['x'], test_x)

def test_double_uneven():
    test_double_uneven_base()
    test_double_uneven_base(test_data=[1.0,2.0,2.0,4.0], test_x=[20.0,100,300,500])


def test_double_uneven_base(test_data=[1.0,2.0,2.0,4.0,1.0], test_x=[20.0,100,300,500,2000]):
    save_to_file = './data/temp.uff'
    if os.path.exists(save_to_file):
        os.remove(save_to_file)
        
    uffwrite = pyuff.UFF(save_to_file)
    my_set = pyuff.prepare_58(binary=None, id1='my_test', 
                 func_type=9, ver_num=0, load_case_id=0, 
                 rsp_ent_name='CH1', rsp_node=0, rsp_dir=1, 
                 ord_data_type=4, num_pts=5, abscissa_spacing=0, 
                 abscissa_spec_data_type=18,
                 ordinate_spec_data_type=12,
                 data = test_data,
                 x = test_x,
                 return_full_dict=True)

    for key,value in my_set.items():
        if value is None:
            my_set[key] = 0

    uffwrite._write_set(my_set,'add')

    uffread = pyuff.UFF(save_to_file)
    read_data = uffread.read_sets()

    # Clean up the file
    if os.path.exists(save_to_file):
        os.remove(save_to_file)

    # Validate the content
    assert read_data, "Expected one dataset to be read."
    # No need to index; use read_data directly
    read_dataset = read_data

    # Check basic metadata
    np.testing.assert_equal(read_dataset['func_type'], 9)
    np.testing.assert_string_equal(read_dataset['id1'], 'my_test')
    np.testing.assert_string_equal(read_dataset['rsp_ent_name'], 'CH1')
    # Validate data content
    np.testing.assert_array_equal(read_dataset['data'], test_data)
    np.testing.assert_array_equal(read_dataset['x'], test_x)

def test_write_read_58_utf8_headers():
    """Test round-trip with non-ASCII (UTF-8) characters in header fields. Ref: GitHub issues #87, #89."""
    for binary in (0, 1):
        save_to_file = './data/temp58_utf8.uff'
        if os.path.exists(save_to_file):
            os.remove(save_to_file)

        frequency = np.arange(10, dtype=float)
        data = np.random.normal(size=len(frequency)) + 1j * np.random.normal(size=len(frequency))

        dataset = pyuff.prepare_58(
            binary=binary,
            func_type=4,
            rsp_node=1,
            rsp_dir=1,
            ref_dir=1,
            ref_node=1,
            data=data,
            x=frequency,
            id1='Time for 车速',
            id2='Ünit tëst àcc',
            id3='Données résultat',
            rsp_ent_name='Tëst',
            ref_ent_name='Rëf',
            abscissa_spacing=1,
            abscissa_spec_data_type=18,
            ordinate_spec_data_type=12,
            orddenom_spec_data_type=13)

        uffwrite = pyuff.UFF(save_to_file)
        uffwrite._write_set(dataset, 'add')

        uff_read = pyuff.UFF(save_to_file)
        b = uff_read.read_sets(0)

        if os.path.exists(save_to_file):
            os.remove(save_to_file)

        np.testing.assert_string_equal(b['id1'], 'Time for 车速')
        np.testing.assert_string_equal(b['id2'], 'Ünit tëst àcc')
        np.testing.assert_string_equal(b['id3'], 'Données résultat')
        np.testing.assert_string_equal(b['rsp_ent_name'], 'Tëst')
        np.testing.assert_string_equal(b['ref_ent_name'], 'Rëf')


if __name__ == '__main__':
    test_double_uneven(test_data=[1.0,2.0,2.0,4.0,1.0], test_x=[20.0,100,300,500,2000])
    test_double_uneven(test_data=[1.0,2.0,2.0,4.0], test_x=[20.0,100,300,500])

if __name__ == '__mains__':
    np.testing.run_module_suite()